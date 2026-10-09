r"""
evaluate_vs_rois.py - Score a predicted mitochondria label image against a hand-drawn ImageJ ROI set.

  python evaluate_vs_rois.py --pred "mito_seg\img_mito_labels.tif" --gt-rois RoiSet.zip --image img.tif --outdir eval

Reports
  * object level: precision / recall / F1 at IoU 0.3, 0.5, 0.75, mean IoU of matches, panoptic quality
  * pixel level: Dice and IoU of the mitochondria foreground
  * over-/under-segmentation: ground-truth objects split into several predictions, and predictions
    that merge several ground-truth objects
  * where false positives fall: inside vs. outside the region covered by ground-truth annotations
    (useful when only part of an image was annotated)
Writes metrics.json, summary.md, matches.csv, missed_gt.csv, false_positives.csv, overview.png and
crops/*.png (green = matched GT, yellow = missed GT, magenta = matched prediction, red = false positive).
"""
import argparse
import json
import os

import numpy as np
import tifffile


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred', required=True, help='predicted label image (.tif)')
    p.add_argument('--gt-rois', required=True, help='ImageJ RoiSet.zip (or single .roi) with ground-truth outlines')
    p.add_argument('--image', default=None, help='original image, for overlays')
    p.add_argument('--outdir', default=None, help=r'output folder (default: <pred folder>\eval_<pred name>)')
    p.add_argument('--iou-thr', type=float, default=0.5, help='IoU threshold for the headline numbers (default 0.5)')
    p.add_argument('--crop-size', type=int, default=2048, help='size of native-resolution QC crops (default 2048)')
    p.add_argument('--n-crops', type=int, default=6, help='number of QC crops (default 6)')
    p.add_argument('--cell', type=int, default=2048, help='grid cell size used to define the annotated region')
    p.add_argument('--region-dilate', type=int, default=0,
                   help='if > 0, the annotated region is everything within this many pixels of a GT object '
                        '(for partially annotated images) instead of grid cells containing GT')
    p.add_argument('--label', default='', help='free-text label stored in metrics.json (e.g. parameter set)')
    p.add_argument('--no-overlays', action='store_true', help='skip PNG overlays')
    p.add_argument('--roi-offset', type=int, nargs=2, default=(0, 0), metavar=('Y', 'X'),
                   help='origin of the predicted image inside the ROI coordinate system (when --pred is a crop)')
    return p.parse_args()


# ----------------------------------------------------------------------------- ground truth
def load_gt_rois(path, shape, offset=(0, 0)):
    """Rasterise ImageJ ROIs into a label image of `shape`; `offset` = (y, x) of the image origin in ROI coordinates."""
    import roifile
    from skimage.draw import polygon as draw_polygon, ellipse as draw_ellipse

    rois = roifile.roiread(path)
    if not isinstance(rois, list):
        rois = [rois]
    gt = np.zeros(shape, dtype=np.int32)
    info, skipped = [], []
    overlap_px = 0
    oy, ox = offset
    for i, roi in enumerate(rois, start=1):
        rt = roi.roitype
        rr = cc = None
        try:
            if rt == roifile.ROI_TYPE.OVAL:
                cy = (roi.top + roi.bottom) / 2.0 - 0.5 - oy
                cx = (roi.left + roi.right) / 2.0 - 0.5 - ox
                rr, cc = draw_ellipse(cy, cx, (roi.bottom - roi.top) / 2.0, (roi.right - roi.left) / 2.0, shape=shape)
            elif rt == roifile.ROI_TYPE.RECT:
                rr, cc = np.mgrid[roi.top - oy:roi.bottom - oy, roi.left - ox:roi.right - ox]
                rr, cc = rr.ravel(), cc.ravel()
                keep = (rr >= 0) & (cc >= 0) & (rr < shape[0]) & (cc < shape[1])
                rr, cc = rr[keep], cc[keep]
            else:
                coords = np.asarray(roi.coordinates(), dtype=np.float64)
                if coords.ndim != 2 or len(coords) < 3:
                    raise ValueError(f'{len(coords) if coords.ndim == 2 else 0} vertices')
                # ImageJ vertices sit on pixel corners; skimage tests pixel centres -> shift half a pixel
                rr, cc = draw_polygon(coords[:, 1] - 0.5 - oy, coords[:, 0] - 0.5 - ox, shape=shape)
        except Exception as e:
            skipped.append({'index': i, 'name': roi.name, 'type': str(rt), 'reason': str(e)})
            continue
        if rr is None or len(rr) == 0:
            skipped.append({'index': i, 'name': roi.name, 'type': str(rt), 'reason': 'empty'})
            continue
        overlap_px += int((gt[rr, cc] != 0).sum())
        gt[rr, cc] = i
        info.append({'gt_id': i, 'name': roi.name, 'type': str(rt).split('.')[-1], 'n_px_drawn': int(len(rr))})
    return gt, info, skipped, overlap_px


# ----------------------------------------------------------------------------- matching
def contingency(gt, pred, chunk_rows=512):
    """Pixel counts for every (gt_id, pred_id) pair where at least one id is non-zero."""
    K = int(pred.max()) + 1
    keys, counts = [], []
    for r0 in range(0, gt.shape[0], chunk_rows):
        g = gt[r0:r0 + chunk_rows].ravel().astype(np.int64)
        p = pred[r0:r0 + chunk_rows].ravel().astype(np.int64)
        m = (g > 0) | (p > 0)
        if not m.any():
            continue
        u, c = np.unique(g[m] * K + p[m], return_counts=True)
        keys.append(u)
        counts.append(c)
    keys = np.concatenate(keys)
    counts = np.concatenate(counts)
    u, inv = np.unique(keys, return_inverse=True)
    tot = np.bincount(inv, weights=counts).astype(np.int64)
    return u // K, u % K, tot, K


def greedy_match(gi, pi, iou):
    order = np.argsort(-iou, kind='stable')
    used_g, used_p, matches = set(), set(), []
    for k in order:
        g, p = int(gi[k]), int(pi[k])
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        matches.append((g, p, float(iou[k])))
    return matches


def centroids(lab):
    from scipy import ndimage as ndi
    n = int(lab.max())
    if n == 0:
        return np.zeros((0, 2))
    cms = ndi.center_of_mass(lab > 0, lab, np.arange(1, n + 1))
    return np.array(cms, dtype=float)


# ----------------------------------------------------------------------------- overlays
def stretch(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (0.5, 99.5))
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def draw_outlines(rgb, lab, ids, colour, thick=True):
    from skimage.segmentation import find_boundaries
    if len(ids) == 0:
        return
    sub = np.where(np.isin(lab, ids), lab, 0)
    rgb[find_boundaries(sub, mode='thick' if thick else 'inner')] = colour


def write_png(path, rgb):
    import cv2
    cv2.imwrite(path, rgb[..., ::-1])


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    pred_path = os.path.abspath(args.pred)
    outdir = args.outdir or os.path.join(os.path.dirname(pred_path),
                                         'eval_' + os.path.splitext(os.path.basename(pred_path))[0])
    os.makedirs(os.path.join(outdir, 'crops'), exist_ok=True)

    from scipy import ndimage as ndi
    from skimage.segmentation import relabel_sequential
    import pandas as pd

    print('loading prediction...')
    pred = tifffile.imread(pred_path)
    if pred.ndim != 2:
        raise SystemExit(f'expected a 2D label image, got {pred.shape}')
    pred, _, _ = relabel_sequential(pred.astype(np.int32, copy=False))
    n_pred = int(pred.max())

    print('rasterising ground-truth ROIs...')
    gt, gt_info, skipped, overlap_px = load_gt_rois(args.gt_rois, pred.shape, offset=tuple(args.roi_offset))
    gt, _, _ = relabel_sequential(gt)
    n_gt = int(gt.max())
    print(f'   {n_gt} ground-truth objects ({len(skipped)} ROIs skipped, {overlap_px} px of overlapping ROIs)')

    print('computing overlaps...')
    g_ids, p_ids, tot, K = contingency(gt, pred)
    gt_area = np.bincount(g_ids, weights=tot, minlength=n_gt + 1)
    pred_area = np.bincount(p_ids, weights=tot, minlength=n_pred + 1)
    both = (g_ids > 0) & (p_ids > 0)
    gi, pi, inter = g_ids[both], p_ids[both], tot[both].astype(np.float64)
    iou = inter / np.maximum(gt_area[gi] + pred_area[pi] - inter, 1)
    matches = greedy_match(gi, pi, iou)

    # pixel level
    fg_gt, fg_pred, fg_inter = float(gt_area[1:].sum()), float(pred_area[1:].sum()), float(inter.sum())
    dice = 2 * fg_inter / max(fg_gt + fg_pred, 1)
    pix_iou = fg_inter / max(fg_gt + fg_pred - fg_inter, 1)
    pix_recall = fg_inter / max(fg_gt, 1)
    pix_precision = fg_inter / max(fg_pred, 1)

    # coverage, splits, merges
    cov = np.zeros(n_gt + 1)
    np.add.at(cov, gi, inter)
    cov = cov / np.maximum(gt_area, 1)
    substantial = (inter / np.maximum(gt_area[gi], 1)) >= 0.2
    splits = np.bincount(gi[substantial], minlength=n_gt + 1)
    merges = np.bincount(pi[substantial], minlength=n_pred + 1)
    n_split_gt = int((splits >= 2).sum())
    n_merge_pred = int((merges >= 2).sum())

    # annotated region: either grid cells touched by any GT object, or (--region-dilate) everything
    # within that distance of a GT object
    cell = args.cell
    H, W = gt.shape
    gt_c, pred_c = centroids(gt), centroids(pred)
    gt_objs = ndi.find_objects(gt)
    if args.region_dilate > 0:
        f = 8
        gs = gt[::f, ::f] > 0
        annotated = ndi.distance_transform_edt(~gs) * f <= args.region_dilate
        region_def = f'within {args.region_dilate} px of a GT object'
        pred_in_region = np.array([annotated[min(int(y) // f, annotated.shape[0] - 1),
                                             min(int(x) // f, annotated.shape[1] - 1)] for y, x in pred_c],
                                  dtype=bool) if n_pred else np.zeros(0, bool)
    else:
        annotated = np.zeros(((H + cell - 1) // cell, (W + cell - 1) // cell), bool)
        for slc in gt_objs:
            if slc is not None:
                annotated[slc[0].start // cell:(slc[0].stop - 1) // cell + 1,
                          slc[1].start // cell:(slc[1].stop - 1) // cell + 1] = True
        region_def = f'{cell} px grid cells containing a GT object'
        pred_in_region = np.array([annotated[int(y) // cell, int(x) // cell] for y, x in pred_c], dtype=bool) \
            if n_pred else np.zeros(0, bool)
    valid = [s for s in gt_objs if s is not None]
    gt_bbox = [min(s[0].start for s in valid), min(s[1].start for s in valid),
               max(s[0].stop for s in valid), max(s[1].stop for s in valid)] if valid else None

    metrics = {
        'label': args.label, 'pred': pred_path, 'gt_rois': os.path.abspath(args.gt_rois),
        'image_shape': [H, W], 'n_gt': n_gt, 'n_pred': n_pred,
        'n_pred_in_annotated_region': int(pred_in_region.sum()),
        'annotated_region_definition': region_def,
        'annotated_region_fraction_of_image': float(annotated.mean()),
        'gt_bbox_y0x0y1x1': gt_bbox, 'gt_rois_skipped': skipped, 'gt_overlap_px': overlap_px,
        'pixel': {'dice': dice, 'iou': pix_iou, 'recall': pix_recall, 'precision': pix_precision,
                  'gt_px': fg_gt, 'pred_px': fg_pred},
        'gt_detected_by_coverage_50pct': float((cov[1:] >= 0.5).mean()) if n_gt else 0.0,
        'n_gt_split': n_split_gt, 'n_pred_merging': n_merge_pred,
        'object': {},
    }
    for thr in (0.3, 0.5, 0.75):
        tp_pairs = [(g, p, v) for g, p, v in matches if v >= thr]
        tp = len(tp_pairs)
        fp, fn = n_pred - tp, n_gt - tp
        tp_p = {p for _, p, _ in tp_pairs}
        fp_in_region = int(sum(1 for p in range(1, n_pred + 1) if p not in tp_p and pred_in_region[p - 1]))
        prec, rec = tp / max(n_pred, 1), tp / max(n_gt, 1)
        prec_region = tp / max(tp + fp_in_region, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1_region = 2 * prec_region * rec / max(prec_region + rec, 1e-9)
        sum_iou = sum(v for _, _, v in tp_pairs)
        metrics['object'][f'iou>={thr}'] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'fp_in_annotated_region': fp_in_region,
            'precision': prec, 'recall': rec, 'f1': f1,
            'precision_in_annotated_region': prec_region, 'f1_in_annotated_region': f1_region,
            'mean_iou_of_matches': (sum_iou / tp) if tp else 0.0,
            'panoptic_quality': sum_iou / max(tp + 0.5 * fp + 0.5 * fn, 1),
            'panoptic_quality_in_annotated_region': sum_iou / max(tp + 0.5 * fp_in_region + 0.5 * fn, 1),
        }

    # tables
    thr = args.iou_thr
    rows = [{'gt_id': g, 'pred_id': p, 'iou': round(v, 4), 'gt_area_px': int(gt_area[g]),
             'pred_area_px': int(pred_area[p]), 'gt_y': round(float(gt_c[g - 1][0]), 1),
             'gt_x': round(float(gt_c[g - 1][1]), 1), 'match': v >= thr} for g, p, v in matches]
    pd.DataFrame(rows).sort_values('iou', ascending=False).to_csv(os.path.join(outdir, 'matches.csv'), index=False)
    matched_g = {g for g, p, v in matches if v >= thr}
    matched_p = {p for g, p, v in matches if v >= thr}
    best_iou_g = {}
    for g, v in zip(gi, iou):
        best_iou_g[int(g)] = max(best_iou_g.get(int(g), 0.0), float(v))
    missed = [{'gt_id': g, 'gt_area_px': int(gt_area[g]), 'y': round(float(gt_c[g - 1][0]), 1),
               'x': round(float(gt_c[g - 1][1]), 1), 'coverage_by_preds': round(float(cov[g]), 3),
               'best_iou': round(best_iou_g.get(g, 0.0), 3), 'n_preds_substantial': int(splits[g])}
              for g in range(1, n_gt + 1) if g not in matched_g]
    pd.DataFrame(missed).to_csv(os.path.join(outdir, 'missed_gt.csv'), index=False)
    fps = [{'pred_id': p, 'pred_area_px': int(pred_area[p]), 'y': round(float(pred_c[p - 1][0]), 1),
            'x': round(float(pred_c[p - 1][1]), 1), 'in_annotated_region': bool(pred_in_region[p - 1]),
            'n_gt_merged': int(merges[p])} for p in range(1, n_pred + 1) if p not in matched_p]
    pd.DataFrame(fps).to_csv(os.path.join(outdir, 'false_positives.csv'), index=False)

    # overlays
    img = None
    if args.image and os.path.isfile(args.image) and not args.no_overlays:
        print('loading image for overlays...')
        img = tifffile.imread(args.image)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., :3].mean(-1)
        if img.shape != gt.shape:
            print(f'   image shape {img.shape} != label shape {gt.shape}; skipping overlays')
            img = None
    if img is not None:
        print('writing overlays...')
        tp_g = np.array(sorted(matched_g), dtype=np.int32)
        fn_g = np.array([g for g in range(1, n_gt + 1) if g not in matched_g], dtype=np.int32)
        tp_p = np.array(sorted(matched_p), dtype=np.int32)
        fp_p = np.array([p for p in range(1, n_pred + 1) if p not in matched_p], dtype=np.int32)

        f = max(1, int(np.ceil(max(H, W) / 4000.0)))
        rgb = np.stack([stretch(img[::f, ::f])] * 3, -1)
        gt_s, pr_s = gt[::f, ::f], pred[::f, ::f]
        draw_outlines(rgb, pr_s, tp_p, (255, 0, 255))
        draw_outlines(rgb, pr_s, fp_p, (255, 40, 40))
        draw_outlines(rgb, gt_s, tp_g, (0, 255, 0))
        draw_outlines(rgb, gt_s, fn_g, (255, 255, 0))
        write_png(os.path.join(outdir, 'overview.png'), rgb)

        cs = args.crop_size
        shape_c = ((H + cs - 1) // cs, (W + cs - 1) // cs)
        dens, fn_d, fp_d = np.zeros(shape_c, int), np.zeros(shape_c, int), np.zeros(shape_c, int)
        for g, (y, x) in enumerate(gt_c, start=1):
            dens[int(y) // cs, int(x) // cs] += 1
            if g not in matched_g:
                fn_d[int(y) // cs, int(x) // cs] += 1
        for p, (y, x) in enumerate(pred_c, start=1):
            if p not in matched_p:
                fp_d[int(y) // cs, int(x) // cs] += 1
        picks = []
        for arr, tag, n in ((dens, 'dense', max(1, args.n_crops - 2)), (fn_d, 'missed', 1), (fp_d, 'falsepos', 1)):
            k = 0
            for flat in np.argsort(-arr.ravel()):
                yy, xx = np.unravel_index(flat, arr.shape)
                if arr[yy, xx] == 0 or any((yy, xx) == (a, b) for a, b, _ in picks):
                    continue
                picks.append((int(yy), int(xx), tag))
                k += 1
                if k >= n:
                    break
        for yy, xx, tag in picks:
            y0, x0 = yy * cs, xx * cs
            sl = (slice(y0, min(y0 + cs, H)), slice(x0, min(x0 + cs, W)))
            rgb = np.stack([stretch(img[sl])] * 3, -1)
            draw_outlines(rgb, pred[sl], tp_p, (255, 0, 255), thick=False)
            draw_outlines(rgb, pred[sl], fp_p, (255, 40, 40), thick=False)
            draw_outlines(rgb, gt[sl], tp_g, (0, 255, 0), thick=False)
            draw_outlines(rgb, gt[sl], fn_g, (255, 255, 0), thick=False)
            write_png(os.path.join(outdir, 'crops', f'{tag}_y{y0}_x{x0}.png'), rgb)

    with open(os.path.join(outdir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2, default=float)

    o = metrics['object'][f'iou>={thr}']
    lines = [
        f'# Evaluation of {os.path.basename(pred_path)}', '',
        f'Ground truth: {n_gt} ROIs from {os.path.basename(args.gt_rois)}.  Prediction: {n_pred} objects, '
        f'{metrics["n_pred_in_annotated_region"]} of them inside the annotated region '
        f'({region_def}; {100 * metrics["annotated_region_fraction_of_image"]:.0f}% of the image).', '',
        f'| metric (IoU >= {thr}) | whole image | annotated region only |', '|---|---|---|',
        f'| true positives | {o["tp"]} | {o["tp"]} |',
        f'| false positives | {o["fp"]} | {o["fp_in_annotated_region"]} |',
        f'| missed (false negatives) | {o["fn"]} | {o["fn"]} |',
        f'| precision | {o["precision"]:.3f} | {o["precision_in_annotated_region"]:.3f} |',
        f'| recall | {o["recall"]:.3f} | {o["recall"]:.3f} |',
        f'| F1 | {o["f1"]:.3f} | {o["f1_in_annotated_region"]:.3f} |',
        f'| panoptic quality | {o["panoptic_quality"]:.3f} | {o["panoptic_quality_in_annotated_region"]:.3f} |',
        f'| mean IoU of matched objects | {o["mean_iou_of_matches"]:.3f} | |', '',
        f'Pixel level: Dice {dice:.3f}, IoU {pix_iou:.3f}, recall {pix_recall:.3f}, precision {pix_precision:.3f}.  ',
        f'GT objects with >= 50% of their area detected: {100 * metrics["gt_detected_by_coverage_50pct"]:.1f}%.  ',
        f'GT objects split across >= 2 predictions: {n_split_gt}.  Predictions merging >= 2 GT objects: {n_merge_pred}.', '',
        'Other thresholds: ' + ', '.join(
            f'IoU>={t}: recall {metrics["object"][f"iou>={t}"]["recall"]:.3f} / '
            f'F1(region) {metrics["object"][f"iou>={t}"]["f1_in_annotated_region"]:.3f}' for t in (0.3, 0.5, 0.75)),
    ]
    with open(os.path.join(outdir, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\noutputs in {outdir}')


if __name__ == '__main__':
    main()
