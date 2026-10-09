r"""
split_by_notches.py - separate fused mitochondria by cutting between matched outline notches.

Idea (from visual inspection of fused objects): where two mitochondria touch, the outline of the fused
object bends inward on BOTH sides of the junction. The correct separation is the short chord between
those two facing notches. Single mitochondria may have a dent on one side (bean shapes) but rarely two
facing dents joined by a chord that is narrower than the parts it would separate.

Per object:
  1. find notches = convexity defects of the (lightly smoothed) outline, with depth >= min_depth
  2. candidate cut = pair of notches whose chord lies inside the object, whose notch directions face each
     other (angle <= max_angle), and which splits the object into two pieces that are each at least
     min_piece_area and wider than the cut (inscribed diameter >= neck_ratio x chord length)
  3. apply the best-scoring cut (deep notches, short chord), then repeat on the resulting pieces
  4. optionally (--single) also cut from a single deep notch straight across to the opposite outline,
     under the same neck / piece-size rules

  python split_by_notches.py --labels seg\img_mito_labels.tif --out seg\img_mito_labels_split.tif [--overlay img.tif]
"""
import argparse
import os

import numpy as np
import tifffile


# ----------------------------------------------------------------------------- geometry helpers
def _outline_notches(mask_ws, min_depth_ws, smooth_sigma):
    """Convexity-defect notches of the largest outline of a binary mask (working scale).
    Returns contour (N,2) [x,y] and a list of dicts: idx, pt (x,y), depth, axis (unit vector pointing inward)."""
    import cv2
    m = mask_ws.astype(np.float32)
    if smooth_sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), smooth_sigma)
    m = (m > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None, []
    cnt = max(cs, key=cv2.contourArea)
    if len(cnt) < 10:
        return cnt.reshape(-1, 2), []
    hull = cv2.convexHull(cnt, returnPoints=False)
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:
        return cnt.reshape(-1, 2), []
    pts = cnt.reshape(-1, 2).astype(np.float64)
    notches = []
    if defects is None:
        return pts, notches
    for s, e, f, d in defects.reshape(-1, 4):
        depth = d / 256.0
        if depth < min_depth_ws:
            continue
        p = pts[f]
        mid = (pts[s] + pts[e]) / 2.0
        axis = p - mid
        n = np.linalg.norm(axis)
        if n < 1e-6:
            continue
        notches.append({'idx': int(f), 'pt': p, 'depth': float(depth), 'axis': axis / n,
                        'arc': (int(s), int(e))})
    return pts, notches


def _curvature_notches(mask_ws, min_angle_deg, k, smooth_sigma, min_sep):
    """Inward corners of the outline found by k-cosine turning angle (working scale).
    Returns contour pts (N,2) [x,y] and notch dicts: idx, pt, depth (= turning angle in degrees, used as
    'depth' score), axis (unit vector pointing into the object)."""
    import cv2
    m = mask_ws.astype(np.float32)
    if smooth_sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), smooth_sigma)
    m = (m > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None, []
    cnt = max(cs, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 3 * k + 10:
        return pts, []
    # orientation: make the contour counter-clockwise in image coords so that "inward" is consistent
    area_signed = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    if area_signed < 0:
        pts = pts[::-1]
    prev = np.roll(pts, k, axis=0)
    nxt = np.roll(pts, -k, axis=0)
    a = prev - pts
    b = nxt - pts
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    cosang = np.clip(np.sum(a * b, axis=1) / (na * nb), -1, 1)
    turn = 180.0 - np.degrees(np.arccos(cosang))          # 0 = straight, large = sharp corner
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]          # sign tells convex vs concave corner
    # for a CCW contour (image coords, y down) convex corners have cross < 0 -> concave = cross > 0
    concave = cross > 0
    cand = np.where(concave & (turn >= min_angle_deg))[0]
    notches = []
    if len(cand) == 0:
        return pts, notches
    # non-maximum suppression along the contour
    order = cand[np.argsort(-turn[cand])]
    taken = np.zeros(n, bool)
    for i in order:
        if taken[i]:
            continue
        lo, hi = i - min_sep, i + min_sep
        idx = np.arange(lo, hi + 1) % n
        taken[idx] = True
        bis = a[i] / na[i] + b[i] / nb[i]      # bisector of the two arms points INTO the object at a concave corner
        nb_ = np.linalg.norm(bis)
        if nb_ < 1e-6:
            continue
        notches.append({'idx': int(i), 'pt': pts[i], 'depth': float(turn[i]), 'axis': -bis / nb_, 'arc': (int(i), int(i))})  # bisector points into the notch void; negate -> into the object
    return pts, notches


def _chord_inside(mask, p, q, tol=0.9):
    """Fraction of sample points along p->q that lie inside the mask (working scale)."""
    n = max(8, int(np.hypot(*(q - p))))
    xs = np.linspace(p[0], q[0], n)
    ys = np.linspace(p[1], q[1], n)
    xi = np.clip(np.round(xs).astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(np.round(ys).astype(int), 0, mask.shape[0] - 1)
    inside = mask[yi, xi].astype(bool)
    # ignore the first/last 2 samples (they sit on the outline)
    core = inside[2:-2] if n > 6 else inside
    return float(core.mean()) if core.size else 0.0


def _apply_cut(mask_full, p_full, q_full, thickness):
    """Cut mask_full along the chord. Returns (label image with pieces 1..k, k) or (None, 0)."""
    import cv2
    from scipy import ndimage as ndi

    cut = mask_full.astype(np.uint8).copy()
    cv2.line(cut, (int(round(p_full[0])), int(round(p_full[1]))), (int(round(q_full[0])), int(round(q_full[1]))),
             0, int(thickness))
    lab, k = ndi.label(cut, structure=np.ones((3, 3)))
    if k < 2:
        return None, 0
    # give the removed line pixels back to the nearest piece
    line = mask_full.astype(bool) & (cut == 0)
    if line.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[line] = lab[iy[line], ix[line]]
    return lab, k


def _inscribed_diameter(piece_mask):
    from scipy import ndimage as ndi
    if not piece_mask.any():
        return 0.0
    edt = ndi.distance_transform_edt(np.pad(piece_mask, 1))
    return 2.0 * float(edt.max())


# ----------------------------------------------------------------------------- per-object splitting
def split_object(mask_full, ws=2, min_depth=15.0, max_angle=60.0, neck_ratio=1.2, min_piece_area=15000,
                 chord_inside_frac=0.9, max_chord=None, smooth_sigma=2.0, single=False, single_min_depth=40.0,
                 max_cuts=40, cut_thickness=3, detector='curvature', min_turn=30.0, k_arc=30, min_arc_ratio=1.5,
                 stats=None):
    """Recursively split one object. mask_full: bool array (crop). Returns (label image 1..k, list of cuts).
    detector: 'curvature' (inward corners by turning angle; min_turn in degrees, k_arc = arm length in
    working px) or 'hull' (convexity defects; min_depth in full-res px). `stats` (dict) collects rejection counts."""
    import cv2

    if stats is None:
        stats = {}
    cuts = []
    labels = mask_full.astype(np.int32)  # 1 = the object
    queue = [1]
    next_id = 2
    cos_thr = np.cos(np.deg2rad(max_angle))
    while queue and len(cuts) < max_cuts:
        cur = queue.pop()
        m = labels == cur
        area = int(m.sum())
        if area < 2 * min_piece_area:
            continue
        # work at reduced scale on a tight crop
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = m[y0:y1, x0:x1]
        pad = 4
        crop = np.pad(crop, pad)
        h, w = crop.shape
        ws_h, ws_w = max(2, h // ws), max(2, w // ws)
        mws = cv2.resize(crop.astype(np.uint8), (ws_w, ws_h), interpolation=cv2.INTER_AREA)
        mws_b = (cv2.GaussianBlur(mws.astype(np.float32), (0, 0), smooth_sigma) > 0.5) if smooth_sigma > 0 else mws.astype(bool)
        if detector == 'hull':
            pts, notches = _outline_notches(mws, min_depth / ws, smooth_sigma)
        else:
            pts, notches = _curvature_notches(mws, min_turn, max(3, int(k_arc)), smooth_sigma, max(3, int(k_arc) // 2))
        if pts is None:
            continue
        n_pts = len(pts)
        stats['notches'] = stats.get('notches', 0) + len(notches)
        cands = []
        # ---- pairs of facing notches
        for i in range(len(notches)):
            for j in range(i + 1, len(notches)):
                a, b = notches[i], notches[j]
                v = b['pt'] - a['pt']
                L = float(np.linalg.norm(v))
                if L < 2:
                    continue
                if max_chord is not None and L * ws > max_chord:
                    stats['rej_chord_long'] = stats.get('rej_chord_long', 0) + 1
                    continue
                # the two notches must be far apart along the outline (not two corners of the same dent)
                da = abs(a['idx'] - b['idx'])
                arc = min(da, n_pts - da)
                if arc < min_arc_ratio * L:
                    stats['rej_arc'] = stats.get('rej_arc', 0) + 1
                    continue
                u = v / L
                if float(np.dot(a['axis'], u)) < cos_thr or float(np.dot(b['axis'], -u)) < cos_thr:
                    stats['rej_angle'] = stats.get('rej_angle', 0) + 1
                    continue
                if _chord_inside(mws_b, a['pt'], b['pt']) < chord_inside_frac:
                    stats['rej_inside'] = stats.get('rej_inside', 0) + 1
                    continue
                score = (a['depth'] + b['depth']) / L
                cands.append(('pair', a['pt'], b['pt'], L * ws, score, a['depth'] * ws, b['depth'] * ws))
        cands.sort(key=lambda c: -c[4])
        best = None
        # try candidates in score order until one passes the piece-size and neck rules
        for cand in cands[:12]:
            kind, pw, qw, L_full, score, d1, d2 = cand
            p_full = pw * ws - pad
            q_full = qw * ws - pad
            lab_try, k_try = _apply_cut(m[y0:y1, x0:x1], p_full, q_full, cut_thickness)
            if lab_try is None:
                stats['rej_nocut'] = stats.get('rej_nocut', 0) + 1
                continue
            sizes_try = np.bincount(lab_try.ravel())[1:]
            big_try = [idx + 1 for idx in np.argsort(-sizes_try) if sizes_try[idx] >= min_piece_area]
            if len(big_try) < 2:
                stats['rej_piece'] = stats.get('rej_piece', 0) + 1
                continue
            if any(_inscribed_diameter(lab_try == idx) < neck_ratio * L_full for idx in big_try[:2]):
                stats['rej_neck'] = stats.get('rej_neck', 0) + 1
                continue
            best = cand
            break
        # ---- single deep notch straight across
        if best is None and single:
            for a in notches:
                if a['depth'] * ws < single_min_depth:
                    continue
                # march along the notch axis until leaving the mask
                p = a['pt'].copy()
                q = None
                for t in range(2, max(h, w)):
                    s = p + a['axis'] * t
                    xi, yi = int(round(s[0])), int(round(s[1]))
                    if xi < 0 or yi < 0 or xi >= ws_w or yi >= ws_h or not mws_b[yi, xi]:
                        q = p + a['axis'] * (t - 1)
                        break
                if q is None:
                    continue
                L = float(np.linalg.norm(q - p))
                if L < 2:
                    continue
                cand = ('single', p, q, L * ws, a['depth'] / L, a['depth'] * ws, 0.0)
                if best is None or cand[4] > best[4]:
                    best = cand
        if best is None:
            continue
        kind, pw, qw, L_full, score, d1, d2 = best
        # back to full resolution inside the padded crop
        p_full = pw * ws - pad
        q_full = qw * ws - pad
        crop_full = m[y0:y1, x0:x1]
        lab, k = _apply_cut(crop_full, p_full, q_full, cut_thickness)
        if lab is None:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        order = np.argsort(-sizes)
        big = [idx + 1 for idx in order if sizes[idx] >= min_piece_area]
        if len(big) < 2:
            # a tiny sliver was cut off: reject this notch pair by lowering its weight -> simply skip object
            continue
        # neck rule: the two largest pieces must both be wider than the cut
        ok = True
        for idx in big[:2]:
            if _inscribed_diameter(lab == idx) < neck_ratio * L_full:
                ok = False
                break
        if not ok:
            continue
        # accept: piece big[0] keeps `cur`, the others get new ids; tiny leftovers join the nearest big piece
        from scipy import ndimage as ndi
        small = np.isin(lab, [idx + 1 for idx in range(len(sizes)) if (idx + 1) not in big]) & (lab > 0)
        if small.any():
            keep = np.where(np.isin(lab, big), lab, 0)
            _, (iy, ix) = ndi.distance_transform_edt(keep == 0, return_indices=True)
            lab[small] = keep[iy[small], ix[small]]
        new_ids = {}
        for n_i, idx in enumerate(big):
            new_ids[idx] = cur if n_i == 0 else next_id
            if n_i > 0:
                next_id += 1
        sub = labels[y0:y1, x0:x1]
        for idx, nid in new_ids.items():
            sub[lab == idx] = nid
        cuts.append({'kind': kind, 'p': (float(p_full[0] + x0), float(p_full[1] + y0)),
                     'q': (float(q_full[0] + x0), float(q_full[1] + y0)), 'length': L_full, 'depths': (d1, d2)})
        queue.extend(new_ids.values())
    # relabel to 1..k
    ids = np.unique(labels)
    ids = ids[ids > 0]
    out = np.zeros_like(labels)
    for n_i, i in enumerate(ids, 1):
        out[labels == i] = n_i
    return out, cuts


def split_labels(inst, stats=None, **params):
    """Apply split_object to every object of an instance label image. Returns (new labels, all cuts)."""
    from scipy import ndimage as ndi

    inst = inst.astype(np.int32)
    out = inst.copy()
    next_id = int(inst.max()) + 1
    all_cuts = []
    for idx, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        m = inst[sl] == idx
        pieces, cuts = split_object(m, stats=stats, **params)
        if not cuts:
            continue
        sub = out[sl]
        for k in range(2, int(pieces.max()) + 1):
            sub[(pieces == k) & m] = next_id
            next_id += 1
        for c in cuts:
            c['p'] = (c['p'][0] + sl[1].start, c['p'][1] + sl[0].start)
            c['q'] = (c['q'][0] + sl[1].start, c['q'][1] + sl[0].start)
            c['object'] = idx
        all_cuts.extend(cuts)
    return out, all_cuts


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--min-depth', type=float, default=15.0, help='minimum notch depth in pixels (default 15)')
    p.add_argument('--max-angle', type=float, default=60.0, help='max angle between a notch direction and the cut (default 60)')
    p.add_argument('--neck-ratio', type=float, default=1.2, help='pieces must be this much wider than the cut (default 1.2)')
    p.add_argument('--min-piece-area', type=int, default=15000, help='minimum piece area in pixels (default 15000)')
    p.add_argument('--max-chord', type=float, default=None, help='maximum cut length in pixels (default: no limit)')
    p.add_argument('--ws', type=int, default=2, help='working downsampling for outline analysis (default 2)')
    p.add_argument('--single', action='store_true', help='also allow single-notch cuts straight across')
    p.add_argument('--single-min-depth', type=float, default=40.0)
    p.add_argument('--detector', default='curvature', choices=['curvature', 'hull'],
                   help='notch detector: inward corners by outline curvature (default) or convexity defects')
    p.add_argument('--min-turn', type=float, default=30.0, help='curvature: minimum inward turning angle in degrees (default 30)')
    p.add_argument('--k-arc', type=int, default=30, help='curvature: arm length along the outline in working px (default 30)')
    p.add_argument('--min-arc-ratio', type=float, default=1.5, help='outline distance between paired notches must exceed this x chord')
    p.add_argument('--overlay', default=None, help='optional image to draw the cuts on (writes <out>_cuts.png)')
    args = p.parse_args()

    inst = tifffile.imread(args.labels).astype(np.int32)
    stats = {}
    out, cuts = split_labels(inst, stats=stats, ws=args.ws, min_depth=args.min_depth, max_angle=args.max_angle,
                             neck_ratio=args.neck_ratio, min_piece_area=args.min_piece_area, max_chord=args.max_chord,
                             single=args.single, single_min_depth=args.single_min_depth, detector=args.detector,
                             min_turn=args.min_turn, k_arc=args.k_arc, min_arc_ratio=args.min_arc_ratio)
    data = out.astype(np.uint16) if out.max() < 65535 else out.astype(np.uint32)
    tifffile.imwrite(args.out, data, compression='lzw')
    n_pair = sum(c['kind'] == 'pair' for c in cuts)
    print(f'{len(cuts)} cuts ({n_pair} notch pairs, {len(cuts) - n_pair} single-notch); {int(inst.max())} -> {int(out.max())} objects -> {args.out}')
    print('candidate statistics:', {k: v for k, v in sorted(stats.items())})
    if args.overlay:
        import cv2
        from skimage.segmentation import find_boundaries
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from input_scaling import to_uint8
        img = tifffile.imread(args.overlay)
        f = max(1, int(np.ceil(max(img.shape) / 4000.0)))
        base = to_uint8(img[::f, ::f], 'percentile')
        rgb = np.stack([base] * 3, -1)
        rgb[find_boundaries(out[::f, ::f], mode='thick')] = (255, 0, 255)
        for c in cuts:
            cv2.line(rgb, (int(c['p'][0] / f), int(c['p'][1] / f)), (int(c['q'][0] / f), int(c['q'][1] / f)), (255, 255, 0), 2)
        png = os.path.splitext(args.out)[0] + '_cuts.png'
        cv2.imwrite(png, rgb[..., ::-1])
        print(f'overlay -> {png}')


if __name__ == '__main__':
    main()
