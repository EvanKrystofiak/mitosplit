r"""
segment_mito.py - Batch mitochondria segmentation of large 2D TEM images (single images, folders
of images, or multi-page TIF stacks) with a model registered in empanada-napari.

It runs exactly the same inference engine as the empanada-napari "2D Inference" widget
(empanada_napari.inference.Engine2d) with tiled GPU inference, so results match napari, but it
runs unattended from the command line and writes ImageJ-friendly outputs.

Outputs per input image (in --outdir, default: <input folder>\mito_seg):
  <name>_mito_labels.tif   instance label image (0 = background, 1..N = mitochondria)
  <name>_mito_rois.zip     ImageJ ROI set, one polygon ROI per mitochondrion (open with ROI Manager)
  <name>_mito_stats.csv    per-object measurements (area, centroid, bbox, axis lengths, ...)
  <name>_mito_overlay.png  downsampled QC overlay (magenta outlines on the image)
  summary.json             parameters, timings and object counts for the whole run

Examples (inside the empanada-gpu environment):
  python segment_mito.py "path\to\montage.tif"
  python segment_mito.py "path\to\montage_folder" --outdir "path\to\seg"
  python segment_mito.py img.tif --downsampling 2 --min-area 2000 --fill-holes

NOTE on input scaling: the fine-tuned model was trained on 8-bit, contrast-stretched images. Raw 16-bit
TEM montages (values in a narrow band such as 34000-45000) give (almost) no detections unless they are
converted first. This script does that conversion internally (--input-scaling, see input_scaling.py);
use to_8bit.py to create the same 8-bit images for interactive work in napari.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import tifffile

from input_scaling import MODES as SCALING_MODES, to_uint8, imagej_display_range

IMAGE_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', help='image file(s) and/or folder(s) of images')
    p.add_argument('--outdir', default=None, help=r'output folder (default: <input folder>\mito_seg)')
    p.add_argument('--model', default='mito_em_totoro', help='registered empanada model name (default: mito_em_totoro)')
    p.add_argument('--input-scaling', default='percentile', choices=SCALING_MODES,
                   help='how to convert the image to 8-bit before inference (default percentile); '
                        '"none" feeds the image as is, like the napari widget does')
    p.add_argument('--p-low', type=float, default=0.5, help='low percentile for --input-scaling percentile (default 0.5)')
    p.add_argument('--p-high', type=float, default=99.5, help='high percentile for --input-scaling percentile (default 99.5)')
    p.add_argument('--downsampling', default='4',
                   help='integer downsample factor applied before the network, or "auto" to derive it from the '
                        'pixel size in the TIF metadata and --target-nm-per-px (default 4)')
    p.add_argument('--target-nm-per-px', type=float, default=7.3,
                   help='with --downsampling auto: pixel size the model expects (default 7.3 = training scale)')
    p.add_argument('--tile-size', type=int, default=None,
                   help='tile size in image pixels before downsampling (default 2048 x downsampling); 0 = no tiling')
    p.add_argument('--confidence-thr', type=float, default=0.5, help='segmentation confidence threshold (default 0.5)')
    p.add_argument('--center-thr', type=float, default=0.1, help='center confidence threshold (default 0.1)')
    p.add_argument('--min-distance', type=int, default=3, help='centers min distance / NMS kernel (default 3)')
    p.add_argument('--fine-boundaries', action='store_true', help='finer boundaries between touching objects')
    p.add_argument('--max-objects', type=int, default=100000, help='max objects per class per tile (label divisor)')
    p.add_argument('--min-area', type=int, default=0, help='drop objects smaller than this many pixels (default 0)')
    p.add_argument('--fill-holes', action='store_true', help='fill holes inside each object')
    p.add_argument('--split-merged', action='store_true',
                   help='split concave objects that are several touching mitochondria fused together (see split_merged.py)')
    p.add_argument('--split-solidity', type=float, default=0.85,
                   help='only objects with solidity below this are considered for splitting (default 0.85)')
    p.add_argument('--split-notches', action='store_true',
                   help='cut fused mitochondria along chords between facing outline notches (see split_by_notches.py)')
    p.add_argument('--notch-min-turn', type=float, default=30.0, help='--split-notches: minimum inward turning angle (default 30)')
    p.add_argument('--notch-neck-ratio', type=float, default=1.2, help='--split-notches: pieces must be this much wider than the cut')
    p.add_argument('--notch-single', action='store_true', help='--split-notches: also allow single-notch cuts straight across')
    p.add_argument('--no-rois', action='store_true', help='skip ImageJ ROI export')
    p.add_argument('--no-overlay', action='store_true', help='skip QC overlay PNG')
    p.add_argument('--no-stats', action='store_true', help='skip per-object CSV')
    p.add_argument('--overlay-max-px', type=int, default=4000, help='longest side of overlay PNG (default 4000)')
    p.add_argument('--cpu', action='store_true', help='force CPU inference')
    p.add_argument('--recursive', action='store_true', help='search folders recursively')
    p.add_argument('--overwrite', action='store_true', help='re-process images that already have outputs')
    return p.parse_args()


# ----------------------------------------------------------------------------- I/O helpers
def collect_inputs(inputs, recursive=False):
    files = []
    for item in inputs:
        if os.path.isdir(item):
            pattern = os.path.join(item, '**', '*') if recursive else os.path.join(item, '*')
            for f in sorted(glob.glob(pattern, recursive=recursive)):
                if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(f):
                    files.append(f)
        elif os.path.isfile(item):
            files.append(item)
        else:
            matches = sorted(glob.glob(item))
            if not matches:
                print(f'WARNING: no such file or folder: {item}')
            files.extend(m for m in matches if os.path.isfile(m))
    seen, out = set(), []
    for f in files:
        a = os.path.abspath(f)
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def read_pixel_size(tif):
    """Return (pixel_size, unit) from a tifffile.TiffFile, or (None, None)."""
    try:
        page = tif.pages[0]
        xres = page.tags.get('XResolution')
        unit = None
        if tif.is_imagej and tif.imagej_metadata:
            unit = tif.imagej_metadata.get('unit')
        if xres is None:
            return None, None
        num, den = xres.value
        if not num:
            return None, None
        px = den / num
        if unit is None:
            ru = page.tags.get('ResolutionUnit')
            ru = int(ru.value) if ru is not None else None
            if ru == 2:
                unit = 'inch'
            elif ru == 3:
                unit = 'cm'
            else:
                return None, None
        return float(px), str(unit)
    except Exception:
        return None, None


def load_image(path):
    """Return (array, pixel_size, unit, display_range). Array is 2D (H, W) or a 3D stack (N, H, W)."""
    ext = os.path.splitext(path)[1].lower()
    display_range = None
    if ext in ('.tif', '.tiff'):
        with tifffile.TiffFile(path) as tif:
            arr = tif.asarray()
            px, unit = read_pixel_size(tif)
            display_range = imagej_display_range(tif)
    else:
        import cv2
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f'could not read {path}')
        if arr.ndim == 3:
            arr = arr[..., ::-1]  # BGR -> RGB
        px, unit = None, None

    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 4 and arr.shape[1] > 4:
        arr = np.round(arr[..., :3].astype(np.float32).mean(-1)).astype(arr.dtype)  # colour -> grey
    if arr.ndim not in (2, 3):
        raise ValueError(f'{path}: expected a 2D image or a stack of 2D images, got shape {arr.shape}')
    return arr, px, unit, display_range


def save_labels(inst, path, px=None, unit=None):
    n = int(inst.max()) if inst.size else 0
    data = inst.astype(np.uint16) if n < 65535 else inst.astype(np.uint32)
    kwargs = {}
    if px:
        kwargs['resolution'] = (1.0 / px, 1.0 / px)
    description = 'ImageJ=1.54g\n' + (f'unit={unit}\n' if unit else '')
    try:
        tifffile.imwrite(path, data, compression='lzw', description=description, **kwargs)
    except Exception:
        tifffile.imwrite(path, data, compression='lzw')
    return data.dtype


# ----------------------------------------------------------------------------- label utilities
def pan_to_instances(pan_seg, label, label_divisor):
    """Convert an empanada panoptic map (label*divisor + instance) to plain instance ids 1..N."""
    lo, hi = label * label_divisor, (label + 1) * label_divisor
    inst = pan_seg.astype(np.int32, copy=True)
    mask = (inst >= lo) & (inst < hi)
    inst[mask] -= lo
    inst[~mask] = 0
    return inst


def postprocess(inst, min_area=0, fill_holes=False):
    """Remove small objects, relabel to consecutive ids, optionally fill holes. Returns (inst, n_removed)."""
    from scipy import ndimage as ndi

    counts = np.bincount(inst.ravel())
    counts[0] = 0
    keep = counts >= max(min_area, 1)
    n_removed = int((counts > 0).sum() - keep.sum())
    lut = np.zeros(len(counts), dtype=np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    inst = lut[inst]

    if fill_holes and inst.max() > 0:
        for idx, slc in enumerate(ndi.find_objects(inst), start=1):
            if slc is None:
                continue
            crop = inst[slc]
            filled = ndi.binary_fill_holes(crop == idx)
            add = filled & (crop == 0)
            if add.any():
                crop[add] = idx
    return inst, n_removed


def object_stats(inst, px=None, unit=None):
    from skimage.measure import regionprops_table
    import pandas as pd

    props = ['label', 'area', 'centroid', 'bbox', 'equivalent_diameter_area',
             'axis_major_length', 'axis_minor_length', 'eccentricity', 'solidity', 'perimeter']
    df = pd.DataFrame(regionprops_table(inst, properties=props))
    df = df.rename(columns={
        'label': 'id', 'area': 'area_px', 'centroid-0': 'centroid_y', 'centroid-1': 'centroid_x',
        'bbox-0': 'bbox_y0', 'bbox-1': 'bbox_x0', 'bbox-2': 'bbox_y1', 'bbox-3': 'bbox_x1',
        'equivalent_diameter_area': 'equiv_diameter_px', 'axis_major_length': 'major_axis_px',
        'axis_minor_length': 'minor_axis_px', 'perimeter': 'perimeter_px',
    })
    if px:
        u = unit or 'unit'
        df[f'area_{u}2'] = df['area_px'] * px * px
        df[f'equiv_diameter_{u}'] = df['equiv_diameter_px'] * px
        df[f'major_axis_{u}'] = df['major_axis_px'] * px
        df[f'minor_axis_{u}'] = df['minor_axis_px'] * px
        df[f'perimeter_{u}'] = df['perimeter_px'] * px
        if u == 'nm':
            df['area_um2'] = df['area_px'] * (px / 1000.0) ** 2
    return df


def export_imagej_rois(inst, path, name_prefix='mito'):
    import cv2
    import roifile
    from scipy import ndimage as ndi

    rois = []
    for idx, slc in enumerate(ndi.find_objects(inst), start=1):
        if slc is None:
            continue
        crop = (inst[slc] == idx).astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        if len(cnt) < 3:
            continue
        # cv2 returns boundary-pixel centres; ImageJ polygon vertices sit on pixel corners, so +0.5
        # puts the polygon through the pixel centres (area error is about half a pixel of border).
        cnt[:, 0] += slc[1].start + 0.5
        cnt[:, 1] += slc[0].start + 0.5
        rois.append(roifile.ImagejRoi.frompoints(cnt, name=f'{name_prefix}_{idx:05d}'))
    if os.path.exists(path):
        os.remove(path)
    roifile.roiwrite(path, rois)
    return len(rois)


def make_overlay(img, inst, path, max_px=4000):
    import cv2
    from skimage.segmentation import find_boundaries

    f = max(1, int(np.ceil(max(img.shape) / float(max_px))))
    small = img[::f, ::f].astype(np.float32)
    lo, hi = np.percentile(small, (0.5, 99.5))
    small8 = np.clip((small - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    bnd = find_boundaries(inst[::f, ::f], mode='thick')
    rgb = np.stack([small8] * 3, axis=-1)
    rgb[bnd] = (255, 0, 255)
    cv2.imwrite(path, rgb[..., ::-1])
    return f


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    t_start = time.time()

    import torch
    from empanada_napari.utils import get_configs
    from empanada.config_loaders import read_yaml
    from empanada_napari.inference import Engine2d

    configs = get_configs()
    if args.model not in configs:
        sys.exit(f'model "{args.model}" is not registered. Available: {", ".join(sorted(configs))}')
    config = read_yaml(configs[args.model])
    thing_label = int(config['thing_list'][0])
    class_name = str(config['class_names'][thing_label])

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_gpu else 'CPU'
    auto_ds = str(args.downsampling).lower() == 'auto'
    ds_fixed = None if auto_ds else max(1, int(args.downsampling))
    if ds_fixed is not None and (ds_fixed & (ds_fixed - 1)) != 0:
        sys.exit(f'--downsampling must be a power of 2 (1, 2, 4, 8, 16, ...); got {ds_fixed}')
    torch.backends.cudnn.benchmark = True

    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        sys.exit('no input images found')

    print(f'model: {args.model} ({configs[args.model]})')
    print(f'device: {device_name}   downsampling: {args.downsampling}')
    print(f'{len(files)} image(s) to process')

    engine = Engine2d(
        config,
        inference_scale=ds_fixed or 1,
        label_divisor=args.max_objects,
        nms_threshold=args.center_thr,
        nms_kernel=args.min_distance,
        confidence_thr=args.confidence_thr,
        semantic_only=False,
        fine_boundaries=args.fine_boundaries,
        tile_size=0,
        use_gpu=use_gpu,
        use_quantized=False,
    )

    def set_scale(ds):
        tile = args.tile_size if args.tile_size is not None else 2048 * ds
        engine.update_params(inference_scale=ds, label_divisor=args.max_objects, nms_threshold=args.center_thr,
                             nms_kernel=args.min_distance, confidence_thr=args.confidence_thr,
                             fine_boundaries=args.fine_boundaries, semantic_only=False, tile_size=tile)
        return tile

    params = {k: v for k, v in vars(args).items() if k != 'inputs'}
    summary = {'model': args.model, 'model_config': configs[args.model], 'device': device_name,
               'parameters': params, 'images': []}
    outdir_global = os.path.abspath(args.outdir) if args.outdir else None
    if outdir_global:
        os.makedirs(outdir_global, exist_ok=True)

    for fi, path in enumerate(files, start=1):
        name = os.path.splitext(os.path.basename(path))[0]
        outdir = outdir_global or os.path.join(os.path.dirname(path), 'mito_seg')
        os.makedirs(outdir, exist_ok=True)
        out_labels = os.path.join(outdir, f'{name}_mito_labels.tif')
        if os.path.exists(out_labels) and not args.overwrite:
            print(f'[{fi}/{len(files)}] {name}: outputs exist, skipping (use --overwrite)')
            continue

        print(f'[{fi}/{len(files)}] {name}: loading...', flush=True)
        t0 = time.time()
        img, px, unit, display_range = load_image(path)
        if auto_ds:
            if not px or unit not in ('nm', 'um', 'µm', 'micron'):
                print(f'   WARNING: no usable pixel size in metadata, using downsampling 4')
                ds = 4
            else:
                px_nm = px * (1000.0 if unit != 'nm' else 1.0)
                ds = int(2 ** max(0, round(np.log2(args.target_nm_per_px / px_nm))))  # engine needs a power of 2
        else:
            ds = ds_fixed
        tile_size = set_scale(ds)
        print(f'   pixel size {px} {unit} -> downsampling {ds}, tile {tile_size} px')
        rec = {'file': path, 'shape': list(img.shape), 'dtype': str(img.dtype), 'pixel_size': px, 'unit': unit,
               'input_scaling': args.input_scaling, 'downsampling': ds, 'tile_size': tile_size}

        planes = [img] if img.ndim == 2 else list(img)
        if len(planes) > 1:
            print(f'   stack of {len(planes)} planes, each {planes[0].shape}')

        results, t_inf = [], 0.0
        for pi, plane in enumerate(planes):
            net_in = to_uint8(plane, args.input_scaling, args.p_low, args.p_high, display_range=display_range)
            t1 = time.time()
            pan = engine.infer(np.ascontiguousarray(net_in))
            del net_in
            t_inf += time.time() - t1
            inst = pan_to_instances(pan, thing_label, args.max_objects)
            del pan
            inst, n_removed = postprocess(inst, args.min_area, args.fill_holes)
            if args.split_merged:
                from split_merged import split_merged_objects
                inst, n_split = split_merged_objects(inst, solidity_thr=args.split_solidity)
                print(f'   split {n_split} merged objects')
            if args.split_notches:
                from split_by_notches import split_labels
                inst, cuts = split_labels(inst, detector='curvature', min_turn=args.notch_min_turn,
                                          neck_ratio=args.notch_neck_ratio, single=args.notch_single)
                print(f'   notch splitting: {len(cuts)} cuts')
            results.append(inst)
            if len(planes) > 1:
                print(f'   plane {pi}: {int(inst.max())} {class_name} objects ({n_removed} removed as too small)')
        if use_gpu:
            torch.cuda.empty_cache()

        inst = results[0] if len(results) == 1 else np.stack(results, 0)
        n_obj = [int(r.max()) for r in results]
        rec['n_objects'] = n_obj if len(n_obj) > 1 else n_obj[0]
        rec['inference_seconds'] = round(t_inf, 1)

        dtype = save_labels(inst, out_labels, px, unit)
        rec['labels'] = out_labels

        if inst.ndim == 2:
            if not args.no_stats:
                csv_path = os.path.join(outdir, f'{name}_mito_stats.csv')
                object_stats(inst, px, unit).to_csv(csv_path, index=False)
                rec['stats'] = csv_path
            if not args.no_rois:
                zip_path = os.path.join(outdir, f'{name}_mito_rois.zip')
                export_imagej_rois(inst, zip_path, name_prefix=class_name)
                rec['rois'] = zip_path
            if not args.no_overlay:
                png_path = os.path.join(outdir, f'{name}_mito_overlay.png')
                rec['overlay_downsample'] = make_overlay(img, inst, png_path, args.overlay_max_px)
                rec['overlay'] = png_path

        rec['total_seconds'] = round(time.time() - t0, 1)
        summary['images'].append(rec)
        print(f'   {rec["n_objects"]} {class_name} objects | inference {t_inf:.0f} s | total {rec["total_seconds"]:.0f} s '
              f'| labels saved as {dtype} -> {out_labels}', flush=True)
        del img, inst, results

        with open(os.path.join(outdir, 'summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=2, default=str)

    summary['elapsed_seconds'] = round(time.time() - t_start, 1)
    print(f'done in {summary["elapsed_seconds"]:.0f} s')


if __name__ == '__main__':
    main()
