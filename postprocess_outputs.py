r"""
postprocess_outputs.py - apply notch-based splitting of fused mitochondria to EXISTING segment_mito.py
outputs (no GPU, no re-inference) and write a new output folder with labels, ImageJ ROI zips, stats CSVs
and QC overlays. The source folder is left untouched.

  python postprocess_outputs.py --src "Data to segment\mito_seg" --images "Data to segment" --out "Data to segment\mito_seg_split"
  python postprocess_outputs.py --src ... --images ... --out ... --min-turn 30 --neck-ratio 1.2 [--single]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_mito import save_labels, object_stats, export_imagej_rois, make_overlay, read_pixel_size  # noqa: E402
from split_by_notches import split_labels  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='folder with <name>_mito_labels.tif files')
    p.add_argument('--images', required=True, help='folder with the original <name>.tif images')
    p.add_argument('--out', required=True, help='output folder')
    p.add_argument('--min-turn', type=float, default=30.0)
    p.add_argument('--neck-ratio', type=float, default=1.2)
    p.add_argument('--k-arc', type=int, default=30)
    p.add_argument('--single', action='store_true')
    p.add_argument('--only', nargs='*', default=None, help='process only these image names (without extension)')
    p.add_argument('--no-overlay', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = {'source': os.path.abspath(args.src), 'parameters': vars(args), 'images': []}
    labels = sorted(glob.glob(os.path.join(args.src, '*_mito_labels.tif')))
    for lf in labels:
        name = os.path.basename(lf).replace('_mito_labels.tif', '')
        if args.only and name not in args.only:
            continue
        img_path = os.path.join(args.images, name + '.tif')
        if not os.path.exists(img_path):
            img_path = os.path.join(args.images, name + '.tiff')
        t0 = time.time()
        inst = tifffile.imread(lf).astype(np.int32)
        n_before = int(inst.max())
        inst, cuts = split_labels(inst, detector='curvature', min_turn=args.min_turn, neck_ratio=args.neck_ratio,
                                  k_arc=args.k_arc, single=args.single)
        n_after = int(inst.max())
        px = unit = None
        img = None
        if os.path.exists(img_path):
            with tifffile.TiffFile(img_path) as tif:
                px, unit = read_pixel_size(tif)
                if not args.no_overlay:
                    img = tif.asarray()
        out_labels = os.path.join(args.out, f'{name}_mito_labels.tif')
        save_labels(inst, out_labels, px, unit)
        object_stats(inst, px, unit).to_csv(os.path.join(args.out, f'{name}_mito_stats.csv'), index=False)
        export_imagej_rois(inst, os.path.join(args.out, f'{name}_mito_rois.zip'), name_prefix='mito')
        if img is not None:
            make_overlay(img, inst, os.path.join(args.out, f'{name}_mito_overlay.png'))
            del img
        rec = {'name': name, 'objects_before': n_before, 'objects_after': n_after, 'cuts': len(cuts),
               'cuts_detail': [{k: v for k, v in c.items() if k in ('kind', 'p', 'q', 'length', 'object')} for c in cuts],
               'seconds': round(time.time() - t0, 1)}
        summary['images'].append(rec)
        print(f'{name}: {n_before} -> {n_after} objects ({len(cuts)} cuts) in {rec["seconds"]:.0f} s', flush=True)
        with open(os.path.join(args.out, 'summary_split.json'), 'w') as fh:
            json.dump(summary, fh, indent=1, default=float)
    print('done')


if __name__ == '__main__':
    main()
