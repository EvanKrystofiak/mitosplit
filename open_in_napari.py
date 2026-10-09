r"""
open_in_napari.py - open one or more TEM images in napari (empanada-gpu env), converted to the 8-bit
contrast-stretched form the fine-tuned model expects, ready for Plugins > empanada-napari > 2D Inference.

  python open_in_napari.py "path\to\montage.tif"
  python open_in_napari.py img.tif --mode zscore      (see input_scaling.py for modes; --mode none = raw)
"""
import argparse
import os
import sys


import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from input_scaling import MODES, to_uint8, imagej_display_range


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('images', nargs='*', help='TIF file(s) to open')
    p.add_argument('--mode', default='percentile', choices=MODES)
    p.add_argument('--rois', default=None, help='ImageJ RoiSet.zip to show as an outline labels layer')
    p.add_argument('--roi-offset', type=int, nargs=2, default=(0, 0), metavar=('Y', 'X'),
                   help='origin of the loaded image inside the ROI coordinate system (for crops)')
    p.add_argument('--save-8bit', action='store_true', help='also save the converted image next to the input')
    p.add_argument('--labels', default=None, help='label image (e.g. *_mito_labels.tif from segment_mito.py) to add as a Labels layer')
    args = p.parse_args()

    import napari
    viewer = napari.Viewer(title='napari - empanada-gpu')
    for f in args.images:
        print(f'loading {f} ...', flush=True)
        with tifffile.TiffFile(f) as tif:
            img = tif.asarray()
            dr = imagej_display_range(tif)
        name = os.path.splitext(os.path.basename(f))[0]
        if args.mode != 'none':
            img = to_uint8(img, args.mode, display_range=dr)
            name += f'_8bit_{args.mode}'
        viewer.add_image(img, name=name)
        if args.save_8bit and args.mode != 'none':
            out = os.path.splitext(f)[0] + f'_8bit_{args.mode}.tif'
            tifffile.imwrite(out, img, compression='lzw')
            print(f'saved {out}')
        if args.labels:
            lab_img = tifffile.imread(args.labels)
            viewer.add_labels(lab_img, name=os.path.basename(args.labels).replace('.tif', ''), opacity=0.4)
            print(f'{int(lab_img.max())} objects in {args.labels}')
        if args.rois:
            from evaluate_vs_rois import load_gt_rois
            gt, info, skipped, _ = load_gt_rois(args.rois, img.shape[:2], offset=tuple(args.roi_offset))
            lab = viewer.add_labels(gt, name='ground_truth_ROIs', opacity=0.8)
            lab.contour = 2
            print(f'{len(info)} ROIs fall inside this image and were rasterised ({len(skipped)} outside or invalid)')
    print('napari ready. Plugins > empanada-napari > 2D Inference, model "mito_em_totoro".', flush=True)
    napari.run()


if __name__ == '__main__':
    main()
