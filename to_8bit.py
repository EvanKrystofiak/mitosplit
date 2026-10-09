r"""
to_8bit.py - convert raw (16-bit) TEM images to the 8-bit, contrast-stretched form the fine-tuned
mitochondria model expects, so they can be opened in napari and segmented with the empanada
"2D Inference" widget. Uses the same conversion as segment_mito.py (see input_scaling.py).

  python to_8bit.py "path\to\montage.tif"                     -> path\to\montage_8bit.tif
  python to_8bit.py "path\to\folder" --outdir "path\to\8bit"   (all TIFs in the folder)
  python to_8bit.py img.tif --mode zscore
"""
import argparse
import glob
import os

import numpy as np
import tifffile

from input_scaling import MODES, to_uint8, imagej_display_range


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', help='TIF file(s) or folder(s)')
    p.add_argument('--outdir', default=None, help='output folder (default: next to the input)')
    p.add_argument('--mode', default='percentile', choices=MODES, help='scaling mode (default percentile)')
    p.add_argument('--p-low', type=float, default=0.5)
    p.add_argument('--p-high', type=float, default=99.5)
    args = p.parse_args()

    files = []
    for item in args.inputs:
        if os.path.isdir(item):
            files += sorted(f for f in glob.glob(os.path.join(item, '*')) if f.lower().endswith(('.tif', '.tiff')))
        else:
            files += sorted(glob.glob(item)) or [item]
    for f in files:
        with tifffile.TiffFile(f) as tif:
            img = tif.asarray()
            dr = imagej_display_range(tif)
            xres = tif.pages[0].tags.get('XResolution')
            unit = (tif.imagej_metadata or {}).get('unit') if tif.is_imagej else None
        out8 = to_uint8(img, args.mode, args.p_low, args.p_high, display_range=dr)
        outdir = args.outdir or os.path.dirname(os.path.abspath(f))
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, os.path.splitext(os.path.basename(f))[0] + '_8bit.tif')
        kwargs = {}
        if xres is not None and xres.value[0]:
            kwargs['resolution'] = (xres.value[0] / xres.value[1], xres.value[0] / xres.value[1])
        tifffile.imwrite(out, out8, compression='lzw', description='ImageJ=1.54g\n' + (f'unit={unit}\n' if unit else ''), **kwargs)
        print(f'{f} -> {out}  ({img.dtype} {img.min()}-{img.max()} -> uint8, mode={args.mode})')


if __name__ == '__main__':
    main()
