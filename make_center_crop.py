r"""
make_center_crop.py - save a centred crop of a large TIF (keeps bit depth and pixel-size metadata).

  python make_center_crop.py "path\to\montage.tif" --area-frac 0.15          -> montage_center15pct.tif
  python make_center_crop.py "path\to\montage.tif" --size 4096                 (square crop in pixels)
"""
import argparse
import math
import os

import tifffile


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image')
    p.add_argument('--area-frac', type=float, default=0.15, help='fraction of the image AREA to keep (default 0.15)')
    p.add_argument('--size', type=int, default=None, help='square crop size in pixels (overrides --area-frac)')
    p.add_argument('--box', type=int, nargs=4, default=None, metavar=('Y0', 'X0', 'H', 'W'),
                   help='explicit crop box (overrides the centred crop)')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    with tifffile.TiffFile(args.image) as tif:
        img = tif.asarray()
        xres = tif.pages[0].tags.get('XResolution')
        md = tif.imagej_metadata or {}
    H, W = img.shape[:2]
    if args.box:
        y0, x0, h, w = args.box
        h, w = min(h, H - y0), min(w, W - x0)
        tag = f'_y{y0}_x{x0}_{h}x{w}'
    elif args.size:
        h = w = min(args.size, H, W)
        tag = f'{h}px'
    else:
        s = math.sqrt(args.area_frac)
        h, w = int(round(H * s)), int(round(W * s))
        tag = f'{int(round(args.area_frac * 100))}pct'
    if not args.box:
        y0, x0 = (H - h) // 2, (W - w) // 2
    crop = img[y0:y0 + h, x0:x0 + w]
    out = args.out or os.path.splitext(args.image)[0] + f'_center{tag}.tif'
    kwargs = {}
    if xres is not None and xres.value[0]:
        kwargs['resolution'] = (xres.value[0] / xres.value[1], xres.value[0] / xres.value[1])
    desc = 'ImageJ=1.54g\n' + ''.join(f'{k}={md[k]}\n' for k in ('unit', 'min', 'max') if k in md)
    tifffile.imwrite(out, crop, description=desc, **kwargs)
    print(f'{out}: {crop.shape} {crop.dtype}, offset y={y0} x={x0} in the original')


if __name__ == '__main__':
    main()
