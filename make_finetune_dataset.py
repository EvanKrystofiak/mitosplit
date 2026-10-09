r"""
make_finetune_dataset.py - build an empanada fine-tuning dataset from a large TEM image and hand-drawn
ImageJ ROIs (RoiSet.zip). Produces the folder layout the empanada-napari "Finetune a model" widget and
finetune_mito.py expect:

  <out>/train/<name>/images/*.tiff   8-bit contrast-stretched image patches
  <out>/train/<name>/masks/*.tiff    int32 instance masks (0 = background, 1..N = one id per ROI)
  <out>/val/<name>/images|masks      held-out patches (a vertical stripe at the right edge by default)

Two sampling modes:
  grid (default)      regular grid of patches; assumes the ROIs are exhaustive over the whole image
  --around-objects    PARTIAL annotations: one patch centred on every annotated object, and the image is
                      blanked (flat grey) beyond --margin pixels from any annotated object, so unannotated
                      mitochondria elsewhere cannot be learned as background.

  python make_finetune_dataset.py --image img.tif --rois RoiSet.zip --out finetune_data\ds4 --downsampling 4 --around-objects
"""
import argparse
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from input_scaling import to_uint8, imagej_display_range  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image', required=True)
    p.add_argument('--rois', required=True)
    p.add_argument('--out', required=True, help='output dataset folder')
    p.add_argument('--name', default='totoro', help='source sub-folder name (default totoro)')
    p.add_argument('--downsampling', type=int, default=4)
    p.add_argument('--patch', type=int, default=512, help='patch size in downsampled pixels (default 512)')
    p.add_argument('--stride', type=int, default=256, help='grid mode: patch stride for training patches')
    p.add_argument('--val-frac', type=float, default=0.2, help='fraction of the image width held out (right edge)')
    p.add_argument('--scaling', default='percentile', help='8-bit conversion mode (see input_scaling.py)')
    p.add_argument('--max-empty-frac', type=float, default=0.15,
                   help='grid mode: max fraction of patches without any object (default 0.15)')
    p.add_argument('--around-objects', action='store_true', help='partial-annotation mode (see above)')
    p.add_argument('--margin', type=int, default=64,
                   help='around-objects: keep image within this many downsampled px of annotated objects (default 64)')
    p.add_argument('--jitter', type=float, default=0.25, help='around-objects: random patch shift, fraction of patch')
    p.add_argument('--per-object', type=int, default=1, help='around-objects: patches per annotated object')
    p.add_argument('--blank-border', type=int, default=0,
                   help='blank (flat grey) a band this many ORIGINAL pixels wide along the image border, e.g. when '
                        'edge-touching objects were not annotated (default 0)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def rasterise_rois(path, shape, ds):
    """Rasterise ImageJ ROIs into a label image of `shape` (already downsampled by ds)."""
    import roifile
    from skimage.draw import polygon as draw_polygon

    rois = roifile.roiread(path)
    if not isinstance(rois, list):
        rois = [rois]
    lab = np.zeros(shape, np.int32)
    n = 0
    for roi in rois:
        try:
            c = np.asarray(roi.coordinates(), dtype=np.float64)
            if c.ndim != 2 or len(c) < 3:
                continue
            rr, cc = draw_polygon((c[:, 1] - 0.5) / ds, (c[:, 0] - 0.5) / ds, shape=shape)
        except Exception:
            continue
        if len(rr) == 0:
            continue
        n += 1
        lab[rr, cc] = n
    return lab, n


def main():
    args = parse_args()
    import cv2
    from scipy import ndimage as ndi
    from skimage import io

    rng = np.random.default_rng(args.seed)
    with tifffile.TiffFile(args.image) as tif:
        img = tif.asarray()
        dr = imagej_display_range(tif)
    H, W = img.shape
    ds = args.downsampling
    img8 = to_uint8(img, args.scaling, display_range=dr)
    del img
    h, w = int(np.ceil(H / ds)), int(np.ceil(W / ds))
    small = cv2.resize(img8, (w, h), interpolation=cv2.INTER_AREA) if ds > 1 else img8
    del img8
    lab, n_rois = rasterise_rois(args.rois, (h, w), ds)
    print(f'image {H}x{W} -> {h}x{w} at 1/{ds}; {n_rois} ROIs rasterised')

    P = args.patch
    x_split = int(round(w * (1 - args.val_frac)))
    for split in ('train', 'val'):
        for sub in ('images', 'masks'):
            os.makedirs(os.path.join(args.out, split, args.name, sub), exist_ok=True)

    if args.around_objects:
        # keep only the neighbourhood of annotated objects; blank the rest with the image's median grey
        dist = ndi.distance_transform_edt(lab == 0)
        valid = dist <= args.margin
        fill = int(np.median(small[valid]))
        source = small.copy()
        source[~valid] = fill
        print(f'annotated neighbourhood (margin {args.margin} px = {args.margin * ds} px in the original): '
              f'{100 * valid.mean():.1f}% of the image kept, rest blanked with grey value {fill}')
    else:
        source = small
    if args.blank_border > 0:
        b = max(1, int(round(args.blank_border / ds)))
        fill = int(np.median(source))
        source = source.copy()
        source[:b, :] = fill; source[-b:, :] = fill; source[:, :b] = fill; source[:, -b:] = fill
        lab[:b, :] = 0; lab[-b:, :] = 0; lab[:, :b] = 0; lab[:, -b:] = 0
        print(f'blanked a {args.blank_border} px border band')

    def save(split, y0, x0, tag=''):
        y0 = int(min(max(y0, 0), h - P))
        x0 = int(min(max(x0, 0), w - P))
        im = source[y0:y0 + P, x0:x0 + P]
        mk = lab[y0:y0 + P, x0:x0 + P]
        if im.shape != (P, P):
            return False
        ids = np.unique(mk)
        ids = ids[ids > 0]
        lut = np.zeros(int(mk.max()) + 1, np.int32)
        lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
        mk = lut[mk]
        fname = f'{args.name}_ds{ds}_y{y0 * ds:05d}_x{x0 * ds:05d}{tag}.tiff'
        io.imsave(os.path.join(args.out, split, args.name, 'images', fname), im, check_contrast=False)
        io.imsave(os.path.join(args.out, split, args.name, 'masks', fname), mk.astype(np.int32), check_contrast=False)
        return True

    counts = {'train': [0, 0], 'val': [0, 0]}  # [with objects, empty]
    if args.around_objects:
        objs = ndi.find_objects(lab)
        for idx, slc in enumerate(objs, start=1):
            if slc is None:
                continue
            cy = (slc[0].start + slc[0].stop) / 2
            cx = (slc[1].start + slc[1].stop) / 2
            split = 'val' if cx >= x_split else 'train'
            for k in range(args.per_object):
                jy, jx = rng.uniform(-args.jitter, args.jitter, 2) * P
                tag = f'_obj{idx:04d}' + (f'_{k}' if args.per_object > 1 else '')
                if save(split, round(cy - P / 2 + jy), round(cx - P / 2 + jx), tag):
                    counts[split][0] += 1
    else:
        S = args.stride
        empty_candidates = {'train': [], 'val': []}
        for split in ('train', 'val'):
            stride = S if split == 'train' else P
            xs = range(0, x_split - P + 1, stride) if split == 'train' else range(x_split, w - P + 1, stride)
            for y0 in range(0, h - P + 1, stride):
                for x0 in xs:
                    if lab[y0:y0 + P, x0:x0 + P].any():
                        if save(split, y0, x0):
                            counts[split][0] += 1
                    else:
                        empty_candidates[split].append((y0, x0))
            n_obj = counts[split][0]
            n_empty_allowed = int(args.max_empty_frac / max(1 - args.max_empty_frac, 1e-6) * n_obj)
            cand = empty_candidates[split]
            if cand and n_empty_allowed > 0:
                for k in rng.choice(len(cand), size=min(n_empty_allowed, len(cand)), replace=False):
                    if save(split, *cand[k], '_empty'):
                        counts[split][1] += 1

    for split in ('train', 'val'):
        print(f'{split}: {counts[split][0]} patches with objects + {counts[split][1]} empty patches '
              f'-> {os.path.join(args.out, split, args.name)}')
    print(f'validation stripe: x >= {x_split * ds} px in the original image')


if __name__ == '__main__':
    main()
