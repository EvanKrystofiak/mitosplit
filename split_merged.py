r"""
split_merged.py - split predicted objects that are really several touching mitochondria fused together.

Candidates are objects that are noticeably concave (solidity below --solidity) and larger than the
typical object. Each candidate is cut by a watershed on its distance transform, seeded at distance
peaks at least ~one typical radius apart. The cut is kept only if every piece is a reasonable size and
the pieces are more convex than the original blob, so single elongated mitochondria stay intact.

  python split_merged.py --labels seg\img_mito_labels.tif --out seg\img_mito_labels_split.tif
Used by segment_mito.py --split-merged.
"""
import argparse

import numpy as np
import tifffile


def split_merged_objects(inst, solidity_thr=0.85, area_factor=1.3, min_distance_frac=0.8, h_frac=0.35,
                         min_piece_frac=0.30, max_pieces=10):
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.segmentation import watershed

    props = regionprops(inst)
    if not props:
        return inst, 0
    areas = np.array([p.area for p in props])
    med_area = float(np.median(areas))
    r_med = np.sqrt(med_area / np.pi)
    out = inst.copy()
    next_id = int(inst.max()) + 1
    n_split = 0
    for p in props:
        if p.area < area_factor * med_area or p.solidity >= solidity_thr:
            continue
        sl = p.slice
        m = inst[sl] == p.label
        edt = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
        peaks = peak_local_max(edt, min_distance=int(max(3, min_distance_frac * r_med)),
                               threshold_abs=h_frac * float(edt.max()), labels=m.astype(np.int32),
                               exclude_border=False, num_peaks=max_pieces)
        if len(peaks) < 2:
            continue
        markers = np.zeros(m.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        lab = watershed(-edt, markers, mask=m)
        pieces = [lab == k for k in range(1, len(peaks) + 1)]
        if any(pc.sum() < min_piece_frac * med_area for pc in pieces):
            continue
        sol = [regionprops(pc.astype(np.int32))[0].solidity for pc in pieces]
        if float(np.mean(sol)) <= p.solidity + 0.02:
            continue
        for pc in pieces[1:]:
            out[sl][pc] = next_id
            next_id += 1
        n_split += 1
    return out, n_split


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--solidity', type=float, default=0.85)
    p.add_argument('--min-piece-frac', type=float, default=0.30)
    p.add_argument('--h-frac', type=float, default=0.35)
    p.add_argument('--min-distance-frac', type=float, default=0.8)
    args = p.parse_args()
    inst = tifffile.imread(args.labels).astype(np.int32)
    out, n = split_merged_objects(inst, solidity_thr=args.solidity, min_piece_frac=args.min_piece_frac,
                                  h_frac=args.h_frac, min_distance_frac=args.min_distance_frac)
    data = out.astype(np.uint16) if out.max() < 65535 else out.astype(np.uint32)
    tifffile.imwrite(args.out, data, compression='lzw')
    print(f'{n} objects split; {int(inst.max())} -> {int(out.max())} objects -> {args.out}')


if __name__ == '__main__':
    main()
