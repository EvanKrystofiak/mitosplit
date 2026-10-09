"""
input_scaling.py - convert raw TEM images (typically 16-bit with a narrow intensity range) into the
8-bit, contrast-stretched form that the fine-tuned mitochondria model was trained on.

Modes
  none        keep the image as is (what empanada-napari does by default)
  minmax      linear stretch of [vmin, vmax] -> [0, 255]; vmin/vmax default to the image min/max
  imagej      like minmax but uses the display range stored in the TIF by ImageJ (falls back to minmax)
  percentile  linear stretch of [p_low, p_high] percentiles -> [0, 255]        (default 0.5 / 99.5)
  zscore      shift/scale so the image mean -> 147 and std -> 32.5, i.e. the intensity statistics the
              MitoNet family of models expects (norms mean 0.57571, std 0.12765 of 255)
"""
import numpy as np

MODES = ('none', 'minmax', 'imagej', 'percentile', 'zscore')
MITONET_MEAN, MITONET_STD = 0.57571 * 255.0, 0.12765 * 255.0


def _stretch(img, lo, hi):
    img = img.astype(np.float32, copy=False)
    out = (img - lo) / max(float(hi - lo), 1e-6) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def to_uint8(img, mode='percentile', p_low=0.5, p_high=99.5, vmin=None, vmax=None, display_range=None,
             sample=4):
    """Return an 8-bit version of img according to mode. `display_range` is (min, max) from ImageJ metadata.
    Percentiles / statistics are computed on a sub-sampled copy (every `sample`-th pixel) for speed."""
    if mode == 'none':
        return img
    sub = img[::sample, ::sample] if sample > 1 else img
    if mode == 'minmax':
        lo = float(sub.min()) if vmin is None else vmin
        hi = float(sub.max()) if vmax is None else vmax
        return _stretch(img, lo, hi)
    if mode == 'imagej':
        if display_range is not None and display_range[1] > display_range[0]:
            return _stretch(img, float(display_range[0]), float(display_range[1]))
        return to_uint8(img, 'minmax', vmin=vmin, vmax=vmax, sample=sample)
    if mode == 'percentile':
        lo, hi = np.percentile(sub, (p_low, p_high))
        return _stretch(img, float(lo), float(hi))
    if mode == 'zscore':
        m, s = float(sub.mean()), float(sub.std())
        out = (img.astype(np.float32, copy=False) - m) / max(s, 1e-6) * MITONET_STD + MITONET_MEAN
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f'unknown scaling mode {mode!r}; choose from {MODES}')


def imagej_display_range(tif):
    """(min, max) display range stored by ImageJ in a tifffile.TiffFile, or None."""
    try:
        md = tif.imagej_metadata or {}
        if 'min' in md and 'max' in md:
            return float(md['min']), float(md['max'])
    except Exception:
        pass
    return None
