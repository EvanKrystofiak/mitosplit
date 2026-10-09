# mitosplit

Mitochondria instance segmentation of large 2D TEM montages with
[empanada-napari](https://github.com/volume-em/empanada-napari) (MitoNet), plus

* a scripted fine-tuning workflow that uses hand-corrected **ImageJ ROI sets** as training data,
* batch inference that writes FIJI-ready outputs (label masks, ROI zips, per-object stats), and
* a **notch-pair splitter** that separates fused mitochondria by cutting between facing inward
  notches of the object outline.

Everything runs from the command line in the same conda environment as napari; results match the
empanada-napari "2D Inference" widget because the same inference engine is used.

## The notch-pair splitter (`split_by_notches.py`)

Center-based instance models (Panoptic-DeepLab, used by MitoNet) merge touching mitochondria when
they predict a single center for a cluster. Where two mitochondria touch, however, the outline of
the fused object bends inward on **both** sides of the junction. The splitter uses only that shape
cue, so it can be applied to any label image:

1. **Trace the outline** of the object (2x downsampled, lightly smoothed).
2. **Find inward corners**: at every contour point the turning angle between the arms 30 points back
   and ahead is measured; inward turns of at least 30 degrees are notch candidates (non-maximum
   suppression keeps one per corner). Each notch gets a direction, the corner bisector pointing into
   the object.
3. **Pair notches across a junction**: two notches form a candidate cut if their directions face each
   other along the chord between them (within 60 degrees), the chord lies inside the object, and the
   notches are far apart along the outline relative to the chord (not two corners of one dent).
4. **Neck rule**: the cut is accepted only if both resulting pieces exceed a minimum area and are
   wider than the cut itself (inscribed diameter >= 1.2 x chord length). A waist across a single
   elongated mitochondrion fails this test; two blobs joined at a neck pass it.
5. **Repeat** on the pieces, so chains are cut at every junction.

Validated against hand-corrected ROI sets on five montages (objects from a fine-tuned MitoNet,
before -> after splitting, IoU >= 0.5 matching):

| montage | matched objects | recall | false splits | objects still merging >= 2 |
|---|---|---|---|---|
| A | 46 -> 79 | 0.40 -> 0.68 | 0 -> 0 | 17 -> 18 |
| B | 56 -> 102 | 0.45 -> 0.82 | 0 -> 0 | 20 -> 13 |
| C | 99 -> 117 | 0.79 -> 0.94 | 1 -> 4 | 14 -> 5 |
| D | 54 -> 85 | 0.59 -> 0.93 | 0 -> 0 | 15 -> 4 |
| E | 101 -> 110 | 0.82 -> 0.89 | 4 -> 5 | 8 -> 5 |

Limitations: cuts are straight chords (ends on the true boundary, middle may clip a curved membrane),
and a junction where one membrane runs straight into the neighbour shows a notch on one side only and
is left merged. Convex-hull notch detection and single-notch cuts were tried and were worse.

```
python split_by_notches.py --labels labels.tif --out labels_split.tif [--overlay image.tif]
python postprocess_outputs.py --src seg_folder --images image_folder --out seg_folder_split   # whole folder
python segment_mito.py ... --split-notches                                                     # inline
```

## Installation

```
conda create -n empanada-gpu python=3.11
conda activate empanada-gpu
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118   # match your CUDA driver
pip install -r requirements.txt
```

Notes from setting this up on an 8 GB Quadro with an old driver: `albumentations` is pinned to 1.3.1
(newer versions need a compiler on Windows), `zarr<3` (empanada uses the zarr 2 API), and
`connected-components-3d` must be added by hand (empanada imports it but does not declare it).

## Scripts

| script | purpose |
|---|---|
| `segment_mito.py IMG_OR_FOLDER --model NAME --downsampling N` | batch segmentation with tiled GPU inference; writes `<name>_mito_labels.tif`, `<name>_mito_rois.zip` (ImageJ ROI Manager), `<name>_mito_stats.csv`, `<name>_mito_overlay.png` |
| `split_by_notches.py` / `postprocess_outputs.py` | notch-pair splitting (see above) |
| `evaluate_vs_rois.py --pred LABELS --gt-rois RoiSet.zip --image IMG` | object- and pixel-level scoring against ImageJ ROIs (recall, precision, F1, merges, false splits) with overlays |
| `make_finetune_dataset.py --image IMG --rois RoiSet.zip --out DIR` | training/validation patches from ROI sets; `--around-objects` for partially annotated images, `--blank-border` when edge objects were skipped |
| `finetune_mito.py --train-dir DIR/train --eval-dir DIR/val --base-model MitoNet_v1 --name NEW` | headless fine-tuning (same code path as the napari Finetune widget); registers the new model |
| `register_finetuned_model.py --pth X.pth --yaml X.yaml` | register any empanada model config |
| `input_scaling.py`, `to_8bit.py`, `open_in_napari.py`, `make_center_crop.py` | 8-bit conversion, napari launcher with ROI/label overlays, cropping |
| `split_merged.py` | earlier distance-transform watershed splitter; not recommended (cuts rod-shaped mitochondria), kept for reference |

## Practical notes

* **Convert to 8-bit with a contrast stretch before inference.** MitoNet-type models trained on 8-bit
  data return almost nothing on raw 16-bit TEM montages whose values sit in a narrow band. The
  scripts do this internally (`--input-scaling percentile`); `to_8bit.py` makes copies for napari.
* **Match the scale.** Choose the downsampling so mitochondria are roughly 60-100 px across for the
  network (about 7 nm/px); `--downsampling auto` derives it from the TIF pixel size. Check the
  metadata against something of known size (sarcomere length) because it can be wrong.
* **Large montages in napari**: do not run the 2D Inference widget on a 29k-pixel image; crop first and
  always set a tile size. On an 8 GB GPU use `--tile-size 8192` at downsampling 8.
* **Partial annotations**: when not every mitochondrion in a training image is outlined, build patches
  with `--around-objects` so unannotated ones are blanked rather than learned as background; score
  false positives only near annotations (`evaluate_vs_rois.py --region-dilate`).
* **Fine-tuning recipe that worked**: continue from the previous model with `--max-lr 0.001` for
  ~1500 iterations; a full-rate second cycle (0.003) destabilised an already-tuned model.

## Acknowledgements

Built on empanada-napari and MitoNet by Conrad & Narayan (Volume EM group, NCI). Models trained
here are fine-tuned MitoNet weights and are not part of this repository.
