# Hybrid YOLO + SAM3 + CellSeg1 Inference

Standalone inference package for CGH adrenal H&E cell-boundary experiments.

The current Trial Run 3 logic is:

- YOLO predicts nuclei.
- SAM3.1 predicts clear-cell boundary candidates.
- CellSeg1 predicts dense cell-boundary candidates, used as residual compact candidates.
- YOLO nuclei validate only CellSeg1 compact candidates.
- SAM3.1 clear-cell candidates are not rejected when YOLO misses a nucleus.
- Final output is SAM3.1 clear cells plus accepted CellSeg1 compact cells.

The script writes segmentation metrics, per-instance provenance, comparison panels,
and per-cell morphology features including cell area, nucleus area, cytoplasm area,
and N/C ratio.

## Clone On GPU Cluster

```bash
cd ~/Desktop
git clone git@github.com:nttssv/inference_yolo_sam_cellseg.git
cd inference_yolo_sam_cellseg
```

HTTPS fallback:

```bash
git clone https://github.com/nttssv/inference_yolo_sam_cellseg.git
```

## Expected External Assets

This repo does not include data, model weights, SAM3, or CellSeg1 source code.
Set paths to the existing cluster assets:

```bash
export TRIAL3_COCO_ROOT="$HOME/Desktop/sam31-cgh-strategy2/dataset/coco_sam3/cgh_pathology_sam31"
export SAM3_REPO="$HOME/Desktop/sam31-cgh-training-data/sam3"
export SAM31_OUTPUT_ROOT="$HOME/Desktop/sam31-cgh-strategy2/outputs/strategy2_41tiles_full_unfreeze_20260625_160623"
export SAM31_CHECKPOINT="$SAM31_OUTPUT_ROOT/checkpoints/checkpoint.pt"

export YOLO_MODEL_PATH="$HOME/Desktop/sam31-cgh-training-data/training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt"

export CELLSEG1_REPO="$HOME/Desktop/1.Data/training_pa_he_annotation_full/outputs/cellseg1_cluster_live/cellseg1_repo"
export CELLSEG1_RUN_DIR="$HOME/Desktop/1.Data/training_pa_he_annotation_full/outputs/cellseg1_cluster_live/cellseg1_cgh_p2_41full_20260625_124306"
```

## Run All Tiles

```bash
export TRIAL3_IMAGE_DIR="$HOME/Desktop/prototype5_whole_image_runs/target_region_6262_13752um/tiles"
export TRIAL3_TILE_KEYS=ALL
export TRIAL3_CELLSEG1_NUCLEUS_DILATION_PX=8
export TRIAL3_SAM31_NMS_IOU_THRESH=0.80

python hybrid_inference_yolo_sam_cellseg.py
```

Outputs are written by default to:

```text
$SAM31_OUTPUT_ROOT/trial_run_3_hybrid_image_dir_all_tiles/
```

If you do not set `TRIAL3_IMAGE_DIR`, the script falls back to the COCO dataset
under `TRIAL3_COCO_ROOT`.

Key files:

```text
trial_run_3_metrics.csv
trial_run_3_summary.csv
trial_run_3_source_instances.csv
trial_run_3_final_instances.csv
trial_run_3_morphology_features.csv
trial_run_3_morphology_summary.csv
comparison_images/*_trial3_compare.png
pred_masks/*_hybrid_instance_mask.png
pred_masks/*_hybrid_class_mask.png
pred_masks/*_yolo_nucleus_mask.png
```

## Quick Debug Run

Run only the first four COCO images:

```bash
export TRIAL3_IMAGE_DIR="$HOME/Desktop/prototype5_whole_image_runs/target_region_6262_13752um/tiles"
export TRIAL3_TILE_KEYS=ALL
export TRIAL3_MAX_IMAGES=4
python hybrid_inference_yolo_sam_cellseg.py
```

Run specific tiles:

```bash
export TRIAL3_TILE_KEYS="train_human_compact_tile_001,train_human_compact_tile_002"
python hybrid_inference_yolo_sam_cellseg.py
```

For raw image-directory mode without ground-truth masks, Dice/IoU/Precision/Recall
are written as `NaN` with `gt_available=False`. The script still writes prediction
masks, comparison panels, provenance CSVs, and morphology features. If same-name
ground-truth masks are available, set:

```bash
export TRIAL3_MASK_DIR="/path/to/same_name_instance_masks"
```

## Useful Parameters

```bash
export TRIAL3_SAM31_SCORE_THRESH=0.30
export TRIAL3_SAM31_MIN_AREA=80
export TRIAL3_SAM31_NMS_IOU_THRESH=0.80

export TRIAL3_CELLSEG1_MIN_AREA=80
export TRIAL3_CELLSEG1_MAX_AREA=8000
export TRIAL3_CELLSEG1_NUCLEUS_DILATION_PX=8
export TRIAL3_MIN_NUCLEUS_OVERLAP_PX=5
export TRIAL3_MAX_OVERLAP_WITH_SAM_CLEAR=0.20

export TRIAL3_YOLO_CONF=0.25
export TRIAL3_YOLO_IOU=0.50
```

## Morphology Output

`trial_run_3_morphology_features.csv` is one row per final cell instance and includes:

- `source_model`: `SAM3.1` or `CellSeg1`
- `final_class_name`: `clear_cell_boundary` or `compact_cell_boundary`
- `cell_area_px`
- `nucleus_area_px`
- `cytoplasm_area_px`
- `nc_ratio`
- `cytoplasm_to_nucleus_ratio`
- `nucleus_count`
- centroid, bbox, perimeter, equivalent diameter

These are pixel-space measurements. Add microns-per-pixel calibration before comparing
absolute sizes across scanners or publications.
