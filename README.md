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

python trial_run_3_hybrid_inference.py
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
diagnostic_images/*_trial3_diagnostic.png
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
python trial_run_3_hybrid_inference.py
```

Run specific tiles:

```bash
export TRIAL3_TILE_KEYS="train_human_compact_tile_001,train_human_compact_tile_002"
python trial_run_3_hybrid_inference.py
```

For raw image-directory mode without ground-truth masks, Dice/IoU/Precision/Recall
are written as `NaN` with `gt_available=False`. The script still writes prediction
masks, comparison panels, provenance CSVs, and morphology features. If same-name
ground-truth masks are available, set:

```bash
export TRIAL3_MASK_DIR="/path/to/same_name_instance_masks"
```

`hybrid_inference_yolo_sam_cellseg.py` is kept as a backward-compatible wrapper.

## Scale Trial Run 3 to Many Tiles

The main script supports deterministic sharding for large folders, such as ~5000
raw image tiles:

```bash
export TRIAL3_IMAGE_DIR="$HOME/Desktop/prototype5_whole_image_runs/target_region_6262_13752um/tiles"
export TRIAL3_TILE_KEYS=ALL
export TRIAL3_NUM_SHARDS=4
export TRIAL3_SHARD_ID=0
export TRIAL3_SKIP_EXISTING=1
export TRIAL3_OUT_DIR="$HOME/Desktop/prototype5_whole_image_runs/target_region_6262_13752um/trial_run_3_worker_00"

python trial_run_3_hybrid_inference.py
```

Sharding rule:

```text
image_index % TRIAL3_NUM_SHARDS == TRIAL3_SHARD_ID
```

Each worker should use a different `TRIAL3_SHARD_ID` and its own
`TRIAL3_OUT_DIR`.

For one GPU, start conservatively:

```python
N_WORKERS = 1
```

or, if memory allows:

```python
N_WORKERS = 2
```

For multiple GPUs:

```python
GPU_IDS = [0, 1, 2, 3]
```

Too many workers on one GPU can cause CUDA out-of-memory errors because each
worker loads YOLO, SAM3, and CellSeg1.

For large runs, keep prediction masks and CSV outputs but reduce visualization
I/O. This is usually faster than saving two large matplotlib PNGs for every
single tile:

```bash
export TRIAL3_SAVE_COMPARISONS=1
export TRIAL3_COMPARISON_EVERY=25
export TRIAL3_SAVE_DIAGNOSTICS=0
export TRIAL3_VIS_DPI=120
export TRIAL3_CSV_APPEND_EVERY=5
```

Use `TRIAL3_COMPARISON_EVERY=1` and `TRIAL3_SAVE_DIAGNOSTICS=1` only for small
debug runs where you want full visual output for every tile.

The notebook monitor is:

```text
notebooks/monitor_trial_run_3_scale.ipynb
```

Open it in Jupyter, set `REPO`, `BASE_OUT`, `N_WORKERS`, and `GPU_IDS`, then run
the launch and monitor cells. The live dashboard refreshes every 5 seconds
in-place with:

- worker status, assigned/completed images, ETA, elapsed time
- latest comparison images
- aggregate Dice, IoU, precision, recall, specificity
- morphology progress, cells/sec, tiles/sec, and estimated remaining time

It writes one output directory and log file per worker and can merge worker CSV
files into:

```text
merged_trial_run_3_metrics.csv
merged_trial_run_3_source_instances.csv
merged_trial_run_3_final_instances.csv
merged_trial_run_3_morphology_features.csv
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
