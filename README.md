# Hybrid YOLO + SAM3 + CellSeg1 Inference

Standalone inference package for CGH adrenal H&E cell-boundary experiments.

The current Trial Run 3 logic is:

- YOLO predicts nuclei.
- SAM3.1 predicts clear-cell boundary candidates.
- CellSeg1 predicts dense cell-boundary candidates, used as residual compact candidates.
- YOLO nuclei validate CellSeg1 compact candidates before merging.
- Final output is SAM3.1 clear cells plus accepted CellSeg1 compact cells.
- Every final cell, from either SAM3.1 or CellSeg1, must have a YOLO nucleus
  anchor after the final nucleus gate. Cells without a nucleus anchor are not
  kept and are not counted in morphology.

The script writes segmentation metrics, per-instance provenance, comparison panels,
and per-cell morphology features including cell area, nucleus area, cytoplasm area,
and N/C ratio.

YOLO, SAM3.1, and CellSeg1 are loaded once per worker. CellSeg1 keeps a cached
mask generator for all tiles assigned to that worker instead of rebuilding the
model for every tile.

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
export TRIAL3_COMPARISON_EVERY=5
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

## Supervised Long Runs

For multi-hour cluster runs, use the supervisor instead of relying on the
notebook launcher. It restarts a worker when the process exits with a native
CUDA/PyTorch crash such as return code `-11`, or when no completed tile appears
for a long stall window. Restarts are safe because the worker runs with
`TRIAL3_SKIP_EXISTING=1` and skips only tiles with complete mask outputs and
complete metrics rows.

One A100 80GB running two workers on GPU 0:

```bash
cd ~/Desktop/inference_yolo_sam_cellseg
python supervise_trial_run_3.py \
  --workers 2 \
  --gpu-ids 0 \
  --tile-dir "$HOME/Desktop/prototype5_whole_image_runs/target_region_6262_13752um/tiles" \
  --base-out "$HOME/Desktop/prototype5_whole_image_runs/trial_run_3_yolo_sam_cellseg" \
  --stale-seconds 900 \
  --traceback-seconds 600 \
  --csv-append-every 1
```

`Timeout (0:03:00)!` in older logs comes from Python `faulthandler` dumping a
stack trace after 180 seconds. It is diagnostic output, not a restart policy.
The supervisor defaults to `--traceback-seconds 600` and only restarts a running
worker after `--stale-seconds` without new completed tiles.

Useful live checks:

```bash
watch -n 2 nvidia-smi
tail -f "$HOME/Desktop/prototype5_whole_image_runs/trial_run_3_yolo_sam_cellseg/worker_00.log"
tail -f "$HOME/Desktop/prototype5_whole_image_runs/trial_run_3_yolo_sam_cellseg/worker_01.log"
```

Stop the supervisor with `Ctrl-C`; it will terminate its worker processes.

## Run Trial Run 3 inference on cohort tile folders

The cohort runner processes the configured slide tile folders one slide at a
time, starts sharded Trial Run 3 workers for each slide, writes worker status and
preview files, and merges worker CSVs after each slide completes. It does not
change the YOLO, SAM3.1, or CellSeg1 inference rules in
`trial_run_3_hybrid_inference.py`.

Set the external model/source paths first:

```bash
cd ~/Desktop/inference_yolo_sam_cellseg

export SAM3_REPO="$HOME/Desktop/sam31-cgh-training-data/sam3"
export SAM31_OUTPUT_ROOT="$HOME/Desktop/sam31-cgh-strategy2/outputs/strategy2_41tiles_full_unfreeze_20260625_160623"
export SAM31_CHECKPOINT="$SAM31_OUTPUT_ROOT/checkpoints/checkpoint.pt"

export YOLO_MODEL_PATH="$HOME/Desktop/sam31-cgh-training-data/training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt"

export CELLSEG1_REPO="$HOME/Desktop/1.Data/training_pa_he_annotation_full/outputs/cellseg1_cluster_live/cellseg1_repo"
export CELLSEG1_RUN_DIR="$HOME/Desktop/1.Data/training_pa_he_annotation_full/outputs/cellseg1_cluster_live/cellseg1_cgh_p2_41full_20260625_124306"
```

Review the configured slides and tile counts:

```bash
python scripts/run_cohort_trial3_inference.py \
  --config configs/cohort_inference_slides.yaml \
  --dry-run
```

Run the cohort:

```bash
python scripts/run_cohort_trial3_inference.py \
  --config configs/cohort_inference_slides.yaml \
  --run
```

Useful overrides:

```bash
python scripts/run_cohort_trial3_inference.py \
  --config configs/cohort_inference_slides.yaml \
  --run \
  --workers 2 \
  --gpu-ids 0 \
  --poll-seconds 30
```

The default output root is:

```text
~/Desktop/Create_Cell_Atlas_Inference_Output/<slide_name>/
```

Each slide output contains `worker_00/`, `worker_01/`, `merged/`, `preview/`,
`logs/`, `_status/`, and one of `_RUNNING.json`, `_COMPLETED.json`, or
`_FAILED.json`. Worker status files are written to:

```text
~/Desktop/Create_Cell_Atlas_Inference_Output/<slide_name>/_status/worker_<id>.json
~/Desktop/Create_Cell_Atlas_Inference_Output/<slide_name>/_status/worker_<id>_timing.csv
```

Merged slide CSVs are written to:

```text
~/Desktop/Create_Cell_Atlas_Inference_Output/<slide_name>/merged/
```

Per-tile final-cell boundary GeoJSON files are written to each worker output:

```text
~/Desktop/Create_Cell_Atlas_Inference_Output/<slide_name>/worker_<id>/geojson/
```

After a slide completes, all worker GeoJSON files are also copied to:

```text
~/Desktop/Create_Cell_Atlas_Inference_Output/<slide_name>/merged/geojson/
```

GeoJSON coordinates default to slide/global pixel coordinates when tile
filenames contain `x..._y...`; otherwise they fall back to local tile
coordinates. To create GeoJSON for tiles that were already processed before
GeoJSON output existed, run:

```bash
python scripts/run_cohort_trial3_inference.py \
  --config configs/cohort_inference_slides.yaml \
  --run \
  --geojson-only
```

Use `--overwrite-geojson` with `--geojson-only` to rewrite existing GeoJSON
files.

Open the cohort dashboard notebook:

```bash
jupyter notebook notebooks/01_cohort_trial3_inference_monitor.ipynb
```

The notebook loads `configs/cohort_inference_slides.yaml`, shows a dry-run table,
launches the same reusable script, and monitors status, latest previews, ETA,
tiles/min, cells/sec, failed tiles, and recent logs. It is safe to reopen the
notebook: running state is read from JSON status files, and the runner lock
prevents duplicate active cohort runs.

`MTO107` / `target_107` is included in the config as disabled and is skipped by
default because it has already completed.

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

# Used by the monitoring notebook to display live morphology in um and um^2.
export TRIAL3_MICRONS_PER_PIXEL=0.25
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
