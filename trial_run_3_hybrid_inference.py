#!/usr/bin/env python3
"""Trial Run 3 hybrid inference for CGH adrenal cell boundary segmentation.

Pipeline:
- YOLO segmentation predicts nuclei.
- SAM3.1 predicts clear-cell boundary candidates.
- CellSeg1 predicts dense cell-boundary candidates, used as residual compact
  candidates after removing overlap with SAM3 clear candidates.
- YOLO nuclei gate CellSeg1 compact candidates and every final merged cell.

This script is intentionally path-defaulted for the SUTD GPU cluster layout used
in this project, while every important path can still be overridden with env vars.
"""

from __future__ import annotations

import csv
import faulthandler
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

faulthandler.enable()
TRACEBACK_AFTER_SECONDS = int(os.getenv("TRIAL3_TRACEBACK_AFTER_SECONDS", "180"))
if TRACEBACK_AFTER_SECONDS > 0:
    faulthandler.dump_traceback_later(TRACEBACK_AFTER_SECONDS, repeat=True)

print(
    f"[trial3 bootstrap] pid={os.getpid()} cwd={Path.cwd()} "
    f"traceback_after={TRACEBACK_AFTER_SECONDS}s",
    flush=True,
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from pycocotools import mask as coco_mask
from ultralytics import YOLO

try:
    from IPython.display import Image as IPyImage, display
except Exception:  # pragma: no cover - only used outside notebooks.
    IPyImage = None

    def display(*args, **kwargs):
        return None


PACKAGE_ROOT = Path(__file__).resolve().parent

COCO_ROOT = Path(
    os.getenv(
        "TRIAL3_COCO_ROOT",
        str(PACKAGE_ROOT / "dataset" / "coco_sam3" / "cgh_pathology_sam31"),
    )
).expanduser()
TRIAL3_IMAGE_DIR_RAW = os.getenv("TRIAL3_IMAGE_DIR", "").strip()
TRIAL3_MASK_DIR_RAW = os.getenv("TRIAL3_MASK_DIR", "").strip()
TRIAL3_IMAGE_DIR = Path(TRIAL3_IMAGE_DIR_RAW).expanduser() if TRIAL3_IMAGE_DIR_RAW else None
TRIAL3_MASK_DIR = Path(TRIAL3_MASK_DIR_RAW).expanduser() if TRIAL3_MASK_DIR_RAW else None
USE_IMAGE_DIR = TRIAL3_IMAGE_DIR is not None
TRIAL3_IMAGE_SPLIT = os.getenv("TRIAL3_IMAGE_SPLIT", "inference")
TRIAL3_IMAGE_SUFFIXES = tuple(
    token.strip().lower()
    for token in os.getenv("TRIAL3_IMAGE_SUFFIXES", ".png,.jpg,.jpeg,.tif,.tiff").split(",")
    if token.strip()
)
FAST_IMAGE_INDEX = os.getenv("TRIAL3_FAST_IMAGE_INDEX", "1") == "1"
SAM3_REPO = Path(
    os.getenv("SAM3_REPO", "/home/jovyan/Desktop/sam31-cgh-training-data/sam3")
).expanduser()
SAM31_OUTPUT_ROOT = Path(
    os.getenv(
        "SAM31_OUTPUT_ROOT",
        str(PACKAGE_ROOT / "outputs" / "strategy2_41tiles_full_unfreeze_20260625_160623"),
    )
).expanduser()
SAM31_CHECKPOINT = Path(
    os.getenv("SAM31_CHECKPOINT", str(SAM31_OUTPUT_ROOT / "checkpoints" / "checkpoint.pt"))
).expanduser()
CELLSEG1_REPO = Path(
    os.getenv(
        "CELLSEG1_REPO",
        "/home/jovyan/Desktop/1.Data/training_pa_he_annotation_full/outputs/cellseg1_cluster_live/cellseg1_repo",
    )
).expanduser()
CELLSEG1_RUN_DIR = Path(
    os.getenv(
        "CELLSEG1_RUN_DIR",
        "/home/jovyan/Desktop/1.Data/training_pa_he_annotation_full/outputs/cellseg1_cluster_live/cellseg1_cgh_p2_41full_20260625_124306",
    )
).expanduser()
CELLSEG1_CONFIG_PATH = Path(
    os.getenv("CELLSEG1_CONFIG_PATH", str(CELLSEG1_RUN_DIR / "cellseg1_cgh_p2_runtime_config.yaml"))
).expanduser()
CELLSEG1_LORA = Path(
    os.getenv("CELLSEG1_LORA", str(CELLSEG1_RUN_DIR / "sam_lora_cgh_p2_cell_boundary.pth"))
).expanduser()
YOLO_MODEL_PATH = Path(
    os.getenv(
        "YOLO_MODEL_PATH",
        "/home/jovyan/Desktop/sam31-cgh-training-data/training_data/reference_models/cellseg1_cgh_p2_yolo_best.pt",
    )
).expanduser()

TRIAL3_TILE_KEYS_RAW = os.getenv("TRIAL3_TILE_KEYS", "ALL").strip()
PROCESS_ALL_TILES = TRIAL3_TILE_KEYS_RAW.upper() in {"", "*", "ALL"}
TRIAL3_NAME = os.getenv(
    "TRIAL3_NAME",
    "trial_run_3_hybrid_image_dir_all_tiles"
    if USE_IMAGE_DIR and PROCESS_ALL_TILES
    else "trial_run_3_hybrid_all_tiles"
    if PROCESS_ALL_TILES
    else "trial_run_3_hybrid_selected_tiles",
)
OUT_DIR = Path(os.getenv("TRIAL3_OUT_DIR", str(SAM31_OUTPUT_ROOT / TRIAL3_NAME))).expanduser()
PRED_MASK_DIR = OUT_DIR / "pred_masks"
COMPARE_DIR = OUT_DIR / "comparison_images"
DIAGNOSTIC_DIR = OUT_DIR / "diagnostic_images"

SPLITS = [token.strip() for token in os.getenv("TRIAL3_SPLITS", "train,test").split(",") if token.strip()]
TRIAL3_TILE_KEYS = [
    token.strip()
    for token in TRIAL3_TILE_KEYS_RAW.split(",")
    if token.strip()
]
TRIAL3_MAX_IMAGES = int(os.getenv("TRIAL3_MAX_IMAGES", "0"))
TRIAL3_SHARD_ID = int(os.getenv("TRIAL3_SHARD_ID", "0"))
TRIAL3_NUM_SHARDS = int(os.getenv("TRIAL3_NUM_SHARDS", "1"))
TRIAL3_SKIP_EXISTING = os.getenv("TRIAL3_SKIP_EXISTING", "0") == "1"

SAM31_CLEAR_PROMPTS = [
    token.strip()
    for token in os.getenv(
        "TRIAL3_SAM31_CLEAR_PROMPTS",
        "clear cell boundary,adrenal cortical clear cell boundary",
    ).split(",")
    if token.strip()
]
SAM31_SCORE_THRESH = float(os.getenv("TRIAL3_SAM31_SCORE_THRESH", "0.30"))
SAM31_MIN_AREA = int(os.getenv("TRIAL3_SAM31_MIN_AREA", "80"))
SAM31_NMS_IOU_THRESH = float(os.getenv("TRIAL3_SAM31_NMS_IOU_THRESH", "0.80"))

CELLSEG1_IOU_THRESH = float(os.getenv("TRIAL3_CELLSEG1_IOU_THRESH", "0.80"))
CELLSEG1_STABILITY_THRESH = float(os.getenv("TRIAL3_CELLSEG1_STABILITY_THRESH", "0.60"))
CELLSEG1_MIN_AREA = int(os.getenv("TRIAL3_CELLSEG1_MIN_AREA", "80"))
CELLSEG1_MAX_AREA = int(os.getenv("TRIAL3_CELLSEG1_MAX_AREA", "8000"))
MAX_OVERLAP_WITH_SAM_CLEAR = float(os.getenv("TRIAL3_MAX_OVERLAP_WITH_SAM_CLEAR", "0.20"))
CELLSEG1_NUCLEUS_DILATION_PX = int(os.getenv("TRIAL3_CELLSEG1_NUCLEUS_DILATION_PX", "8"))

YOLO_NUCLEUS_CLASS_ID = int(os.getenv("TRIAL3_YOLO_NUCLEUS_CLASS_ID", "0"))
YOLO_CONF = float(os.getenv("TRIAL3_YOLO_CONF", "0.25"))
YOLO_IOU = float(os.getenv("TRIAL3_YOLO_IOU", "0.50"))
YOLO_MIN_NUCLEUS_AREA = int(os.getenv("TRIAL3_YOLO_MIN_NUCLEUS_AREA", "10"))

CELLSEG1_NUCLEUS_OVERLAP_PX = int(os.getenv("TRIAL3_MIN_NUCLEUS_OVERLAP_PX", "5"))
DISPLAY_IMAGES = os.getenv("TRIAL3_DISPLAY_IMAGES", "0") == "1"
SAVE_COMPARISONS = os.getenv("TRIAL3_SAVE_COMPARISONS", "1") == "1"
SAVE_DIAGNOSTICS = os.getenv("TRIAL3_SAVE_DIAGNOSTICS", "1") == "1"
COMPARISON_EVERY = max(1, int(os.getenv("TRIAL3_COMPARISON_EVERY", "1")))
DIAGNOSTIC_EVERY = max(1, int(os.getenv("TRIAL3_DIAGNOSTIC_EVERY", "1")))
VIS_DPI = int(os.getenv("TRIAL3_VIS_DPI", "180"))
CSV_APPEND_EVERY = max(1, int(os.getenv("TRIAL3_CSV_APPEND_EVERY", "10")))

GT_COLOR = np.array([0, 180, 120], dtype=np.uint8)
NUCLEUS_COLOR = np.array([255, 0, 0], dtype=np.uint8)
SAM_CLEAR_COLOR = np.array([0, 120, 255], dtype=np.uint8)
CELLSEG_COMPACT_COLOR = np.array([255, 180, 0], dtype=np.uint8)
HYBRID_COLOR = np.array([0, 255, 255], dtype=np.uint8)


def assert_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def log_step(message: str) -> None:
    print(f"[trial3] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def load_coco() -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], List[dict]], List[dict]]:
    cat_by_split_and_id: Dict[Tuple[str, int], str] = {}
    anns_by_split_and_image: Dict[Tuple[str, int], List[dict]] = {}
    image_infos: List[dict] = []

    for split in SPLITS:
        coco_path = COCO_ROOT / split / "_annotations.coco.json"
        assert_path(coco_path, f"{split} COCO annotations")
        coco = json.loads(coco_path.read_text())
        for cat in coco["categories"]:
            cat_by_split_and_id[(split, int(cat["id"]))] = cat["name"]
        for ann in coco["annotations"]:
            anns_by_split_and_image.setdefault((split, int(ann["image_id"])), []).append(ann)
        for image_info in coco["images"]:
            row = dict(image_info)
            row["_split"] = split
            row["_source"] = "coco"
            row["_image_path"] = str(COCO_ROOT / split / row["file_name"])
            row["_gt_mask_path"] = ""
            row.update(parse_tile_geometry(row["file_name"]))
            image_infos.append(row)

    image_infos.sort(key=lambda x: (x["_split"], x["file_name"]))
    return cat_by_split_and_id, anns_by_split_and_image, image_infos


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TRIAL3_IMAGE_SUFFIXES and not path.name.startswith("._")


def find_mask_for_image(image_path: Path) -> Path | None:
    if TRIAL3_MASK_DIR is None:
        return None
    for suffix in TRIAL3_IMAGE_SUFFIXES + (".png", ".tif", ".tiff"):
        candidate = TRIAL3_MASK_DIR / f"{image_path.stem}{suffix}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def parse_tile_geometry(file_name: str) -> Dict[str, int | float]:
    match = re.search(r"(?:^|[_-])x(?P<x>\d+)[_-]y(?P<y>\d+)[_-]w(?P<w>\d+)[_-]h(?P<h>\d+)", file_name)
    if not match:
        return {"tile_x": np.nan, "tile_y": np.nan, "tile_w": np.nan, "tile_h": np.nan}
    return {
        "tile_x": int(match.group("x")),
        "tile_y": int(match.group("y")),
        "tile_w": int(match.group("w")),
        "tile_h": int(match.group("h")),
    }


def load_image_dir() -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], List[dict]], List[dict]]:
    assert TRIAL3_IMAGE_DIR is not None
    assert_path(TRIAL3_IMAGE_DIR, "tile image directory")
    if TRIAL3_MASK_DIR is not None:
        assert_path(TRIAL3_MASK_DIR, "optional tile mask directory")

    image_infos = []
    image_files = sorted(p for p in TRIAL3_IMAGE_DIR.iterdir() if is_image_file(p))
    for idx, image_path in enumerate(image_files, start=1):
        tile_geometry = parse_tile_geometry(image_path.name)
        if FAST_IMAGE_INDEX and not pd.isna(tile_geometry["tile_w"]) and not pd.isna(tile_geometry["tile_h"]):
            width = int(tile_geometry["tile_w"])
            height = int(tile_geometry["tile_h"])
        else:
            with Image.open(image_path) as image:
                width, height = image.size
        mask_path = find_mask_for_image(image_path)
        image_infos.append(
            {
                "id": idx,
                "file_name": image_path.name,
                "width": int(width),
                "height": int(height),
                **tile_geometry,
                "_split": TRIAL3_IMAGE_SPLIT,
                "_source": "image_dir",
                "_image_path": str(image_path),
                "_gt_mask_path": str(mask_path) if mask_path else "",
            }
        )
    if not image_infos:
        raise FileNotFoundError(f"No image files with suffixes {TRIAL3_IMAGE_SUFFIXES} found in {TRIAL3_IMAGE_DIR}")
    return {}, {}, image_infos


def load_inputs() -> Tuple[Dict[Tuple[str, int], str], Dict[Tuple[str, int], List[dict]], List[dict]]:
    return load_image_dir() if USE_IMAGE_DIR else load_coco()


def select_images(image_infos: Sequence[dict]) -> List[dict]:
    if PROCESS_ALL_TILES:
        selected = list(image_infos)
        return selected[:TRIAL3_MAX_IMAGES] if TRIAL3_MAX_IMAGES > 0 else selected

    wanted = set(TRIAL3_TILE_KEYS)
    selected = []
    for info in image_infos:
        split = info["_split"]
        stem = Path(info["file_name"]).stem
        key = f"{split}_{stem}"
        if key in wanted or stem in wanted:
            selected.append(info)
    if not selected:
        raise RuntimeError(f"No selected trial tiles found: {TRIAL3_TILE_KEYS}")
    return selected[:TRIAL3_MAX_IMAGES] if TRIAL3_MAX_IMAGES > 0 else selected


def apply_sharding(image_infos: Sequence[dict]) -> List[dict]:
    if TRIAL3_NUM_SHARDS < 1:
        raise ValueError(f"TRIAL3_NUM_SHARDS must be >= 1, got {TRIAL3_NUM_SHARDS}")
    if TRIAL3_SHARD_ID < 0 or TRIAL3_SHARD_ID >= TRIAL3_NUM_SHARDS:
        raise ValueError(
            f"TRIAL3_SHARD_ID must be in [0, {TRIAL3_NUM_SHARDS - 1}], got {TRIAL3_SHARD_ID}"
        )
    return [
        image_info
        for idx, image_info in enumerate(image_infos)
        if idx % TRIAL3_NUM_SHARDS == TRIAL3_SHARD_ID
    ]


def expected_output_paths(tile_key: str) -> Dict[str, Path]:
    return {
        "hybrid": PRED_MASK_DIR / f"{tile_key}_hybrid_instance_mask.png",
        "class": PRED_MASK_DIR / f"{tile_key}_hybrid_class_mask.png",
        "nucleus": PRED_MASK_DIR / f"{tile_key}_yolo_nucleus_mask.png",
        "sam_clear": PRED_MASK_DIR / f"{tile_key}_sam31_clear_mask.png",
        "compact": PRED_MASK_DIR / f"{tile_key}_cellseg1_residual_compact_mask.png",
    }


def should_skip_tile(tile_key: str) -> bool:
    if not TRIAL3_SKIP_EXISTING:
        return False
    return expected_output_paths(tile_key)["hybrid"].exists()


def decode_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = coco_mask.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        return coco_mask.decode(rle).astype(bool)
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        decoded = coco_mask.decode(rles)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        return decoded.astype(bool)
    raise TypeError(f"Unsupported COCO segmentation type: {type(segmentation)!r}")


def gt_mask_for_categories(
    image_info: dict,
    categories: set[str],
    cat_by_split_and_id: Dict[Tuple[str, int], str],
    anns_by_split_and_image: Dict[Tuple[str, int], List[dict]],
) -> np.ndarray:
    split = image_info["_split"]
    height, width = int(image_info["height"]), int(image_info["width"])
    out = np.zeros((height, width), dtype=np.uint16)
    label = 1

    for ann in anns_by_split_and_image.get((split, int(image_info["id"])), []):
        cat_name = cat_by_split_and_id.get((split, int(ann["category_id"])), "")
        if cat_name not in categories:
            continue
        mask = decode_coco_segmentation(ann["segmentation"], height, width)
        pixels = mask & (out == 0)
        if int(pixels.sum()) < 20:
            continue
        out[pixels] = label
        label += 1
    return out


def read_instance_mask(path: Path, image_rgb: np.ndarray) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = resize_mask_to_image(mask.astype(np.uint16), image_rgb)
    return mask.astype(np.uint16)


def load_gt_masks_for_image(
    image_info: dict,
    image_rgb: np.ndarray,
    cat_by_split_and_id: Dict[Tuple[str, int], str],
    anns_by_split_and_image: Dict[Tuple[str, int], List[dict]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, bool], str]:
    height, width = image_rgb.shape[:2]
    empty = np.zeros((height, width), dtype=np.uint16)

    mask_path_raw = str(image_info.get("_gt_mask_path", "") or "")
    if mask_path_raw:
        gt_all = read_instance_mask(Path(mask_path_raw), image_rgb)
        return (
            empty.copy(),
            empty.copy(),
            gt_all,
            {
                "SAM3_clear_vs_GT_clear": False,
                "CellSeg1_residual_compact_vs_GT_compact": False,
                "Hybrid_vs_GT_all_boundaries": True,
            },
            "mask_dir",
        )

    if image_info.get("_source") == "coco":
        gt_clear = gt_mask_for_categories(
            image_info, {"clear_cell_boundary"}, cat_by_split_and_id, anns_by_split_and_image
        )
        gt_compact = gt_mask_for_categories(
            image_info, {"compact_cell_boundary"}, cat_by_split_and_id, anns_by_split_and_image
        )
        gt_all = gt_mask_for_categories(
            image_info,
            {"clear_cell_boundary", "compact_cell_boundary"},
            cat_by_split_and_id,
            anns_by_split_and_image,
        )
        return (
            gt_clear,
            gt_compact,
            gt_all,
            {
                "SAM3_clear_vs_GT_clear": True,
                "CellSeg1_residual_compact_vs_GT_compact": True,
                "Hybrid_vs_GT_all_boundaries": True,
            },
            "coco",
        )

    return (
        empty.copy(),
        empty.copy(),
        empty.copy(),
        {
            "SAM3_clear_vs_GT_clear": False,
            "CellSeg1_residual_compact_vs_GT_compact": False,
            "Hybrid_vs_GT_all_boundaries": False,
        },
        "unavailable",
    )


def resize_mask_to_image(mask: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    height, width = img_rgb.shape[:2]
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask.astype(np.uint16), (width, height), interpolation=cv2.INTER_NEAREST)


def mask_boundaries(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return boundary & (mask > 0)


def draw_instance_mask(img_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    mask = resize_mask_to_image(mask, img_rgb)
    out = img_rgb.copy()
    overlay = img_rgb.copy()
    overlay[mask > 0] = color
    out[mask_boundaries(mask)] = color

    for label in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
        mask_bin = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(map(int, color)), 2)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def draw_class_mask(
    img_rgb: np.ndarray,
    class_mask: np.ndarray,
    colors_by_class: Dict[int, np.ndarray],
    alpha: float = 0.32,
) -> np.ndarray:
    class_mask = resize_mask_to_image(class_mask, img_rgb)
    out = img_rgb.copy()
    overlay = img_rgb.copy()

    for class_id, color in colors_by_class.items():
        pixels = class_mask == int(class_id)
        if not pixels.any():
            continue
        overlay[pixels] = color
        contours, _ = cv2.findContours(pixels.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(map(int, color)), 2)

    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def text_panel_like(img_rgb: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    panel = np.full_like(img_rgb, 245, dtype=np.uint8)
    y = 36
    for line in lines:
        cv2.putText(
            panel,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
        y += 34
    return panel


def binary_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    gt = gt_mask > 0
    pred = pred_mask > 0
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    tn = int((~gt & ~pred).sum())
    return {
        "tp_px": tp,
        "fp_px": fp,
        "fn_px": fn,
        "tn_px": tn,
        "dice": (2 * tp) / (2 * tp + fp + fn + 1e-8),
        "iou": tp / (tp + fp + fn + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall_sensitivity": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "gt_area_px": int(gt.sum()),
        "pred_area_px": int(pred.sum()),
        "gt_instances": int(gt_mask.max()),
        "pred_instances": int(pred_mask.max()),
    }


def unavailable_gt_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    pred = pred_mask > 0
    gt = gt_mask > 0
    return {
        "tp_px": np.nan,
        "fp_px": np.nan,
        "fn_px": np.nan,
        "tn_px": np.nan,
        "dice": np.nan,
        "iou": np.nan,
        "precision": np.nan,
        "recall_sensitivity": np.nan,
        "specificity": np.nan,
        "gt_area_px": int(gt.sum()),
        "pred_area_px": int(pred.sum()),
        "gt_instances": int(gt_mask.max()) if gt_mask.size else 0,
        "pred_instances": int(pred_mask.max()) if pred_mask.size else 0,
    }


def metrics_with_gt_availability(gt_mask: np.ndarray, pred_mask: np.ndarray, gt_available: bool) -> dict:
    row = binary_metrics(gt_mask, pred_mask) if gt_available else unavailable_gt_metrics(gt_mask, pred_mask)
    row["gt_available"] = bool(gt_available)
    return row


def relabel_instance_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.uint16)
    label = 1
    for old in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
        pixels = mask == old
        if int(pixels.sum()) == 0:
            continue
        out[pixels] = label
        label += 1
    return out


def centroid_of_mask(mask_bool: np.ndarray) -> Tuple[int, int] | None:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def dilate_binary_mask(mask_bool: np.ndarray, dilation_px: int) -> np.ndarray:
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if dilation_px <= 0:
        return mask_bool
    kernel_size = 2 * dilation_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = int((a & b).sum())
    union = int((a | b).sum())
    return intersection / (union + 1e-8)


def cell_has_nucleus(
    cell_bool: np.ndarray,
    nucleus_mask: np.ndarray,
    dilation_px: int = CELLSEG1_NUCLEUS_DILATION_PX,
) -> Tuple[bool, int, str]:
    """Validate a cell candidate using dilated YOLO nuclei geometry."""
    dilated_cell = dilate_binary_mask(cell_bool, dilation_px)
    nucleus_binary = nucleus_mask > 0
    overlap_px = int((dilated_cell & nucleus_binary).sum())

    for nuc_label in sorted(int(v) for v in np.unique(nucleus_mask) if int(v) != 0):
        centroid = centroid_of_mask(nucleus_mask == nuc_label)
        if centroid is None:
            continue
        x, y = centroid
        if 0 <= y < dilated_cell.shape[0] and 0 <= x < dilated_cell.shape[1] and dilated_cell[y, x]:
            return True, overlap_px, "inside"

    if overlap_px >= CELLSEG1_NUCLEUS_OVERLAP_PX:
        return True, overlap_px, "overlap"

    return False, overlap_px, "rejected"


def yolo_nucleus_mask(yolo_model: YOLO, image_path: Path, image_rgb: np.ndarray, tile_key: str = "") -> Tuple[np.ndarray, List[dict]]:
    result = yolo_model.predict(
        source=str(image_path),
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
        retina_masks=True,
    )[0]

    height, width = image_rgb.shape[:2]
    nucleus_mask = np.zeros((height, width), dtype=np.uint16)
    rows = []
    next_label = 1

    if result.masks is None:
        print(f"{tile_key} | YOLO nucleus: none")
        return nucleus_mask, rows

    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()

    for idx, raw_mask in enumerate(masks):
        cls = int(classes[idx])
        conf = float(confs[idx])
        if cls != YOLO_NUCLEUS_CLASS_ID:
            continue
        binary = raw_mask > 0.5
        binary = resize_mask_to_image(binary.astype(np.uint8), image_rgb) > 0
        area = int(binary.sum())
        if area < YOLO_MIN_NUCLEUS_AREA:
            continue
        pixels = binary & (nucleus_mask == 0)
        if int(pixels.sum()) < YOLO_MIN_NUCLEUS_AREA:
            continue
        nucleus_mask[pixels] = next_label
        rows.append(
            {
                "source_model": "YOLO",
                "class_name": "nucleus",
                "label": next_label,
                "score": conf,
                "area_px": int(pixels.sum()),
            }
        )
        next_label += 1

    print(f"{tile_key} | YOLO nucleus kept={int(nucleus_mask.max())}")
    return nucleus_mask, rows


def sam31_autocast_context():
    if not torch.cuda.is_available():
        return nullcontext()
    raw_dtype = os.getenv("TRIAL3_SAM31_AMP_DTYPE", "float16").lower()
    dtype = torch.float16 if raw_dtype in {"fp16", "float16", "half"} else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def masks_to_numpy_stack(masks) -> np.ndarray:
    if hasattr(masks, "detach"):
        arr = masks.detach().float().cpu().numpy()
    else:
        arr = np.asarray(masks)
    arr = np.squeeze(arr)
    if arr.size == 0:
        return np.zeros((0, 1, 1), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim > 3:
        arr = arr.reshape((-1,) + arr.shape[-2:])
    return arr


def scores_to_numpy(scores, n: int) -> np.ndarray:
    if scores is None:
        return np.ones(n)
    if hasattr(scores, "detach"):
        arr = scores.detach().float().cpu().numpy()
    else:
        arr = np.asarray(scores, dtype=float)
    arr = np.squeeze(arr).reshape(-1)
    if arr.size < n:
        arr = np.pad(arr, (0, n - arr.size), constant_values=1.0)
    return arr[:n]


def sam3_clear_mask(
    processor,
    pil_image: Image.Image,
    image_rgb: np.ndarray,
    tile_key: str = "",
) -> Tuple[np.ndarray, List[dict]]:
    with torch.inference_mode(), sam31_autocast_context():
        state = processor.set_image(pil_image)

    height, width = pil_image.height, pil_image.width
    pred = np.zeros((height, width), dtype=np.uint16)
    rows = []
    next_label = 1
    candidates = []

    for prompt in SAM31_CLEAR_PROMPTS:
        with torch.inference_mode(), sam31_autocast_context():
            output = processor.set_text_prompt(state=state, prompt=prompt)

        mask_stack = masks_to_numpy_stack(output.get("masks", []))
        scores = scores_to_numpy(output.get("scores"), len(mask_stack))

        for idx, (raw_mask, score) in enumerate(zip(mask_stack, scores)):
            if float(score) < SAM31_SCORE_THRESH:
                continue
            binary = raw_mask > 0
            binary = resize_mask_to_image(binary.astype(np.uint8), image_rgb) > 0
            area = int(binary.sum())
            if area < SAM31_MIN_AREA:
                continue
            candidates.append(
                {
                    "mask": binary,
                    "prompt": prompt,
                    "sam3_index": idx,
                    "score": float(score),
                    "raw_area_px": area,
                }
            )

        print(
            f"{tile_key} | SAM3.1 clear | prompt={prompt!r} raw={len(mask_stack)} "
            f"after_score_area={sum(1 for row in candidates if row['prompt'] == prompt)}"
        )

    kept_masks = []
    removed_duplicate = 0
    removed_consumed_overlap = 0

    for candidate in sorted(candidates, key=lambda row: row["score"], reverse=True):
        max_iou_with_kept = max((mask_iou(candidate["mask"], kept) for kept in kept_masks), default=0.0)
        if 0.0 < SAM31_NMS_IOU_THRESH < 1.0 and max_iou_with_kept > SAM31_NMS_IOU_THRESH:
            removed_duplicate += 1
            continue

        pixels = candidate["mask"] & (pred == 0)
        area = int(pixels.sum())
        if area < SAM31_MIN_AREA:
            removed_consumed_overlap += 1
            continue

        pred[pixels] = next_label
        kept_masks.append(candidate["mask"])
        rows.append(
            {
                "source_model": "SAM3.1",
                "class_name": "clear_cell_boundary",
                "label": next_label,
                "prompt": candidate["prompt"],
                "score": candidate["score"],
                "area_px": area,
                "raw_area_px": candidate["raw_area_px"],
                "sam3_index": candidate["sam3_index"],
                "max_iou_with_kept": max_iou_with_kept,
                "nucleus_overlap_px": np.nan,
                "nucleus_gate_type": "not_applied",
            }
        )
        next_label += 1

    print(
        f"{tile_key} | SAM3.1 clear final kept={int(pred.max())} "
        f"removed_duplicate={removed_duplicate} removed_consumed_overlap={removed_consumed_overlap}"
    )
    return pred, rows


def cellseg1_predict_one(
    image_path: Path,
    cellseg_config: dict,
    read_image_to_numpy,
    resize_image,
    mask_generator,
    sam_output_to_mask,
) -> Tuple[np.ndarray, np.ndarray]:
    image = resize_image(read_image_to_numpy(image_path), cellseg_config["resize_size"])
    with torch.no_grad():
        output = mask_generator.generate(image)
    if output == []:
        pred = np.zeros_like(image[:, :, 0], dtype=np.uint16)
    else:
        pred = sam_output_to_mask(output)
    return image, pred.astype(np.uint16)


def filter_cellseg_compact_candidates(
    cellseg_mask: np.ndarray,
    sam_clear_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    image_rgb: np.ndarray,
) -> Tuple[np.ndarray, List[dict]]:
    cellseg_mask = resize_mask_to_image(cellseg_mask, image_rgb)
    sam_clear_mask = resize_mask_to_image(sam_clear_mask, image_rgb)
    nucleus_mask = resize_mask_to_image(nucleus_mask, image_rgb)

    compact = np.zeros(cellseg_mask.shape, dtype=np.uint16)
    rows = []
    next_label = 1
    sam_clear_binary = sam_clear_mask > 0
    removed_no_dilated_nucleus = 0
    removed_overlap_clear = 0
    removed_area = 0

    for old_label in sorted(int(v) for v in np.unique(cellseg_mask) if int(v) != 0):
        candidate = cellseg_mask == old_label
        area = int(candidate.sum())
        if area < CELLSEG1_MIN_AREA or area > CELLSEG1_MAX_AREA:
            removed_area += 1
            continue
        overlap = int((candidate & sam_clear_binary).sum())
        overlap_ratio = overlap / (area + 1e-8)
        if overlap_ratio > MAX_OVERLAP_WITH_SAM_CLEAR:
            removed_overlap_clear += 1
            continue
        has_nuc, nuc_overlap, gate_type = cell_has_nucleus(
            candidate,
            nucleus_mask,
            dilation_px=CELLSEG1_NUCLEUS_DILATION_PX,
        )
        if not has_nuc:
            removed_no_dilated_nucleus += 1
            continue
        compact[candidate] = next_label
        rows.append(
            {
                "source_model": "CellSeg1",
                "class_name": "compact_cell_boundary",
                "label": next_label,
                "prompt": "cellseg1_residual_mask",
                "score": np.nan,
                "area_px": area,
                "overlap_with_sam_clear_ratio": overlap_ratio,
                "nucleus_overlap_px": nuc_overlap,
                "nucleus_gate_type": gate_type,
            }
        )
        next_label += 1

    print(
        f"CellSeg1 residual compact kept={int(compact.max())} "
        f"removed_no_dilated_nucleus={removed_no_dilated_nucleus} "
        f"removed_overlap_clear={removed_overlap_clear} removed_area={removed_area}"
    )
    return compact, rows


def combine_clear_and_compact(sam_clear: np.ndarray, compact_candidate: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    sam_clear = relabel_instance_mask(sam_clear)
    compact_candidate = relabel_instance_mask(compact_candidate)
    final = np.zeros(sam_clear.shape, dtype=np.uint16)
    class_mask = np.zeros(sam_clear.shape, dtype=np.uint8)
    rows = []
    next_label = 1

    for old in sorted(int(v) for v in np.unique(sam_clear) if int(v) != 0):
        pixels = sam_clear == old
        final[pixels] = next_label
        class_mask[pixels] = 1
        rows.append(
            {
                "final_label": next_label,
                "final_class_id": 1,
                "final_class_name": "clear_cell_boundary",
                "source_model": "SAM3.1",
                "source_label": old,
                "area_px": int(pixels.sum()),
            }
        )
        next_label += 1

    for old in sorted(int(v) for v in np.unique(compact_candidate) if int(v) != 0):
        pixels = (compact_candidate == old) & (final == 0)
        if int(pixels.sum()) < CELLSEG1_MIN_AREA:
            continue
        final[pixels] = next_label
        class_mask[pixels] = 2
        rows.append(
            {
                "final_label": next_label,
                "final_class_id": 2,
                "final_class_name": "compact_cell_boundary",
                "source_model": "CellSeg1",
                "source_label": old,
                "area_px": int(pixels.sum()),
            }
        )
        next_label += 1
    return final, class_mask, rows


def filter_final_cells_by_nucleus(
    hybrid_mask: np.ndarray,
    hybrid_class_mask: np.ndarray,
    final_rows: List[dict],
    nucleus_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Keep only final cells that have a YOLO nucleus anchor."""
    source_by_label = {int(row["final_label"]): row for row in final_rows}
    filtered = np.zeros(hybrid_mask.shape, dtype=np.uint16)
    filtered_class = np.zeros(hybrid_class_mask.shape, dtype=np.uint8)
    filtered_rows = []
    removed_by_class: Dict[str, int] = {}
    next_label = 1

    for old in sorted(int(v) for v in np.unique(hybrid_mask) if int(v) != 0):
        pixels = hybrid_mask == old
        source = dict(source_by_label.get(old, {}))
        class_id = int(source.get("final_class_id", 0))
        class_name = source.get("final_class_name", "unknown")
        has_nuc, nuc_overlap, gate_type = cell_has_nucleus(
            pixels,
            nucleus_mask,
            dilation_px=CELLSEG1_NUCLEUS_DILATION_PX,
        )
        if not has_nuc:
            removed_by_class[class_name] = removed_by_class.get(class_name, 0) + 1
            continue
        filtered[pixels] = next_label
        filtered_class[pixels] = class_id
        source["original_final_label"] = old
        source["final_label"] = next_label
        source["area_px"] = int(pixels.sum())
        source["nucleus_overlap_px"] = nuc_overlap
        source["nucleus_gate_type"] = gate_type
        filtered_rows.append(source)
        next_label += 1

    print(
        f"Final nucleus gate kept={int(filtered.max())} "
        f"removed={sum(removed_by_class.values())} removed_by_class={removed_by_class}"
    )
    return filtered, filtered_class, filtered_rows


def mask_bbox(mask_bool: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def mask_perimeter(mask_bool: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def assign_nuclei_to_cell(cell_bool: np.ndarray, nucleus_mask: np.ndarray) -> Tuple[List[int], int, int, int]:
    assigned_labels = []
    overlap_area = 0
    full_area = 0
    centroid_hits = 0

    for nuc_label in sorted(int(v) for v in np.unique(nucleus_mask) if int(v) != 0):
        nucleus = nucleus_mask == nuc_label
        overlap_px = int((cell_bool & nucleus).sum())
        centroid = centroid_of_mask(nucleus)
        centroid_inside = False
        if centroid is not None:
            x, y = centroid
            centroid_inside = 0 <= y < cell_bool.shape[0] and 0 <= x < cell_bool.shape[1] and bool(cell_bool[y, x])
        if overlap_px >= CELLSEG1_NUCLEUS_OVERLAP_PX or centroid_inside:
            assigned_labels.append(nuc_label)
            overlap_area += overlap_px
            full_area += int(nucleus.sum())
            centroid_hits += int(centroid_inside)

    return assigned_labels, overlap_area, full_area, centroid_hits


def extract_morphology_rows(
    hybrid_mask: np.ndarray,
    hybrid_class_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    final_rows: List[dict],
    split: str,
    tile_id: str,
    file_name: str,
    image_path: Path,
    tile_geometry: Dict[str, int | float],
) -> List[dict]:
    source_by_final_label = {int(row["final_label"]): row for row in final_rows}
    rows = []

    for final_label in sorted(int(v) for v in np.unique(hybrid_mask) if int(v) != 0):
        cell = hybrid_mask == final_label
        cell_area = int(cell.sum())
        if cell_area == 0:
            continue

        source = source_by_final_label.get(final_label, {})
        class_ids = hybrid_class_mask[cell]
        final_class_id = int(np.bincount(class_ids.astype(np.int64)).argmax()) if class_ids.size else 0
        final_class_name = source.get(
            "final_class_name",
            "clear_cell_boundary" if final_class_id == 1 else "compact_cell_boundary" if final_class_id == 2 else "unknown",
        )
        nucleus_labels, nucleus_area, nucleus_full_area, centroid_hits = assign_nuclei_to_cell(cell, nucleus_mask)
        cytoplasm_area = max(cell_area - nucleus_area, 0)
        centroid = centroid_of_mask(cell)
        bbox_x, bbox_y, bbox_w, bbox_h = mask_bbox(cell)
        perimeter = mask_perimeter(cell)

        rows.append(
            {
                "trial": TRIAL3_NAME,
                "split": split,
                "tile_id": tile_id,
                "file_name": file_name,
                "image_path": str(image_path),
                **tile_geometry,
                "final_label": final_label,
                "final_class_id": final_class_id,
                "final_class_name": final_class_name,
                "source_model": source.get("source_model", "unknown"),
                "source_label": source.get("source_label", np.nan),
                "cell_area_px": cell_area,
                "cell_perimeter_px": perimeter,
                "cell_equiv_diameter_px": float(np.sqrt(4 * cell_area / np.pi)),
                "cell_centroid_x": centroid[0] if centroid else np.nan,
                "cell_centroid_y": centroid[1] if centroid else np.nan,
                "cell_bbox_x": bbox_x,
                "cell_bbox_y": bbox_y,
                "cell_bbox_w": bbox_w,
                "cell_bbox_h": bbox_h,
                "nucleus_count": len(nucleus_labels),
                "nucleus_labels": ";".join(str(label) for label in nucleus_labels),
                "nucleus_area_px": nucleus_area,
                "nucleus_full_area_px": nucleus_full_area,
                "nucleus_centroid_hits": centroid_hits,
                "nucleus_equiv_diameter_px": float(np.sqrt(4 * nucleus_area / np.pi)) if nucleus_area > 0 else np.nan,
                "cytoplasm_area_px": cytoplasm_area,
                "nc_ratio": nucleus_area / (cell_area + 1e-8),
                "cytoplasm_to_nucleus_ratio": cytoplasm_area / nucleus_area if nucleus_area > 0 else np.nan,
            }
        )

    return rows


def load_sam3_processor():
    if str(SAM3_REPO) not in sys.path:
        sys.path.insert(0, str(SAM3_REPO))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        eval_mode=True,
        enable_segmentation=True,
    )
    checkpoint = torch.load(SAM31_CHECKPOINT, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    print("SAM3 missing:", len(missing))
    print("SAM3 unexpected:", len(unexpected))
    if missing or unexpected:
        raise RuntimeError(f"SAM3 checkpoint mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return Sam3Processor(model)


def load_cellseg1():
    if str(CELLSEG1_REPO) not in sys.path:
        sys.path.insert(0, str(CELLSEG1_REPO))
    from data.utils import read_image_to_numpy, resize_image
    from predict import load_model_from_config, sam_output_to_mask
    from segment_anything import SamAutomaticMaskGeneratorOptMaskNMS
    from set_environment import set_env

    with CELLSEG1_CONFIG_PATH.open("r") as handle:
        cellseg_config = yaml.safe_load(handle)
    cellseg_config["result_pth_path"] = str(CELLSEG1_LORA)
    cellseg_config["pred_iou_thresh"] = CELLSEG1_IOU_THRESH
    cellseg_config["stability_score_thresh"] = CELLSEG1_STABILITY_THRESH
    cellseg_config["deterministic"] = False

    set_env(
        cellseg_config["deterministic"],
        cellseg_config["seed"],
        cellseg_config["allow_tf32_on_cudnn"],
        cellseg_config["allow_tf32_on_matmul"],
    )
    model = load_model_from_config(cellseg_config, empty_lora=False)
    model.eval()
    model_sam = model.sam if hasattr(model, "sam") else model
    mask_generator = SamAutomaticMaskGeneratorOptMaskNMS(
        model=model_sam,
        points_per_side=cellseg_config["points_per_side"],
        points_per_batch=cellseg_config["points_per_batch"],
        crop_n_layers=cellseg_config["crop_n_layers"],
        crop_n_points_downscale_factor=cellseg_config["crop_n_points_downscale_factor"],
        box_nms_thresh=cellseg_config["box_nms_thresh"],
        crop_nms_thresh=cellseg_config["crop_nms_thresh"],
        pred_iou_thresh=cellseg_config["pred_iou_thresh"],
        min_mask_region_area=cellseg_config["min_mask_region_area"],
        max_mask_region_area_ratio=cellseg_config["max_mask_region_area_ratio"],
        stability_score_thresh=cellseg_config["stability_score_thresh"],
        stability_score_offset=cellseg_config["stability_score_offset"],
    )
    return cellseg_config, read_image_to_numpy, resize_image, mask_generator, sam_output_to_mask


def save_csv(path: Path, rows: List[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow(["empty"])


def append_csv_rows(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame(rows).to_csv(path, mode="a", header=write_header, index=False)


def load_existing_csv_rows(path: Path) -> List[dict]:
    if not (TRIAL3_SKIP_EXISTING and path.exists() and path.stat().st_size > 0):
        return []
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: could not load existing CSV for resume: {path} ({exc})")
        return []
    if list(df.columns) == ["empty"]:
        return []
    return df.to_dict("records")


def main() -> None:
    log_step("main started")
    required_paths = [
        (SAM3_REPO, "SAM3 repo"),
        (SAM31_CHECKPOINT, "SAM3 checkpoint"),
        (CELLSEG1_REPO, "CellSeg1 repo"),
        (CELLSEG1_CONFIG_PATH, "CellSeg1 config"),
        (CELLSEG1_LORA, "CellSeg1 LoRA checkpoint"),
        (YOLO_MODEL_PATH, "YOLO model"),
    ]
    if USE_IMAGE_DIR:
        required_paths.append((TRIAL3_IMAGE_DIR, "tile image directory"))
        if TRIAL3_MASK_DIR is not None:
            required_paths.append((TRIAL3_MASK_DIR, "optional tile mask directory"))
    else:
        required_paths.append((COCO_ROOT, "COCO root"))

    for path, label in required_paths:
        log_step(f"checking {label}: {path}")
        assert_path(path, label)

    log_step("creating output directories")
    PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    log_step("loading input index")
    cat_by_split_and_id, anns_by_split_and_image, all_images = load_inputs()
    trial_images = select_images(all_images)
    total_images_before_sharding = len(trial_images)
    trial_images = apply_sharding(trial_images)

    print("Trial Run 3 output:", OUT_DIR, flush=True)
    print("Input mode:", "image_dir" if USE_IMAGE_DIR else "coco", flush=True)
    if USE_IMAGE_DIR:
        print("Tile image directory:", TRIAL3_IMAGE_DIR, flush=True)
        print("Optional mask directory:", TRIAL3_MASK_DIR or "(not set)", flush=True)
        print("TRIAL3_FAST_IMAGE_INDEX:", int(FAST_IMAGE_INDEX), flush=True)
    print("Total images before sharding:", total_images_before_sharding, flush=True)
    print("TRIAL3_SHARD_ID:", TRIAL3_SHARD_ID, flush=True)
    print("TRIAL3_NUM_SHARDS:", TRIAL3_NUM_SHARDS, flush=True)
    print("Images assigned to this worker:", len(trial_images), flush=True)
    assigned_keys = [f"{x['_split']}_{Path(x['file_name']).stem}" for x in trial_images]
    preview_count = min(20, len(assigned_keys))
    print("Assigned image keys preview:", assigned_keys[:preview_count], flush=True)
    if len(assigned_keys) > preview_count:
        print(f"Assigned image keys truncated: showing {preview_count}/{len(assigned_keys)}", flush=True)
    print("TRIAL3_SKIP_EXISTING:", int(TRIAL3_SKIP_EXISTING), flush=True)
    print("SAM3 prompts:", SAM31_CLEAR_PROMPTS, flush=True)
    print("SAM3 score threshold:", SAM31_SCORE_THRESH, flush=True)

    metrics_csv = OUT_DIR / "trial_run_3_metrics.csv"
    source_csv = OUT_DIR / "trial_run_3_source_instances.csv"
    final_csv = OUT_DIR / "trial_run_3_final_instances.csv"
    summary_csv = OUT_DIR / "trial_run_3_summary.csv"
    morphology_csv = OUT_DIR / "trial_run_3_morphology_features.csv"
    morphology_summary_csv = OUT_DIR / "trial_run_3_morphology_summary.csv"

    log_step(f"loading YOLO model: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    print("YOLO names:", yolo_model.names, flush=True)
    log_step("loading SAM3 processor")
    processor = load_sam3_processor()
    log_step("loading cached CellSeg1 predictor")
    cellseg_config, read_image_to_numpy, resize_image, cellseg_mask_generator, sam_output_to_mask = load_cellseg1()
    print("CellSeg1 LoRA:", cellseg_config["result_pth_path"], flush=True)

    metrics_rows: List[dict] = load_existing_csv_rows(metrics_csv)
    source_instance_rows: List[dict] = load_existing_csv_rows(source_csv)
    final_instance_rows: List[dict] = load_existing_csv_rows(final_csv)
    morphology_rows: List[dict] = load_existing_csv_rows(morphology_csv)
    pending_metrics_rows: List[dict] = []
    pending_source_instance_rows: List[dict] = []
    pending_final_instance_rows: List[dict] = []
    pending_morphology_rows: List[dict] = []
    saved_images: List[Path] = []

    for tile_index, image_info in enumerate(trial_images, start=1):
        split = image_info["_split"]
        tile_id = Path(image_info["file_name"]).stem
        tile_key = f"{split}_{tile_id}"
        if should_skip_tile(tile_key):
            print("\n" + "=" * 100)
            print(f"Skipping existing tile: {tile_key}")
            print("Existing hybrid mask:", expected_output_paths(tile_key)["hybrid"])
            continue

        image_path = Path(image_info["_image_path"])
        tile_geometry = {key: image_info.get(key, np.nan) for key in ("tile_x", "tile_y", "tile_w", "tile_h")}
        print("\n" + "=" * 100)
        print("Trial 3 running:", tile_key)
        started = time.time()

        pil_image = Image.open(image_path).convert("RGB")
        original_rgb = np.asarray(pil_image)
        gt_clear, gt_compact, gt_all, gt_available_by_model, gt_source = load_gt_masks_for_image(
            image_info,
            original_rgb,
            cat_by_split_and_id,
            anns_by_split_and_image,
        )

        nucleus_mask, nucleus_rows = yolo_nucleus_mask(yolo_model, image_path, original_rgb, tile_key=tile_key)
        sam_clear, sam_rows = sam3_clear_mask(processor, pil_image, original_rgb, tile_key=tile_key)
        _, cellseg_raw = cellseg1_predict_one(
            image_path,
            cellseg_config,
            read_image_to_numpy,
            resize_image,
            cellseg_mask_generator,
            sam_output_to_mask,
        )
        cellseg_raw = resize_mask_to_image(cellseg_raw, original_rgb)
        compact_candidate, compact_rows = filter_cellseg_compact_candidates(
            cellseg_raw, sam_clear, nucleus_mask, original_rgb
        )
        pre_gate_hybrid_mask, pre_gate_hybrid_class_mask, pre_gate_final_rows = combine_clear_and_compact(
            sam_clear, compact_candidate
        )
        hybrid_mask, hybrid_class_mask, final_rows = filter_final_cells_by_nucleus(
            pre_gate_hybrid_mask,
            pre_gate_hybrid_class_mask,
            pre_gate_final_rows,
            nucleus_mask,
        )
        elapsed = time.time() - started
        tile_morphology_rows = extract_morphology_rows(
            hybrid_mask,
            hybrid_class_mask,
            nucleus_mask,
            final_rows,
            split,
            tile_id,
            image_info["file_name"],
            image_path,
            tile_geometry,
        )
        morphology_rows.extend(tile_morphology_rows)
        pending_morphology_rows.extend(tile_morphology_rows)

        paths = expected_output_paths(tile_key)
        Image.fromarray(hybrid_mask.astype(np.uint16)).save(paths["hybrid"])
        Image.fromarray(hybrid_class_mask.astype(np.uint8)).save(paths["class"])
        Image.fromarray(nucleus_mask.astype(np.uint16)).save(paths["nucleus"])
        Image.fromarray(sam_clear.astype(np.uint16)).save(paths["sam_clear"])
        Image.fromarray(compact_candidate.astype(np.uint16)).save(paths["compact"])

        for model_name, pred_mask, gt_mask in [
            ("SAM3_clear_vs_GT_clear", sam_clear, gt_clear),
            ("CellSeg1_residual_compact_vs_GT_compact", compact_candidate, gt_compact),
            ("Hybrid_vs_GT_all_boundaries", hybrid_mask, gt_all),
        ]:
            row = metrics_with_gt_availability(
                gt_mask,
                pred_mask,
                gt_available_by_model.get(model_name, False),
            )
            row.update(
                {
                    "trial": TRIAL3_NAME,
                    "model": model_name,
                    "split": split,
                    "tile_id": tile_id,
                    "file_name": image_info["file_name"],
                    "image_path": str(image_path),
                    **tile_geometry,
                    "gt_source": gt_source,
                    "inference_time_sec": elapsed,
                    "yolo_nucleus_instances": int(nucleus_mask.max()),
                    "sam31_clear_instances": int(sam_clear.max()),
                    "cellseg1_residual_compact_instances": int(compact_candidate.max()),
                    "hybrid_instances_before_nucleus_gate": int(pre_gate_hybrid_mask.max()),
                    "hybrid_instances": int(hybrid_mask.max()),
                    "hybrid_removed_no_nucleus": int(pre_gate_hybrid_mask.max()) - int(hybrid_mask.max()),
                    "sam31_score_thresh": SAM31_SCORE_THRESH,
                    "sam31_min_area": SAM31_MIN_AREA,
                    "sam31_nms_iou_thresh": SAM31_NMS_IOU_THRESH,
                    "cellseg1_min_area": CELLSEG1_MIN_AREA,
                    "cellseg1_max_area": CELLSEG1_MAX_AREA,
                    "cellseg1_nucleus_dilation_px": CELLSEG1_NUCLEUS_DILATION_PX,
                    "cellseg1_nucleus_overlap_px": CELLSEG1_NUCLEUS_OVERLAP_PX,
                    "max_overlap_with_sam_clear": MAX_OVERLAP_WITH_SAM_CLEAR,
                    "yolo_conf": YOLO_CONF,
                    "yolo_iou": YOLO_IOU,
                }
            )
            metrics_rows.append(row)
            pending_metrics_rows.append(row)

        for row in nucleus_rows + sam_rows + compact_rows:
            row.update(
                {
                    "trial": TRIAL3_NAME,
                    "split": split,
                    "tile_id": tile_id,
                    "file_name": image_info["file_name"],
                    "image_path": str(image_path),
                    **tile_geometry,
                }
            )
            source_instance_rows.append(row)
            pending_source_instance_rows.append(row)
        for row in final_rows:
            row.update(
                {
                    "trial": TRIAL3_NAME,
                    "split": split,
                    "tile_id": tile_id,
                    "file_name": image_info["file_name"],
                    "image_path": str(image_path),
                    **tile_geometry,
                }
            )
            final_instance_rows.append(row)
            pending_final_instance_rows.append(row)

        write_comparison = SAVE_COMPARISONS and (tile_index == 1 or tile_index % COMPARISON_EVERY == 0)
        write_diagnostic = SAVE_DIAGNOSTICS and (tile_index == 1 or tile_index % DIAGNOSTIC_EVERY == 0)
        if write_comparison or write_diagnostic:
            gt_available = gt_available_by_model.get("Hybrid_vs_GT_all_boundaries", False)
            gt_panel = (
                draw_instance_mask(original_rgb, gt_all, GT_COLOR)
                if gt_available
                else text_panel_like(original_rgb, ["GT unavailable", "raw tile mode", "metrics: NaN"])
            )

            if write_comparison:
                panels = [
                    original_rgb,
                    gt_panel,
                    draw_instance_mask(original_rgb, sam_clear, SAM_CLEAR_COLOR),
                    draw_instance_mask(original_rgb, compact_candidate, CELLSEG_COMPACT_COLOR),
                    draw_instance_mask(original_rgb, hybrid_mask, HYBRID_COLOR),
                ]
                titles = [
                    f"{tile_key}: original",
                    f"GT all n={int(gt_all.max())}" if gt_available else "GT unavailable",
                    f"SAM3 clear n={int(sam_clear.max())}",
                    f"CellSeg1 residual compact n={int(compact_candidate.max())}",
                    f"Hybrid n={int(hybrid_mask.max())}",
                ]
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                for ax, panel, title in zip(axes, panels, titles):
                    ax.imshow(panel)
                    ax.set_title(title)
                    ax.axis("off")
                fig.tight_layout()
                out_img = COMPARE_DIR / f"{tile_key}_trial3_compare.png"
                fig.savefig(out_img, dpi=VIS_DPI)
                plt.close(fig)
                saved_images.append(out_img)
                if DISPLAY_IMAGES and IPyImage is not None:
                    display(IPyImage(filename=str(out_img)))

            if write_diagnostic:
                hybrid_by_class = draw_class_mask(
                    original_rgb,
                    hybrid_class_mask,
                    {
                        1: SAM_CLEAR_COLOR,
                        2: CELLSEG_COMPACT_COLOR,
                    },
                )
                diagnostic_panels = [
                    original_rgb,
                    draw_instance_mask(original_rgb, nucleus_mask, NUCLEUS_COLOR),
                    draw_instance_mask(original_rgb, sam_clear, SAM_CLEAR_COLOR),
                    draw_instance_mask(original_rgb, compact_candidate, CELLSEG_COMPACT_COLOR),
                    draw_instance_mask(original_rgb, hybrid_mask, HYBRID_COLOR),
                    hybrid_by_class,
                ]
                diagnostic_titles = [
                    f"{tile_key}: original",
                    f"YOLO nuclei n={int(nucleus_mask.max())}",
                    f"SAM3 clear n={int(sam_clear.max())}",
                    f"CellSeg1 compact n={int(compact_candidate.max())}",
                    f"Final hybrid n={int(hybrid_mask.max())}",
                    "Hybrid by class: blue=SAM3, amber=CellSeg1",
                ]
                diag_fig, diag_axes = plt.subplots(1, 6, figsize=(30, 5))
                for ax, panel, title in zip(diag_axes, diagnostic_panels, diagnostic_titles):
                    ax.imshow(panel)
                    ax.set_title(title)
                    ax.axis("off")
                diag_fig.tight_layout()
                diag_img = DIAGNOSTIC_DIR / f"{tile_key}_trial3_diagnostic.png"
                diag_fig.savefig(diag_img, dpi=VIS_DPI)
                plt.close(diag_fig)

        if tile_index % CSV_APPEND_EVERY == 0:
            append_csv_rows(metrics_csv, pending_metrics_rows)
            append_csv_rows(source_csv, pending_source_instance_rows)
            append_csv_rows(final_csv, pending_final_instance_rows)
            append_csv_rows(morphology_csv, pending_morphology_rows)
            pending_metrics_rows.clear()
            pending_source_instance_rows.clear()
            pending_final_instance_rows.clear()
            pending_morphology_rows.clear()

    append_csv_rows(metrics_csv, pending_metrics_rows)
    append_csv_rows(source_csv, pending_source_instance_rows)
    append_csv_rows(final_csv, pending_final_instance_rows)
    append_csv_rows(morphology_csv, pending_morphology_rows)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_csv, index=False)
    save_csv(source_csv, source_instance_rows)
    save_csv(final_csv, final_instance_rows)
    save_csv(morphology_csv, morphology_rows)

    if "model" in metrics_df.columns and not metrics_df.empty:
        summary_df = metrics_df.groupby("model").agg(
            n_tiles=("tile_id", "count"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            iou_mean=("iou", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall_sensitivity", "mean"),
            pred_instances_mean=("pred_instances", "mean"),
            gt_instances_mean=("gt_instances", "mean"),
            sam31_clear_instances_mean=("sam31_clear_instances", "mean"),
            cellseg1_residual_compact_instances_mean=("cellseg1_residual_compact_instances", "mean"),
            hybrid_instances_mean=("hybrid_instances", "mean"),
            inference_time_sec_mean=("inference_time_sec", "mean"),
        ).reset_index()
        summary_df.to_csv(summary_csv, index=False)
    else:
        summary_df = pd.DataFrame()
        save_csv(summary_csv, [])

    if morphology_rows:
        morphology_df = pd.DataFrame(morphology_rows)
        morphology_summary_df = morphology_df.groupby(["split", "final_class_name", "source_model"]).agg(
            n_cells=("final_label", "count"),
            cell_area_px_mean=("cell_area_px", "mean"),
            cell_area_px_std=("cell_area_px", "std"),
            nucleus_area_px_mean=("nucleus_area_px", "mean"),
            cytoplasm_area_px_mean=("cytoplasm_area_px", "mean"),
            nc_ratio_mean=("nc_ratio", "mean"),
            nc_ratio_std=("nc_ratio", "std"),
            nucleus_count_mean=("nucleus_count", "mean"),
            cell_equiv_diameter_px_mean=("cell_equiv_diameter_px", "mean"),
            nucleus_equiv_diameter_px_mean=("nucleus_equiv_diameter_px", "mean"),
        ).reset_index()
        morphology_summary_df.to_csv(morphology_summary_csv, index=False)
    else:
        morphology_summary_df = pd.DataFrame()
        save_csv(morphology_summary_csv, [])

    print("\nTrial run 3 complete.")
    print("Output:", OUT_DIR)
    print("Metrics:", metrics_csv)
    print("Summary:", summary_csv)
    print("Source instances:", source_csv)
    print("Final instances:", final_csv)
    print("Morphology features:", morphology_csv)
    print("Morphology summary:", morphology_summary_csv)
    print("Comparison images:", COMPARE_DIR)
    print("Diagnostic images:", DIAGNOSTIC_DIR)
    print(summary_df)
    if not morphology_summary_df.empty:
        print(morphology_summary_df)


if __name__ == "__main__":
    main()
