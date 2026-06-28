#!/usr/bin/env python3
"""Run and monitor Trial Run 3 inference across a cohort of slide tile folders.

This script intentionally treats trial_run_3_hybrid_inference.py as the inference
unit. It only sets per-slide/per-worker environment variables, supervises worker
processes, writes lightweight status files, copies one latest preview per worker,
and merges worker CSVs after a slide finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, TextIO

import pandas as pd
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cohort_inference_slides.yaml"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
HYBRID_SUFFIX = "_hybrid_instance_mask.png"
OUTPUT_SUFFIXES = (
    "_hybrid_instance_mask.png",
    "_hybrid_class_mask.png",
    "_yolo_nucleus_mask.png",
    "_sam31_clear_mask.png",
    "_cellseg1_residual_compact_mask.png",
)
MERGE_SPECS = {
    "trial_run_3_metrics.csv": "merged_trial_run_3_metrics.csv",
    "trial_run_3_source_instances.csv": "merged_trial_run_3_source_instances.csv",
    "trial_run_3_final_instances.csv": "merged_trial_run_3_final_instances.csv",
    "trial_run_3_morphology_features.csv": "merged_trial_run_3_morphology_features.csv",
}
TIMING_COLUMNS = [
    "timestamp",
    "slide_name",
    "worker_id",
    "tile_name",
    "duration_seconds",
    "success",
    "num_instances",
]
STOP_REQUESTED = False


@dataclass
class WorkerRuntime:
    worker_id: int
    tiles_total: int
    process: subprocess.Popen | None = None
    log_handle: TextIO | None = None
    restarts: int = 0
    last_completed: int = 0
    last_progress_ts: float = 0.0
    last_log_mtime: float = 0.0
    launch_ts: float = 0.0
    started_at_iso: str = ""
    status: str = "pending"
    error: str = ""
    failed_tile_recorded: bool = False


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def expand_path(raw: str | Path | None, base: Path | None = None) -> Path | None:
    if raw is None:
        return None
    text = os.path.expandvars(str(raw)).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base or REPO_ROOT) / path
    return path.resolve()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = expand_path(path, REPO_ROOT)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    config["_config_path"] = str(config_path)
    return config


def get_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key) or {}
    return value if isinstance(value, dict) else {}


def parse_gpu_ids(raw: Any) -> list[int]:
    if isinstance(raw, str):
        gpu_ids = [int(token.strip()) for token in raw.split(",") if token.strip()]
    else:
        gpu_ids = [int(x) for x in (raw or [])]
    if not gpu_ids:
        raise ValueError("cohort.gpu_ids must contain at least one GPU id")
    return gpu_ids


def settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    cohort = get_section(config, "cohort")
    trial3 = get_section(config, "trial3")
    monitor = get_section(config, "monitor")
    repo = expand_path(cohort.get("repo", REPO_ROOT), REPO_ROOT) or REPO_ROOT
    script = expand_path(cohort.get("inference_script", "trial_run_3_hybrid_inference.py"), repo)
    output_root = expand_path(
        cohort.get("output_root", "~/Desktop/Create_Cell_Atlas_Inference_Output"),
        REPO_ROOT,
    )
    suffixes = cohort.get("image_suffixes", sorted(IMAGE_SUFFIXES))
    image_suffixes = {str(s).lower() for s in suffixes if str(s).strip()}
    workers = int(cohort.get("workers", 1))
    if workers < 1:
        raise ValueError("cohort.workers must be >= 1")
    return {
        "repo": repo,
        "script": script,
        "output_root": output_root,
        "workers": workers,
        "gpu_ids": parse_gpu_ids(cohort.get("gpu_ids", [0])),
        "poll_seconds": int(cohort.get("poll_seconds", 30)),
        "stale_seconds": int(cohort.get("stale_seconds", 900)),
        "startup_stale_seconds": int(cohort.get("startup_stale_seconds", 3600)),
        "restart_delay_seconds": int(cohort.get("restart_delay_seconds", 20)),
        "kill_timeout_seconds": int(cohort.get("kill_timeout_seconds", 30)),
        "max_restarts": int(cohort.get("max_restarts", -1)),
        "continue_on_error": bool(cohort.get("continue_on_error", False)),
        "preflight_paths": bool(cohort.get("preflight_paths", True)),
        "image_suffixes": image_suffixes,
        "refresh_seconds": int(monitor.get("refresh_seconds", 5)),
        "log_tail_lines": int(monitor.get("log_tail_lines", 40)),
        "trial3": trial3,
    }


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cohort = config.setdefault("cohort", {})
    if args.output_root:
        cohort["output_root"] = args.output_root
    if args.workers is not None:
        cohort["workers"] = args.workers
    if args.gpu_ids:
        cohort["gpu_ids"] = [int(token.strip()) for token in args.gpu_ids.split(",") if token.strip()]
    if args.poll_seconds is not None:
        cohort["poll_seconds"] = args.poll_seconds
    if args.stale_seconds is not None:
        cohort["stale_seconds"] = args.stale_seconds
    if args.no_preflight:
        cohort["preflight_paths"] = False
    return config


def iter_slides(config: dict[str, Any], only_slide: str | None = None) -> list[dict[str, Any]]:
    slides = []
    for raw in config.get("slides") or []:
        if not isinstance(raw, dict):
            continue
        slide = dict(raw)
        tile_dir = expand_path(slide.get("tile_dir"), REPO_ROOT)
        name = str(slide.get("name") or infer_slide_name(tile_dir) or "unnamed_slide")
        if only_slide and name != only_slide:
            continue
        slide["name"] = name
        slide["_tile_dir_path"] = tile_dir
        slide["enabled"] = bool(slide.get("enabled", True))
        slides.append(slide)
    return slides


def infer_slide_name(tile_dir: Path | None) -> str | None:
    if tile_dir is None:
        return None
    try:
        if tile_dir.name == "accepted_tiles" and tile_dir.parent.name == "tiles":
            return tile_dir.parent.parent.name
    except IndexError:
        return None
    return tile_dir.name


def list_tiles(tile_dir: Path, image_suffixes: set[str] = IMAGE_SUFFIXES) -> list[Path]:
    if not tile_dir.exists():
        return []
    return sorted(
        path
        for path in tile_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_suffixes and not path.name.startswith("._")
    )


def tile_key(path: Path, split: str = "inference") -> str:
    return f"{split}_{path.stem}"


def assigned_tiles(tiles: list[Path], worker_id: int, workers: int) -> list[Path]:
    return [path for index, path in enumerate(tiles) if index % workers == worker_id]


def slide_out_dir(settings: dict[str, Any], slide_name: str) -> Path:
    return settings["output_root"] / slide_name


def worker_out_dir(slide_dir: Path, worker_id: int) -> Path:
    return slide_dir / f"worker_{worker_id:02d}"


def worker_log_path(slide_dir: Path, worker_id: int) -> Path:
    return slide_dir / "logs" / f"worker_{worker_id:02d}.log"


def worker_status_path(slide_dir: Path, worker_id: int) -> Path:
    return slide_dir / "_status" / f"worker_{worker_id:02d}.json"


def worker_timing_path(slide_dir: Path, worker_id: int) -> Path:
    return slide_dir / "_status" / f"worker_{worker_id:02d}_timing.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == ["empty"]:
                return []
            return [dict(row) for row in reader]
    except Exception:
        return []


def tile_keys_with_complete_outputs(out_dir: Path) -> set[str]:
    pred_dir = out_dir / "pred_masks"
    if not pred_dir.exists():
        return set()
    complete = set()
    for hybrid_path in pred_dir.glob(f"*{HYBRID_SUFFIX}"):
        key = hybrid_path.name[: -len(HYBRID_SUFFIX)]
        if all((pred_dir / f"{key}{suffix}").exists() for suffix in OUTPUT_SUFFIXES):
            complete.add(key)
    return complete


def tile_keys_with_complete_metrics(out_dir: Path) -> set[str]:
    counts: dict[str, int] = {}
    for row in read_csv_rows(out_dir / "trial_run_3_metrics.csv"):
        split = (row.get("split") or "").strip()
        tile_id = (row.get("tile_id") or "").strip()
        if not split or not tile_id:
            continue
        key = f"{split}_{tile_id}"
        counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count >= 3}


def count_completed(out_dir: Path) -> int:
    return len(tile_keys_with_complete_outputs(out_dir) & tile_keys_with_complete_metrics(out_dir))


def count_completed_for_slide(slide_dir: Path, workers: int) -> int:
    return sum(count_completed(worker_out_dir(slide_dir, worker_id)) for worker_id in range(workers))


def ensure_slide_dirs(slide_dir: Path, workers: int) -> None:
    for subdir in ("merged", "preview", "logs", "_status"):
        (slide_dir / subdir).mkdir(parents=True, exist_ok=True)
    for worker_id in range(workers):
        worker_out_dir(slide_dir, worker_id).mkdir(parents=True, exist_ok=True)


def dry_run_records(config: dict[str, Any], only_slide: str | None = None) -> list[dict[str, Any]]:
    settings = settings_from_config(config)
    workers = settings["workers"]
    records: list[dict[str, Any]] = []
    for slide in iter_slides(config, only_slide=only_slide):
        tile_dir = slide["_tile_dir_path"]
        out_dir = slide_out_dir(settings, slide["name"])
        total_tiles = len(list_tiles(tile_dir, settings["image_suffixes"])) if tile_dir else 0
        completed_tiles = count_completed_for_slide(out_dir, workers) if out_dir.exists() else 0
        action = "run"
        reason = ""
        if not slide["enabled"]:
            action = "skip"
            reason = str(slide.get("skip_reason") or "disabled in config")
        elif tile_dir is None:
            action = "skip"
            reason = "tile_dir is empty"
        elif not tile_dir.exists():
            action = "missing"
            reason = "tile directory not found"
        elif total_tiles == 0:
            action = "skip"
            reason = "no image tiles found"
        elif completed_tiles >= total_tiles:
            action = "skip"
            reason = "already complete in output directory"
        elif completed_tiles > 0:
            action = "resume"
            reason = f"{completed_tiles}/{total_tiles} tiles already complete"
        records.append(
            {
                "slide_name": slide["name"],
                "action": action,
                "reason": reason,
                "enabled": slide["enabled"],
                "total_tiles": total_tiles,
                "completed_tiles": completed_tiles,
                "workers": workers,
                "tile_dir": str(tile_dir) if tile_dir else "",
                "output_dir": str(out_dir),
            }
        )
    return records


def print_dry_run(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No slides matched.")
        return
    df = pd.DataFrame(records)
    cols = [
        "slide_name",
        "action",
        "reason",
        "total_tiles",
        "completed_tiles",
        "workers",
        "tile_dir",
        "output_dir",
    ]
    print(df[cols].to_string(index=False))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(output_root: Path, force: bool = False) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "_cohort_RUN.lock"
    if lock_path.exists() and not force:
        try:
            payload = json.loads(lock_path.read_text())
            pid = int(payload.get("pid", 0))
        except Exception:
            pid = 0
        if pid and pid_alive(pid):
            raise RuntimeError(f"cohort run already active with pid {pid}; lock={lock_path}")
        lock_path.unlink(missing_ok=True)
    atomic_write_json(lock_path, {"pid": os.getpid(), "started_at": now_iso(), "script": str(Path(__file__).resolve())})
    return lock_path


def handle_stop_signal(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[cohort] received signal {signum}; stopping workers...", flush=True)


def build_worker_env(
    config: dict[str, Any],
    settings: dict[str, Any],
    slide: dict[str, Any],
    worker_id: int,
    tile_dir: Path,
    out_dir: Path,
) -> dict[str, str]:
    trial3 = settings["trial3"]
    env = os.environ.copy()
    for key, value in get_section(config, "environment").items():
        if value is None or str(value).strip() == "":
            continue
        if not env.get(key):
            env[key] = str(expand_path(value, REPO_ROOT) or value)
    for key, value in get_section(slide, "environment").items():
        if value is None or str(value).strip() == "":
            continue
        env[key] = str(expand_path(value, REPO_ROOT) or value)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(settings["gpu_ids"][worker_id % len(settings["gpu_ids"])]),
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRIAL3_TRACEBACK_AFTER_SECONDS": str(trial3.get("traceback_after_seconds", 600)),
            "TRIAL3_IMAGE_DIR": str(tile_dir),
            "TRIAL3_TILE_KEYS": str(trial3.get("tile_keys", "ALL")),
            "TRIAL3_NUM_SHARDS": str(settings["workers"]),
            "TRIAL3_SHARD_ID": str(worker_id),
            "TRIAL3_SKIP_EXISTING": str(trial3.get("skip_existing", 1)),
            "TRIAL3_OUT_DIR": str(out_dir),
            "TRIAL3_SAVE_COMPARISONS": str(trial3.get("save_comparisons", 1)),
            "TRIAL3_COMPARISON_EVERY": str(trial3.get("comparison_every", 10)),
            "TRIAL3_SAVE_DIAGNOSTICS": str(trial3.get("save_diagnostics", 0)),
            "TRIAL3_DIAGNOSTIC_EVERY": str(trial3.get("diagnostic_every", 1)),
            "TRIAL3_CSV_APPEND_EVERY": str(trial3.get("csv_append_every", 1)),
            "TRIAL3_VIS_DPI": str(trial3.get("vis_dpi", 120)),
            "TRIAL3_FAST_IMAGE_INDEX": str(trial3.get("fast_image_index", 1)),
            "TRIAL3_IMAGE_INDEX_READ_SIZES": str(trial3.get("image_index_read_sizes", 0)),
            "TRIAL3_FAST_IMAGE_DEFAULT_WIDTH": str(trial3.get("fast_image_default_width", 512)),
            "TRIAL3_FAST_IMAGE_DEFAULT_HEIGHT": str(trial3.get("fast_image_default_height", 512)),
            "TRIAL3_IMAGE_SUFFIXES": ",".join(sorted(settings["image_suffixes"])),
        }
    )
    return env


def validate_external_paths(env: dict[str, str]) -> None:
    cellseg_run_raw = env.get("CELLSEG1_RUN_DIR", "")
    cellseg_run_dir = Path(cellseg_run_raw).expanduser()
    checks = {
        "SAM3 repo": (env.get("SAM3_REPO", ""), Path(env.get("SAM3_REPO", "")).expanduser()),
        "SAM3 checkpoint": (
            env.get("SAM31_CHECKPOINT", ""),
            Path(env.get("SAM31_CHECKPOINT", "")).expanduser(),
        ),
        "YOLO model": (
            env.get("YOLO_MODEL_PATH", ""),
            Path(env.get("YOLO_MODEL_PATH", "")).expanduser(),
        ),
        "CellSeg1 repo": (
            env.get("CELLSEG1_REPO", ""),
            Path(env.get("CELLSEG1_REPO", "")).expanduser(),
        ),
        "CellSeg1 run dir": (cellseg_run_raw, cellseg_run_dir),
        "CellSeg1 config": (
            env.get("CELLSEG1_CONFIG_PATH") or str(cellseg_run_dir / "cellseg1_cgh_p2_runtime_config.yaml"),
            Path(env.get("CELLSEG1_CONFIG_PATH") or cellseg_run_dir / "cellseg1_cgh_p2_runtime_config.yaml").expanduser(),
        ),
        "CellSeg1 LoRA": (
            env.get("CELLSEG1_LORA") or str(cellseg_run_dir / "sam_lora_cgh_p2_cell_boundary.pth"),
            Path(env.get("CELLSEG1_LORA") or cellseg_run_dir / "sam_lora_cgh_p2_cell_boundary.pth").expanduser(),
        ),
    }
    missing = [f"{label}: {path}" for label, (raw, path) in checks.items() if not str(raw).strip() or not path.exists()]
    if missing:
        raise FileNotFoundError("preflight missing required paths:\n- " + "\n- ".join(missing))


def close_log_handle(state: WorkerRuntime) -> None:
    if state.log_handle is not None:
        state.log_handle.close()
        state.log_handle = None


def launch_worker(
    config: dict[str, Any],
    settings: dict[str, Any],
    slide: dict[str, Any],
    state: WorkerRuntime,
    tile_dir: Path,
    slide_dir: Path,
    restart: bool = False,
) -> None:
    if restart:
        state.restarts += 1
    out_dir = worker_out_dir(slide_dir, state.worker_id)
    log_path = worker_log_path(slide_dir, state.worker_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    close_log_handle(state)
    state.log_handle = log_path.open("a", buffering=1)
    state.launch_ts = time.time()
    state.last_progress_ts = state.launch_ts
    state.last_log_mtime = log_path.stat().st_mtime if log_path.exists() else state.launch_ts
    state.started_at_iso = state.started_at_iso or now_iso()
    state.last_completed = count_completed(out_dir)
    state.status = "running"
    state.error = ""
    state.process = subprocess.Popen(
        [sys.executable, str(settings["script"])],
        cwd=str(settings["repo"]),
        env=build_worker_env(config, settings, slide, state.worker_id, tile_dir, out_dir),
        stdout=state.log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    print(
        f"[cohort] launched {slide['name']} worker {state.worker_id:02d} "
        f"pid={state.process.pid} restarts={state.restarts} completed={state.last_completed}/{state.tiles_total}",
        flush=True,
    )


def stop_worker(state: WorkerRuntime, kill_timeout_seconds: int) -> None:
    process = state.process
    if process is None:
        close_log_handle(state)
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=kill_timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=kill_timeout_seconds)
    close_log_handle(state)


def restart_allowed(settings: dict[str, Any], state: WorkerRuntime) -> bool:
    return settings["max_restarts"] < 0 or state.restarts < settings["max_restarts"]


def ensure_timing_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=TIMING_COLUMNS).writeheader()


def read_timing_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    return rows if rows else []


def append_timing_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_timing_header(path)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_COLUMNS)
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in TIMING_COLUMNS})


def sync_timing_from_metrics(slide_dir: Path, slide_name: str, worker_id: int) -> None:
    timing_path = worker_timing_path(slide_dir, worker_id)
    existing = read_timing_rows(timing_path)
    seen_success = {row.get("tile_name", "") for row in existing if str(row.get("success", "")).lower() == "true"}
    metrics_rows = read_csv_rows(worker_out_dir(slide_dir, worker_id) / "trial_run_3_metrics.csv")
    new_rows: list[dict[str, Any]] = []
    for row in metrics_rows:
        model = row.get("model", "")
        if model and model != "Hybrid_vs_GT_all_boundaries":
            continue
        split = (row.get("split") or "").strip()
        tile_id = (row.get("tile_id") or "").strip()
        if not split or not tile_id:
            continue
        name = f"{split}_{tile_id}"
        if name in seen_success:
            continue
        seen_success.add(name)
        duration = row.get("inference_time_sec", "")
        instances = row.get("hybrid_instances") or row.get("pred_instances") or ""
        new_rows.append(
            {
                "timestamp": now_iso(),
                "slide_name": slide_name,
                "worker_id": worker_id,
                "tile_name": name,
                "duration_seconds": duration,
                "success": "true",
                "num_instances": instances,
            }
        )
    append_timing_rows(timing_path, new_rows)


def append_failed_timing(slide_dir: Path, slide_name: str, worker_id: int, tile_name: str, error: str) -> None:
    if not tile_name:
        return
    timing_path = worker_timing_path(slide_dir, worker_id)
    existing = read_timing_rows(timing_path)
    if any(row.get("tile_name") == tile_name and str(row.get("success", "")).lower() == "false" for row in existing):
        return
    append_timing_rows(
        timing_path,
        [
            {
                "timestamp": now_iso(),
                "slide_name": slide_name,
                "worker_id": worker_id,
                "tile_name": tile_name,
                "duration_seconds": "",
                "success": "false",
                "num_instances": 0,
            }
        ],
    )
    timing_path.with_suffix(".last_error.txt").write_text(error + "\n")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def compute_eta(timing_rows: list[dict[str, str]], tiles_done: int, tiles_total: int) -> dict[str, Any]:
    success_durations: list[float] = []
    for row in timing_rows:
        if str(row.get("success", "")).lower() != "true":
            continue
        try:
            duration = float(row.get("duration_seconds", ""))
        except ValueError:
            continue
        if duration > 0:
            success_durations.append(duration)
    remaining = max(tiles_total - tiles_done, 0)
    if not success_durations:
        return {
            "eta_seconds_median": None,
            "eta_human_median": "calculating",
            "eta_seconds_conservative": None,
            "eta_human_conservative": "calculating",
        }
    if len(success_durations) <= 5:
        return {
            "eta_seconds_median": None,
            "eta_human_median": "warming up",
            "eta_seconds_conservative": None,
            "eta_human_conservative": "warming up",
        }
    rolling = success_durations[5:][-50:]
    if not rolling:
        return {
            "eta_seconds_median": None,
            "eta_human_median": "calculating",
            "eta_seconds_conservative": None,
            "eta_human_conservative": "calculating",
        }
    median_eta = remaining * median(rolling)
    p75_eta = remaining * percentile(rolling, 0.75)
    return {
        "eta_seconds_median": round(median_eta, 3),
        "eta_human_median": format_seconds(median_eta),
        "eta_seconds_conservative": round(p75_eta, 3),
        "eta_human_conservative": format_seconds(p75_eta),
    }


def latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = [path for path in directory.glob(pattern) if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def tile_key_from_hybrid_mask(path: Path | None) -> str:
    if path is None:
        return ""
    if path.name.endswith(HYBRID_SUFFIX):
        return path.name[: -len(HYBRID_SUFFIX)]
    return path.stem


def read_log_tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def parse_current_tile_from_log(path: Path) -> str:
    tail = read_log_tail(path, lines=300)
    current = ""
    for line in tail.splitlines():
        match = re.search(r"Trial 3 running:\s*(\S+)", line)
        if match:
            current = match.group(1)
            continue
        match = re.search(r"Skipping existing tile:\s*(\S+)", line)
        if match:
            current = match.group(1)
    return current


def log_phase_and_latest(path: Path) -> tuple[str, str]:
    tail = read_log_tail(path, lines=160)
    phase = "waiting_for_log"
    latest = ""
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        latest = stripped
        lowered = stripped.lower()
        if "traceback" in lowered or "error" in lowered or "exception" in lowered:
            phase = "error"
        elif "trial run 3 complete" in lowered:
            phase = "complete"
        elif "cellseg1 generate start" in lowered:
            phase = "cellseg1_running"
        elif "cellseg1 generate done" in lowered:
            phase = "cellseg1_done"
        elif "sam3.1 clear" in lowered:
            phase = "sam3_running"
        elif "yolo nucleus" in lowered:
            phase = "yolo_running"
        elif "trial 3 running:" in lowered:
            phase = "tile_running"
        elif "loading cached cellseg1 predictor" in lowered or "cellseg1 lora" in lowered:
            phase = "loading_cellseg1"
        elif "loading sam3 processor" in lowered:
            phase = "loading_sam3"
        elif "sam3 import" in lowered:
            phase = "sam3_import"
        elif "sam3 build model" in lowered:
            phase = "sam3_build"
        elif "sam3 checkpoint" in lowered or "sam3 state dict" in lowered:
            phase = "sam3_checkpoint"
        elif "sam3 cuda move" in lowered:
            phase = "sam3_cuda"
        elif "sam3 processor ready" in lowered:
            phase = "sam3_ready"
        elif "loading yolo model" in lowered:
            phase = "loading_yolo"
        elif "loading input index" in lowered:
            phase = "indexing_inputs"
        elif "checking " in lowered:
            phase = "checking_paths"
        elif "main started" in lowered or "trial3 bootstrap" in lowered:
            phase = "started"
    return phase, latest


def write_preview(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            image.convert("RGB").save(destination)
    except Exception:
        shutil.copyfile(source, destination)


def copy_latest_preview(
    slide_dir: Path,
    worker_id: int,
    tile_map: dict[str, Path],
    current_tile: str,
    latest_comparison: Path | None,
) -> Path | None:
    source = latest_comparison
    if source is None and current_tile:
        source = tile_map.get(current_tile)
    if source is None and tile_map:
        source = next(iter(tile_map.values()))
    if source is None or not source.exists():
        return None
    destination = slide_dir / "preview" / f"latest_worker_{worker_id:02d}.png"
    write_preview(source, destination)
    return destination


def worker_status_payload(
    slide_name: str,
    slide_dir: Path,
    worker_id: int,
    tiles_total: int,
    tile_map: dict[str, Path],
    state: WorkerRuntime | None = None,
    status_override: str | None = None,
    error: str = "",
) -> dict[str, Any]:
    out_dir = worker_out_dir(slide_dir, worker_id)
    log_path = worker_log_path(slide_dir, worker_id)
    sync_timing_from_metrics(slide_dir, slide_name, worker_id)
    tiles_done = count_completed(out_dir)
    timing_rows = read_timing_rows(worker_timing_path(slide_dir, worker_id))
    latest_comparison = latest_file(out_dir / "comparison_images", "*_trial3_compare.png")
    latest_mask = latest_file(out_dir / "pred_masks", f"*{HYBRID_SUFFIX}")
    phase, latest_log_line = log_phase_and_latest(log_path)
    current_tile = parse_current_tile_from_log(log_path) or tile_key_from_hybrid_mask(latest_mask)
    if not current_tile:
        completed_keys = tile_keys_with_complete_outputs(out_dir)
        current_tile = next((key for key in tile_map if key not in completed_keys), "")
    preview_path = copy_latest_preview(slide_dir, worker_id, tile_map, current_tile, latest_comparison)
    started_at = state.started_at_iso if state and state.started_at_iso else ""
    launch_ts = state.launch_ts if state else 0.0
    elapsed = max(0.0, time.time() - launch_ts) if launch_ts else 0.0
    tiles_per_minute = (tiles_done / elapsed * 60.0) if elapsed > 0 else 0.0
    status = status_override or (state.status if state else "unknown")
    payload = {
        "worker_id": worker_id,
        "slide_name": slide_name,
        "status": status,
        "phase": phase,
        "current_tile": current_tile,
        "tiles_done": tiles_done,
        "tiles_total": tiles_total,
        "percent_complete": round((tiles_done / tiles_total * 100.0) if tiles_total else 0.0, 3),
        "latest_input_tile_path": str(tile_map.get(current_tile, "")) if current_tile else "",
        "latest_comparison_image_path": str(latest_comparison) if latest_comparison else "",
        "latest_mask_path": str(latest_mask) if latest_mask else "",
        "latest_preview_image_path": str(preview_path) if preview_path else "",
        "started_at": started_at,
        "updated_at": now_iso(),
        "elapsed_seconds": round(elapsed, 3),
        **compute_eta(timing_rows, tiles_done, tiles_total),
        "tiles_per_minute": round(tiles_per_minute, 4),
        "latest_log_line": latest_log_line,
        "error": error or (state.error if state else ""),
    }
    atomic_write_json(worker_status_path(slide_dir, worker_id), payload)
    return payload


def slide_status_payload(
    slide_name: str,
    slide_dir: Path,
    tile_dir: Path,
    workers: int,
    status: str,
    started_at: str,
    error: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker_payloads = []
    for worker_id in range(workers):
        path = worker_status_path(slide_dir, worker_id)
        if path.exists():
            try:
                worker_payloads.append(json.loads(path.read_text()))
            except Exception:
                pass
    tiles_total = sum(int(row.get("tiles_total", 0)) for row in worker_payloads)
    tiles_done = sum(int(row.get("tiles_done", 0)) for row in worker_payloads)
    failed_workers = [
        row.get("worker_id")
        for row in worker_payloads
        if str(row.get("status", "")).lower() in {"failed", "error"}
    ]
    payload = {
        "slide_name": slide_name,
        "status": status,
        "tile_dir": str(tile_dir),
        "output_dir": str(slide_dir),
        "workers": workers,
        "started_at": started_at,
        "updated_at": now_iso(),
        "completed_at": now_iso() if status == "completed" else "",
        "tiles_done": tiles_done,
        "tiles_total": tiles_total,
        "percent_complete": round((tiles_done / tiles_total * 100.0) if tiles_total else 0.0, 3),
        "failed_workers": failed_workers,
        "error": error,
    }
    if extra:
        payload.update(extra)
    return payload


def write_slide_marker(slide_dir: Path, marker: str, payload: dict[str, Any]) -> None:
    marker_paths = {
        "running": slide_dir / "_RUNNING.json",
        "completed": slide_dir / "_COMPLETED.json",
        "failed": slide_dir / "_FAILED.json",
    }
    atomic_write_json(marker_paths[marker], payload)
    for key, path in marker_paths.items():
        if key != marker:
            path.unlink(missing_ok=True)


def safe_read_df(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if list(df.columns) == ["empty"]:
        return pd.DataFrame()
    return df


def merge_worker_csvs(slide_dir: Path, workers: int, slide_name: str) -> dict[str, str]:
    merged_dir = slide_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    merged_features = pd.DataFrame()
    for src_name, out_name in MERGE_SPECS.items():
        frames = []
        for worker_id in range(workers):
            df = safe_read_df(worker_out_dir(slide_dir, worker_id) / src_name)
            if df.empty:
                continue
            df.insert(0, "worker_id", worker_id)
            df.insert(0, "slide_name", slide_name)
            frames.append(df)
        out_path = merged_dir / out_name
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged.to_csv(out_path, index=False)
            outputs[src_name] = str(out_path)
            if src_name == "trial_run_3_morphology_features.csv":
                merged_features = merged
        else:
            pd.DataFrame().to_csv(out_path, index=False)
            outputs[src_name] = str(out_path)

    summary_path = merged_dir / "merged_trial_run_3_morphology_summary.csv"
    if not merged_features.empty and {"split", "final_class_name", "source_model", "final_label"}.issubset(
        merged_features.columns
    ):
        agg_specs = {"n_cells": ("final_label", "count")}
        optional_aggs = {
            "cell_area_px_mean": ("cell_area_px", "mean"),
            "cell_area_px_std": ("cell_area_px", "std"),
            "nucleus_area_px_mean": ("nucleus_area_px", "mean"),
            "cytoplasm_area_px_mean": ("cytoplasm_area_px", "mean"),
            "nc_ratio_mean": ("nc_ratio", "mean"),
            "nc_ratio_std": ("nc_ratio", "std"),
            "nucleus_count_mean": ("nucleus_count", "mean"),
            "cell_equiv_diameter_px_mean": ("cell_equiv_diameter_px", "mean"),
            "nucleus_equiv_diameter_px_mean": ("nucleus_equiv_diameter_px", "mean"),
        }
        agg_specs.update({name: spec for name, spec in optional_aggs.items() if spec[0] in merged_features.columns})
        summary = (
            merged_features.groupby(["slide_name", "split", "final_class_name", "source_model"])
            .agg(**agg_specs)
            .reset_index()
        )
        summary.to_csv(summary_path, index=False)
    else:
        frames = []
        for worker_id in range(workers):
            df = safe_read_df(worker_out_dir(slide_dir, worker_id) / "trial_run_3_morphology_summary.csv")
            if not df.empty:
                df.insert(0, "worker_id", worker_id)
                df.insert(0, "slide_name", slide_name)
                frames.append(df)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(summary_path, index=False)
        else:
            pd.DataFrame().to_csv(summary_path, index=False)
    outputs["trial_run_3_morphology_summary.csv"] = str(summary_path)
    return outputs


def validate_run_paths(settings: dict[str, Any]) -> None:
    for label, path in (("repo", settings["repo"]), ("inference script", settings["script"])):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def run_slide(config: dict[str, Any], settings: dict[str, Any], slide: dict[str, Any]) -> int:
    validate_run_paths(settings)
    slide_name = slide["name"]
    tile_dir = slide["_tile_dir_path"]
    if tile_dir is None or not tile_dir.exists():
        raise FileNotFoundError(f"{slide_name}: tile directory not found: {tile_dir}")
    tiles = list_tiles(tile_dir, settings["image_suffixes"])
    if not tiles:
        raise FileNotFoundError(f"{slide_name}: no image tiles found in {tile_dir}")

    workers = settings["workers"]
    slide_dir = slide_out_dir(settings, slide_name)
    ensure_slide_dirs(slide_dir, workers)
    if settings["preflight_paths"]:
        preflight_env = build_worker_env(
            config,
            settings,
            slide,
            0,
            tile_dir,
            worker_out_dir(slide_dir, 0),
        )
        validate_external_paths(preflight_env)
    started_at = now_iso()
    states: dict[int, WorkerRuntime] = {}
    maps_by_worker: dict[int, dict[str, Path]] = {}
    for worker_id in range(workers):
        assigned = assigned_tiles(tiles, worker_id, workers)
        maps_by_worker[worker_id] = {tile_key(path): path for path in assigned}
        states[worker_id] = WorkerRuntime(worker_id=worker_id, tiles_total=len(assigned))
        ensure_timing_header(worker_timing_path(slide_dir, worker_id))

    print(f"[cohort] slide {slide_name}: {len(tiles)} tiles, {workers} workers, out={slide_dir}", flush=True)
    write_slide_marker(
        slide_dir,
        "running",
        slide_status_payload(slide_name, slide_dir, tile_dir, workers, "running", started_at),
    )

    already_complete = True
    for worker_id, state in states.items():
        completed = count_completed(worker_out_dir(slide_dir, worker_id))
        state.last_completed = completed
        state.last_progress_ts = time.time()
        if completed >= state.tiles_total:
            state.status = "completed"
            worker_status_payload(
                slide_name,
                slide_dir,
                worker_id,
                state.tiles_total,
                maps_by_worker[worker_id],
                state=state,
                status_override="completed",
            )
            continue
        already_complete = False
        launch_worker(config, settings, slide, state, tile_dir, slide_dir)
        worker_status_payload(slide_name, slide_dir, worker_id, state.tiles_total, maps_by_worker[worker_id], state=state)

    if already_complete:
        merged_outputs = merge_worker_csvs(slide_dir, workers, slide_name)
        payload = slide_status_payload(
            slide_name,
            slide_dir,
            tile_dir,
            workers,
            "completed",
            started_at,
            extra={"merged_outputs": merged_outputs, "reason": "already complete"},
        )
        write_slide_marker(slide_dir, "completed", payload)
        print(f"[cohort] slide {slide_name} already complete; merged CSVs refreshed", flush=True)
        return 0

    while not STOP_REQUESTED:
        all_done = True
        now = time.time()
        for worker_id, state in states.items():
            out_dir = worker_out_dir(slide_dir, worker_id)
            log_path = worker_log_path(slide_dir, worker_id)
            completed = count_completed(out_dir)
            try:
                log_mtime = log_path.stat().st_mtime
            except FileNotFoundError:
                log_mtime = 0.0
            if log_mtime > state.last_log_mtime:
                state.last_log_mtime = log_mtime
                state.last_progress_ts = now
            if completed > state.last_completed:
                print(
                    f"[cohort] {slide_name} worker {worker_id:02d} progress "
                    f"{state.last_completed}->{completed}/{state.tiles_total}",
                    flush=True,
                )
                state.last_completed = completed
                state.last_progress_ts = now

            process = state.process
            return_code = None if process is None else process.poll()
            worker_done = completed >= state.tiles_total and (process is None or return_code is not None)
            if worker_done:
                state.status = "completed"
                close_log_handle(state)
                worker_status_payload(
                    slide_name,
                    slide_dir,
                    worker_id,
                    state.tiles_total,
                    maps_by_worker[worker_id],
                    state=state,
                    status_override="completed",
                )
                continue

            all_done = False
            if completed >= state.tiles_total:
                state.status = "finishing"
                worker_status_payload(slide_name, slide_dir, worker_id, state.tiles_total, maps_by_worker[worker_id], state=state)
                continue

            if process is None:
                if restart_allowed(settings, state):
                    state.status = "restarting"
                    state.error = f"worker process missing before completion at {completed}/{state.tiles_total}; restarting"
                    print(f"[cohort] {slide_name} worker {worker_id:02d} {state.error}", flush=True)
                    worker_status_payload(
                        slide_name,
                        slide_dir,
                        worker_id,
                        state.tiles_total,
                        maps_by_worker[worker_id],
                        state=state,
                    )
                    launch_worker(config, settings, slide, state, tile_dir, slide_dir, restart=True)
                else:
                    state.status = "failed"
                    state.error = "restart limit reached"
                    return fail_slide(
                        slide_name,
                        slide_dir,
                        tile_dir,
                        workers,
                        states,
                        maps_by_worker,
                        started_at,
                        state.error,
                        kill_timeout_seconds=settings["kill_timeout_seconds"],
                    )
                worker_status_payload(slide_name, slide_dir, worker_id, state.tiles_total, maps_by_worker[worker_id], state=state)
                continue

            if return_code is not None:
                state.error = (
                    f"worker stopped before completion rc={return_code} at "
                    f"{completed}/{state.tiles_total}; restarting with TRIAL3_SKIP_EXISTING=1"
                )
                current_tile = parse_current_tile_from_log(log_path)
                if not state.failed_tile_recorded:
                    append_failed_timing(slide_dir, slide_name, worker_id, current_tile, state.error)
                    state.failed_tile_recorded = True
                print(f"[cohort] {slide_name} worker {worker_id:02d} {state.error}", flush=True)
                if not restart_allowed(settings, state):
                    state.status = "failed"
                    return fail_slide(
                        slide_name,
                        slide_dir,
                        tile_dir,
                        workers,
                        states,
                        maps_by_worker,
                        started_at,
                        state.error,
                        kill_timeout_seconds=settings["kill_timeout_seconds"],
                    )
                state.status = "restarting"
                worker_status_payload(slide_name, slide_dir, worker_id, state.tiles_total, maps_by_worker[worker_id], state=state)
                time.sleep(settings["restart_delay_seconds"])
                state.failed_tile_recorded = False
                launch_worker(config, settings, slide, state, tile_dir, slide_dir, restart=True)
                continue

            phase, latest_log_line = log_phase_and_latest(log_path)
            startup_phases = {
                "waiting_for_log",
                "started",
                "checking_paths",
                "indexing_inputs",
                "loading_yolo",
                "loading_sam3",
                "sam3_import",
                "sam3_build",
                "sam3_checkpoint",
                "sam3_cuda",
                "sam3_ready",
                "loading_cellseg1",
            }
            effective_stale_seconds = (
                settings["startup_stale_seconds"]
                if completed == 0 and phase in startup_phases
                else settings["stale_seconds"]
            )
            stale_for = now - state.last_progress_ts
            if stale_for >= effective_stale_seconds:
                state.error = (
                    f"stale for {int(stale_for)}s at {completed}/{state.tiles_total} "
                    f"phase={phase} limit={effective_stale_seconds}s latest_log={latest_log_line[:160]}"
                )
                print(f"[cohort] {slide_name} worker {worker_id:02d} {state.error}; restarting", flush=True)
                stop_worker(state, settings["kill_timeout_seconds"])
                if not restart_allowed(settings, state):
                    state.status = "failed"
                    return fail_slide(
                        slide_name,
                        slide_dir,
                        tile_dir,
                        workers,
                        states,
                        maps_by_worker,
                        started_at,
                        state.error,
                        kill_timeout_seconds=settings["kill_timeout_seconds"],
                    )
                state.status = "restarting"
                worker_status_payload(slide_name, slide_dir, worker_id, state.tiles_total, maps_by_worker[worker_id], state=state)
                time.sleep(settings["restart_delay_seconds"])
                launch_worker(config, settings, slide, state, tile_dir, slide_dir, restart=True)
                continue

            state.status = "running"
            worker_status_payload(slide_name, slide_dir, worker_id, state.tiles_total, maps_by_worker[worker_id], state=state)

        write_slide_marker(
            slide_dir,
            "running",
            slide_status_payload(slide_name, slide_dir, tile_dir, workers, "running", started_at),
        )

        if all_done:
            merged_outputs = merge_worker_csvs(slide_dir, workers, slide_name)
            payload = slide_status_payload(
                slide_name,
                slide_dir,
                tile_dir,
                workers,
                "completed",
                started_at,
                extra={"merged_outputs": merged_outputs},
            )
            write_slide_marker(slide_dir, "completed", payload)
            print(f"[cohort] slide {slide_name} complete; merged CSVs written to {slide_dir / 'merged'}", flush=True)
            return 0

        time.sleep(settings["poll_seconds"])

    return fail_slide(
        slide_name,
        slide_dir,
        tile_dir,
        workers,
        states,
        maps_by_worker,
        started_at,
        "stopped by signal",
        stop_workers=True,
        kill_timeout_seconds=settings["kill_timeout_seconds"],
    )


def fail_slide(
    slide_name: str,
    slide_dir: Path,
    tile_dir: Path,
    workers: int,
    states: dict[int, WorkerRuntime],
    maps_by_worker: dict[int, dict[str, Path]],
    started_at: str,
    error: str,
    stop_workers: bool = True,
    kill_timeout_seconds: int = 30,
) -> int:
    if stop_workers:
        for state in states.values():
            stop_worker(state, kill_timeout_seconds)
    for worker_id, state in states.items():
        if state.status not in {"completed", "failed"}:
            state.status = "stopped"
        worker_status_payload(
            slide_name,
            slide_dir,
            worker_id,
            state.tiles_total,
            maps_by_worker[worker_id],
            state=state,
            status_override=state.status,
            error=state.error or error,
        )
    payload = slide_status_payload(slide_name, slide_dir, tile_dir, workers, "failed", started_at, error=error)
    write_slide_marker(slide_dir, "failed", payload)
    print(f"[cohort] slide {slide_name} failed: {error}", flush=True)
    return 2


def run_cohort(config: dict[str, Any], only_slide: str | None = None, force_lock: bool = False) -> int:
    settings = settings_from_config(config)
    settings["output_root"].mkdir(parents=True, exist_ok=True)
    lock_path = acquire_lock(settings["output_root"], force=force_lock)
    try:
        exit_code = 0
        for slide in iter_slides(config, only_slide=only_slide):
            if not slide["enabled"]:
                print(f"[cohort] skipping {slide['name']}: {slide.get('skip_reason') or 'disabled in config'}", flush=True)
                continue
            try:
                result = run_slide(config, settings, slide)
            except Exception as exc:
                result = 2
                slide_dir = slide_out_dir(settings, slide["name"])
                slide_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "slide_name": slide["name"],
                    "status": "failed",
                    "tile_dir": str(slide.get("_tile_dir_path") or ""),
                    "output_dir": str(slide_dir),
                    "workers": settings["workers"],
                    "started_at": "",
                    "updated_at": now_iso(),
                    "completed_at": "",
                    "tiles_done": 0,
                    "tiles_total": 0,
                    "percent_complete": 0.0,
                    "failed_workers": [],
                    "error": str(exc),
                }
                write_slide_marker(slide_dir, "failed", payload)
                print(f"[cohort] {slide['name']} failed before launch: {exc}", flush=True)
            if result != 0:
                exit_code = result
                if not settings["continue_on_error"]:
                    return exit_code
        return exit_code
    finally:
        lock_path.unlink(missing_ok=True)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def slide_marker_status(slide_dir: Path) -> dict[str, Any] | None:
    for marker in ("_RUNNING.json", "_FAILED.json", "_COMPLETED.json"):
        payload = read_json_if_exists(slide_dir / marker)
        if payload:
            return payload
    return None


def monitor_snapshot(config: dict[str, Any], only_slide: str | None = None) -> dict[str, Any]:
    settings = settings_from_config(config)
    slides = []
    workers = []
    for slide in iter_slides(config, only_slide=only_slide):
        slide_dir = slide_out_dir(settings, slide["name"])
        slide_status = slide_marker_status(slide_dir) or {
            "slide_name": slide["name"],
            "status": "not_started" if slide["enabled"] else "skipped",
            "tile_dir": str(slide.get("_tile_dir_path") or ""),
            "output_dir": str(slide_dir),
            "workers": settings["workers"],
            "updated_at": "",
            "tiles_done": 0,
            "tiles_total": 0,
            "percent_complete": 0.0,
            "error": slide.get("skip_reason", "") if not slide["enabled"] else "",
        }
        slides.append(slide_status)
        for worker_id in range(settings["workers"]):
            payload = read_json_if_exists(worker_status_path(slide_dir, worker_id))
            if payload:
                workers.append(payload)
    total_done = sum(int(row.get("tiles_done", 0)) for row in slides)
    total_tiles = sum(int(row.get("tiles_total", 0)) for row in slides)
    running = sum(1 for row in slides if row.get("status") == "running")
    completed = sum(1 for row in slides if row.get("status") == "completed")
    failed = sum(1 for row in slides if row.get("status") == "failed")
    lock = settings["output_root"] / "_cohort_RUN.lock"
    lock_payload = read_json_if_exists(lock)
    return {
        "cohort": {
            "output_root": str(settings["output_root"]),
            "config_path": config.get("_config_path", ""),
            "updated_at": now_iso(),
            "slides_total": len(slides),
            "slides_running": running,
            "slides_completed": completed,
            "slides_failed": failed,
            "tiles_done": total_done,
            "tiles_total": total_tiles,
            "percent_complete": round((total_done / total_tiles * 100.0) if total_tiles else 0.0, 3),
            "lock": lock_payload or {},
        },
        "slides": slides,
        "workers": workers,
    }


def print_monitor_snapshot(snapshot: dict[str, Any], log_tail_lines: int = 20) -> None:
    print(json.dumps(snapshot["cohort"], indent=2))
    slides = pd.DataFrame(snapshot["slides"])
    if not slides.empty:
        cols = [c for c in ["slide_name", "status", "tiles_done", "tiles_total", "percent_complete", "error"] if c in slides]
        print("\nSlides")
        print(slides[cols].to_string(index=False))
    workers = pd.DataFrame(snapshot["workers"])
    if not workers.empty:
        cols = [
            c
            for c in [
                "slide_name",
                "worker_id",
                "status",
                "tiles_done",
                "tiles_total",
                "percent_complete",
                "eta_human_median",
                "eta_human_conservative",
                "tiles_per_minute",
                "current_tile",
                "error",
            ]
            if c in workers
        ]
        print("\nWorkers")
        print(workers[cols].to_string(index=False))
    output_root = Path(snapshot["cohort"]["output_root"])
    launcher_log = output_root / "cohort_launcher.log"
    tail = read_log_tail(launcher_log, log_tail_lines)
    if tail:
        print("\nRecent launcher log")
        print(tail)


def monitor_loop(config: dict[str, Any], only_slide: str | None = None, once: bool = False) -> int:
    settings = settings_from_config(config)
    while True:
        snapshot = monitor_snapshot(config, only_slide=only_slide)
        print("\033[2J\033[H", end="")
        print_monitor_snapshot(snapshot, settings["log_tail_lines"])
        if once:
            return 0
        time.sleep(settings["refresh_seconds"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Trial Run 3 inference on configured cohort slides.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    parser.add_argument("--dry-run", action="store_true", help="print slides that would run or skip")
    parser.add_argument("--run", action="store_true", help="run enabled slides")
    parser.add_argument("--monitor", action="store_true", help="print live status from JSON files")
    parser.add_argument("--once", action="store_true", help="with --monitor, print one snapshot and exit")
    parser.add_argument("--slide", help="run or monitor only one slide name")
    parser.add_argument("--workers", type=int, help="override cohort.workers")
    parser.add_argument("--gpu-ids", help="override cohort.gpu_ids, comma-separated")
    parser.add_argument("--output-root", help="override cohort.output_root")
    parser.add_argument("--poll-seconds", type=int, help="override cohort.poll_seconds")
    parser.add_argument("--stale-seconds", type=int, help="override cohort.stale_seconds")
    parser.add_argument("--no-preflight", action="store_true", help="skip external model/source path checks before launch")
    parser.add_argument("--force-lock", action="store_true", help="ignore an existing stale run lock")
    return parser


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args)
    if args.dry_run or not (args.run or args.monitor):
        print_dry_run(dry_run_records(config, only_slide=args.slide))
    if args.run:
        return run_cohort(config, only_slide=args.slide, force_lock=args.force_lock)
    if args.monitor:
        return monitor_loop(config, only_slide=args.slide, once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
