#!/usr/bin/env python3
"""Supervise sharded Trial Run 3 inference workers.

This launcher restarts workers after native/CUDA crashes or long stalls. It
relies on trial_run_3_hybrid_inference.py resume mode, so restarted workers
skip tiles that have complete mask outputs and complete metrics CSV rows.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
HYBRID_SUFFIX = "_hybrid_instance_mask.png"
OUTPUT_SUFFIXES = (
    "_hybrid_instance_mask.png",
    "_hybrid_class_mask.png",
    "_yolo_nucleus_mask.png",
    "_sam31_clear_mask.png",
    "_cellseg1_residual_compact_mask.png",
)

STOP_REQUESTED = False


@dataclass
class WorkerState:
    worker_id: int
    process: subprocess.Popen | None = None
    log_handle: TextIO | None = None
    restarts: int = 0
    last_completed: int = 0
    last_progress_ts: float = 0.0
    launch_ts: float = 0.0


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def parse_gpu_ids(raw: str) -> list[int]:
    gpu_ids = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not gpu_ids:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    return gpu_ids


def parse_args() -> argparse.Namespace:
    home = Path.home()
    repo = Path(__file__).resolve().parent
    default_output_root = (
        home
        / "Desktop"
        / "sam31-cgh-strategy2"
        / "outputs"
        / "strategy2_41tiles_full_unfreeze_20260625_160623"
    )
    parser = argparse.ArgumentParser(description="Restart Trial Run 3 workers after crashes or stalls.")
    parser.add_argument("--repo", type=expand_path, default=repo)
    parser.add_argument("--script", type=expand_path, default=repo / "trial_run_3_hybrid_inference.py")
    parser.add_argument(
        "--tile-dir",
        type=expand_path,
        default=home / "Desktop" / "prototype5_whole_image_runs" / "target_region_6262_13752um" / "tiles",
    )
    parser.add_argument(
        "--base-out",
        type=expand_path,
        default=home / "Desktop" / "prototype5_whole_image_runs" / "trial_run_3_yolo_sam_cellseg",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default=parse_gpu_ids("0"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--restart-delay-seconds", type=int, default=20)
    parser.add_argument("--kill-timeout-seconds", type=int, default=30)
    parser.add_argument("--max-restarts", type=int, default=-1, help="-1 means unlimited")
    parser.add_argument("--traceback-seconds", type=int, default=600)
    parser.add_argument("--csv-append-every", type=int, default=1)
    parser.add_argument("--save-comparisons", choices=("0", "1"), default="1")
    parser.add_argument("--comparison-every", type=int, default=10)
    parser.add_argument("--save-diagnostics", choices=("0", "1"), default="0")
    parser.add_argument("--vis-dpi", type=int, default=120)
    parser.add_argument(
        "--sam3-repo",
        type=expand_path,
        default=home / "Desktop" / "sam31-cgh-training-data" / "sam3",
    )
    parser.add_argument("--sam31-output-root", type=expand_path, default=default_output_root)
    parser.add_argument("--sam31-checkpoint", type=expand_path, default=default_output_root / "checkpoints" / "checkpoint.pt")
    parser.add_argument(
        "--yolo-model-path",
        type=expand_path,
        default=home
        / "Desktop"
        / "sam31-cgh-training-data"
        / "training_data"
        / "reference_models"
        / "cellseg1_cgh_p2_yolo_best.pt",
    )
    parser.add_argument(
        "--cellseg1-repo",
        type=expand_path,
        default=home
        / "Desktop"
        / "1.Data"
        / "training_pa_he_annotation_full"
        / "outputs"
        / "cellseg1_cluster_live"
        / "cellseg1_repo",
    )
    parser.add_argument(
        "--cellseg1-run-dir",
        type=expand_path,
        default=home
        / "Desktop"
        / "1.Data"
        / "training_pa_he_annotation_full"
        / "outputs"
        / "cellseg1_cluster_live"
        / "cellseg1_cgh_p2_41full_20260625_124306",
    )
    return parser.parse_args()


def handle_stop_signal(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[supervisor] received signal {signum}; stopping workers...", flush=True)


def list_tiles(tile_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in tile_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith("._")
    )


def worker_out_dir(base_out: Path, worker_id: int) -> Path:
    return base_out / f"worker_{worker_id:02d}"


def worker_log_path(base_out: Path, worker_id: int) -> Path:
    return base_out / f"worker_{worker_id:02d}.log"


def assigned_count(tiles: list[Path], worker_id: int, workers: int) -> int:
    return sum(1 for index, _path in enumerate(tiles) if index % workers == worker_id)


def tile_keys_with_complete_outputs(out_dir: Path) -> set[str]:
    pred_dir = out_dir / "pred_masks"
    if not pred_dir.exists():
        return set()

    complete = set()
    for hybrid_path in pred_dir.glob(f"*{HYBRID_SUFFIX}"):
        tile_key = hybrid_path.name[: -len(HYBRID_SUFFIX)]
        if all((pred_dir / f"{tile_key}{suffix}").exists() for suffix in OUTPUT_SUFFIXES):
            complete.add(tile_key)
    return complete


def tile_keys_with_complete_metrics(out_dir: Path) -> set[str]:
    metrics_path = out_dir / "trial_run_3_metrics.csv"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return set()

    counts: dict[str, int] = {}
    try:
        with metrics_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == ["empty"]:
                return set()
            for row in reader:
                split = (row.get("split") or "").strip()
                tile_id = (row.get("tile_id") or "").strip()
                if not split or not tile_id:
                    continue
                tile_key = f"{split}_{tile_id}"
                counts[tile_key] = counts.get(tile_key, 0) + 1
    except Exception as exc:
        print(f"[supervisor] warning: could not read metrics CSV {metrics_path}: {exc}", flush=True)
        return set()

    return {tile_key for tile_key, count in counts.items() if count >= 3}


def count_completed(out_dir: Path) -> int:
    return len(tile_keys_with_complete_outputs(out_dir) & tile_keys_with_complete_metrics(out_dir))


def close_log_handle(state: WorkerState) -> None:
    if state.log_handle is not None:
        state.log_handle.close()
        state.log_handle = None


def build_worker_env(args: argparse.Namespace, worker_id: int) -> dict[str, str]:
    out_dir = worker_out_dir(args.base_out, worker_id)
    gpu_id = args.gpu_ids[worker_id % len(args.gpu_ids)]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "TRIAL3_TRACEBACK_AFTER_SECONDS": str(args.traceback_seconds),
            "TRIAL3_IMAGE_DIR": str(args.tile_dir),
            "TRIAL3_TILE_KEYS": "ALL",
            "TRIAL3_SHARD_ID": str(worker_id),
            "TRIAL3_NUM_SHARDS": str(args.workers),
            "TRIAL3_SKIP_EXISTING": "1",
            "TRIAL3_SAVE_COMPARISONS": args.save_comparisons,
            "TRIAL3_COMPARISON_EVERY": str(args.comparison_every),
            "TRIAL3_SAVE_DIAGNOSTICS": args.save_diagnostics,
            "TRIAL3_VIS_DPI": str(args.vis_dpi),
            "TRIAL3_CSV_APPEND_EVERY": str(args.csv_append_every),
            "TRIAL3_OUT_DIR": str(out_dir),
            "SAM3_REPO": str(args.sam3_repo),
            "SAM31_OUTPUT_ROOT": str(args.sam31_output_root),
            "SAM31_CHECKPOINT": str(args.sam31_checkpoint),
            "YOLO_MODEL_PATH": str(args.yolo_model_path),
            "CELLSEG1_REPO": str(args.cellseg1_repo),
            "CELLSEG1_RUN_DIR": str(args.cellseg1_run_dir),
        }
    )
    return env


def launch_worker(args: argparse.Namespace, state: WorkerState, restart: bool = False) -> None:
    if restart:
        state.restarts += 1
    out_dir = worker_out_dir(args.base_out, state.worker_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = worker_log_path(args.base_out, state.worker_id)
    close_log_handle(state)
    state.log_handle = log_path.open("a", buffering=1)
    state.launch_ts = time.time()
    state.last_progress_ts = state.launch_ts
    state.last_completed = count_completed(out_dir)
    state.process = subprocess.Popen(
        [sys.executable, str(args.script)],
        cwd=str(args.repo),
        env=build_worker_env(args, state.worker_id),
        stdout=state.log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    print(
        f"[supervisor] launched worker {state.worker_id:02d} "
        f"pid={state.process.pid} restarts={state.restarts} "
        f"completed={state.last_completed} out={out_dir} log={log_path}",
        flush=True,
    )


def stop_worker(state: WorkerState, kill_timeout_seconds: int) -> None:
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


def restart_allowed(args: argparse.Namespace, state: WorkerState) -> bool:
    return args.max_restarts < 0 or state.restarts < args.max_restarts


def supervise(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    for required_path in (
        args.repo,
        args.script,
        args.tile_dir,
        args.sam3_repo,
        args.sam31_checkpoint,
        args.yolo_model_path,
        args.cellseg1_repo,
        args.cellseg1_run_dir,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    args.base_out.mkdir(parents=True, exist_ok=True)
    tiles = list_tiles(args.tile_dir)
    if not tiles:
        raise FileNotFoundError(f"no image tiles found in {args.tile_dir}")

    assigned = {worker_id: assigned_count(tiles, worker_id, args.workers) for worker_id in range(args.workers)}
    print("[supervisor] repo:", args.repo, flush=True)
    print("[supervisor] tile_dir:", args.tile_dir, flush=True)
    print("[supervisor] base_out:", args.base_out, flush=True)
    print("[supervisor] workers:", args.workers, "gpu_ids:", args.gpu_ids, flush=True)
    print("[supervisor] assigned:", assigned, flush=True)

    states = {worker_id: WorkerState(worker_id=worker_id) for worker_id in range(args.workers)}
    for worker_id, state in states.items():
        completed = count_completed(worker_out_dir(args.base_out, worker_id))
        state.last_completed = completed
        state.last_progress_ts = time.time()
        if completed >= assigned[worker_id]:
            print(f"[supervisor] worker {worker_id:02d} already complete: {completed}/{assigned[worker_id]}", flush=True)
            continue
        launch_worker(args, state)

    while not STOP_REQUESTED:
        all_done = True
        now = time.time()

        for worker_id, state in states.items():
            out_dir = worker_out_dir(args.base_out, worker_id)
            completed = count_completed(out_dir)
            if completed > state.last_completed:
                print(
                    f"[supervisor] worker {worker_id:02d} progress "
                    f"{state.last_completed}->{completed}/{assigned[worker_id]}",
                    flush=True,
                )
                state.last_completed = completed
                state.last_progress_ts = now

            process = state.process
            return_code = None if process is None else process.poll()
            worker_done = completed >= assigned[worker_id] and (process is None or return_code is not None)
            if worker_done:
                close_log_handle(state)
                continue

            all_done = False
            if completed >= assigned[worker_id]:
                continue

            if process is None:
                if restart_allowed(args, state):
                    launch_worker(args, state, restart=True)
                else:
                    print(f"[supervisor] worker {worker_id:02d} restart limit reached", flush=True)
                    return 2
                continue

            if return_code is not None:
                print(
                    f"[supervisor] worker {worker_id:02d} exited rc={return_code} "
                    f"completed={completed}/{assigned[worker_id]}",
                    flush=True,
                )
                if not restart_allowed(args, state):
                    print(f"[supervisor] worker {worker_id:02d} restart limit reached", flush=True)
                    return 2
                time.sleep(args.restart_delay_seconds)
                launch_worker(args, state, restart=True)
                continue

            stale_for = now - state.last_progress_ts
            if stale_for >= args.stale_seconds:
                print(
                    f"[supervisor] worker {worker_id:02d} stale for {int(stale_for)}s "
                    f"at {completed}/{assigned[worker_id]}; restarting",
                    flush=True,
                )
                stop_worker(state, args.kill_timeout_seconds)
                if not restart_allowed(args, state):
                    print(f"[supervisor] worker {worker_id:02d} restart limit reached", flush=True)
                    return 2
                time.sleep(args.restart_delay_seconds)
                launch_worker(args, state, restart=True)

        if all_done:
            print("[supervisor] all workers complete", flush=True)
            return 0

        time.sleep(args.poll_seconds)

    for state in states.values():
        stop_worker(state, args.kill_timeout_seconds)
    return 130


def main() -> int:
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)
    args = parse_args()
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
