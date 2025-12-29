"""
Validate outputs produced by `example/mpm_fem_rgb_compare.py --save-dir ...`.

This script is dependency-light and is intended to help audit/review regression
outputs in a reproducible way (no ezgl/taichi required).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    return 1


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _collect_files(save_dir: Path, pattern: str) -> List[Path]:
    return sorted(save_dir.glob(pattern))


def _parse_int(value: object, *, name: str) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Invalid int for {name}: {value!r} ({e})")


def _validate_manifest(manifest: Dict[str, object]) -> Tuple[int, List[str], Dict[str, object]]:
    if "trajectory" not in manifest:
        raise ValueError("run_manifest.json missing key: trajectory")
    traj = manifest.get("trajectory") or {}
    if not isinstance(traj, dict):
        raise ValueError("run_manifest.json trajectory is not a dict")

    total_frames = _parse_int(traj.get("total_frames"), name="trajectory.total_frames")
    if total_frames <= 0:
        raise ValueError(f"trajectory.total_frames must be > 0, got {total_frames}")

    frame_to_phase = traj.get("frame_to_phase") or []
    if not isinstance(frame_to_phase, list):
        raise ValueError("trajectory.frame_to_phase is not a list")
    if len(frame_to_phase) != total_frames:
        raise ValueError(
            f"trajectory.frame_to_phase length mismatch: {len(frame_to_phase)} vs total_frames={total_frames}"
        )
    phases = []
    for i, p in enumerate(frame_to_phase):
        if p is None:
            phases.append("")
            continue
        if not isinstance(p, str):
            raise ValueError(f"trajectory.frame_to_phase[{i}] is not a str: {p!r}")
        phases.append(p)

    deps = manifest.get("deps") or {}
    if not isinstance(deps, dict):
        deps = {}

    return total_frames, phases, deps


def _validate_metrics(save_dir: Path, *, total_frames: int) -> Tuple[Optional[Path], Optional[Path]]:
    metrics_csv = save_dir / "metrics.csv"
    metrics_json = save_dir / "metrics.json"
    if not metrics_csv.exists():
        raise FileNotFoundError(str(metrics_csv))
    if not metrics_json.exists():
        raise FileNotFoundError(str(metrics_json))

    rows = _read_metrics_csv(metrics_csv)
    if len(rows) != total_frames:
        raise ValueError(f"metrics.csv row count mismatch: {len(rows)} vs total_frames={total_frames}")

    frames: List[int] = []
    for i, row in enumerate(rows):
        if "frame" not in row:
            raise ValueError("metrics.csv missing column: frame")
        try:
            frame_id = int(str(row["frame"]).strip())
        except Exception:
            raise ValueError(f"metrics.csv invalid frame at row {i}: {row.get('frame')!r}")
        frames.append(frame_id)

    if len(set(frames)) != len(frames):
        raise ValueError("metrics.csv contains duplicated frame ids")
    if frames and (min(frames) != 0 or max(frames) != total_frames - 1):
        raise ValueError(f"metrics.csv frame id range mismatch: min={min(frames)} max={max(frames)}")

    return metrics_csv, metrics_json


def _validate_intermediate(save_dir: Path, *, enabled: bool, total_frames: int) -> Optional[Path]:
    intermediate_dir = save_dir / "intermediate"
    if not enabled:
        return None
    if not intermediate_dir.exists():
        raise FileNotFoundError(str(intermediate_dir))

    f0 = intermediate_dir / "frame_0000.npz"
    if not f0.exists():
        raise FileNotFoundError(str(f0))

    # 这里不强制每一帧都存在 intermediate（由 export_every 控制），只确保最小可用证据。
    return f0


def _validate_images(save_dir: Path, *, has_cv2: bool, total_frames: int) -> None:
    if not has_cv2:
        return
    fem = _collect_files(save_dir, "fem_*.png")
    mpm = _collect_files(save_dir, "mpm_*.png")
    if len(fem) != total_frames:
        raise ValueError(f"fem_*.png count mismatch: {len(fem)} vs total_frames={total_frames}")
    if len(mpm) != total_frames:
        raise ValueError(f"mpm_*.png count mismatch: {len(mpm)} vs total_frames={total_frames}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate outputs of mpm_fem_rgb_compare --save-dir")
    parser.add_argument("--save-dir", type=str, required=True, help="Output directory produced by mpm_fem_rgb_compare")
    args = parser.parse_args(argv)

    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        return _fail(f"Missing save-dir: {save_dir}")

    manifest_path = save_dir / "run_manifest.json"
    if not manifest_path.exists():
        return _fail(f"Missing run_manifest.json: {manifest_path}")

    try:
        manifest = _read_json(manifest_path)
        total_frames, _, deps = _validate_manifest(manifest)
    except Exception as e:
        return _fail(f"Invalid run_manifest.json: {e}")

    try:
        _validate_metrics(save_dir, total_frames=total_frames)
    except Exception as e:
        return _fail(f"Invalid metrics: {e}")

    run_context = manifest.get("run_context") or {}
    resolved = {}
    if isinstance(run_context, dict):
        resolved = run_context.get("resolved") or {}
    export = resolved.get("export") if isinstance(resolved, dict) else None
    export_enabled = bool(export.get("intermediate")) if isinstance(export, dict) else bool((save_dir / "intermediate").exists())

    try:
        _validate_intermediate(save_dir, enabled=export_enabled, total_frames=total_frames)
    except Exception as e:
        return _fail(f"Invalid intermediate: {e}")

    has_cv2 = bool(deps.get("has_cv2")) if isinstance(deps, dict) else False
    try:
        _validate_images(save_dir, has_cv2=has_cv2, total_frames=total_frames)
    except Exception as e:
        return _fail(f"Invalid frame images: {e}")

    print(f"OK: save_dir={save_dir} total_frames={total_frames}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

