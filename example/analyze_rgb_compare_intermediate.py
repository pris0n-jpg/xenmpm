"""
Analyze intermediate outputs produced by `example/mpm_fem_rgb_compare.py --export-intermediate`.

This script is dependency-light (numpy only) and writes a reproducible `analysis.csv`
to the given `--save-dir`. The goal is to help attribute common RGB artifacts
(dark blob / halo / edge streak) back to height-field and UV displacement signals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    return 1


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_int(value: object, *, name: str) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Invalid int for {name}: {value!r} ({e})")


def _list_intermediate_frames(intermediate_dir: Path) -> List[int]:
    frames: List[int] = []
    for path in intermediate_dir.glob("frame_*.npz"):
        stem = path.stem  # frame_XXXX
        try:
            frame = int(stem.split("_", 1)[1])
        except Exception:
            continue
        frames.append(frame)
    return sorted(set(frames))


def _phase_to_frames(available_frames: Sequence[int], frame_to_phase: Sequence[str]) -> Dict[str, List[int]]:
    by_phase: Dict[str, List[int]] = {}
    for frame in available_frames:
        if frame < 0 or frame >= len(frame_to_phase):
            continue
        phase = frame_to_phase[frame] or ""
        by_phase.setdefault(phase, []).append(frame)
    return by_phase


def _pick_evenly_spaced(frames: Sequence[int], *, count: int) -> List[int]:
    if count <= 0 or not frames:
        return []
    if len(frames) <= count:
        return list(frames)
    idxs = np.linspace(0, len(frames) - 1, num=count, dtype=int).tolist()
    return [frames[i] for i in sorted(set(int(x) for x in idxs))]


def _percentiles(values: np.ndarray, ps: Sequence[float]) -> List[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [math.nan for _ in ps]
    return [float(x) for x in np.percentile(finite, ps).tolist()]


def _compute_height_stats(height_field_mm: np.ndarray) -> Tuple[float, float, float, float]:
    finite = height_field_mm[np.isfinite(height_field_mm)]
    if finite.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    h_min = float(np.min(finite))
    h_p1, h_p99 = _percentiles(finite, [1, 99])
    gy, gx = np.gradient(height_field_mm.astype(np.float32, copy=False))
    grad_mag = np.sqrt(gx * gx + gy * gy)
    (grad_p99,) = _percentiles(grad_mag, [99])
    return h_min, float(h_p1), float(h_p99), float(grad_p99)


def _compute_uv_stats(uv_disp_mm: np.ndarray, *, edge_width: int) -> Tuple[float, float]:
    if uv_disp_mm.ndim != 3 or uv_disp_mm.shape[2] != 2:
        raise ValueError(f"Unexpected uv_disp_mm shape: {uv_disp_mm.shape}")

    u = uv_disp_mm[..., 0].astype(np.float32, copy=False)
    v = uv_disp_mm[..., 1].astype(np.float32, copy=False)
    mag = np.sqrt(u * u + v * v)
    (p99,) = _percentiles(mag, [99])

    edge_width = int(edge_width)
    if edge_width <= 0:
        return float(p99), math.nan
    h, w = mag.shape
    ew = min(edge_width, h // 2, w // 2)
    if ew <= 0:
        return float(p99), math.nan

    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:ew, :] = True
    edge_mask[-ew:, :] = True
    edge_mask[:, :ew] = True
    edge_mask[:, -ew:] = True
    edge_vals = mag[edge_mask]
    (edge_p99,) = _percentiles(edge_vals, [99])
    return float(p99), float(edge_p99)


def _classify_phenomena(
    *,
    height_min_mm: float,
    height_p1_mm: float,
    height_grad_p99_mm_per_cell: float,
    uv_p99_mm: float,
    uv_edge_p99_mm: float,
    height_outlier_threshold_mm: float,
    halo_grad_p99_threshold_mm_per_cell: float,
    edge_uv_spike_ratio: float,
    edge_uv_spike_min_mm: float,
) -> str:
    tags: List[str] = []
    depth_thr = abs(float(height_outlier_threshold_mm))
    if math.isfinite(height_min_mm) and height_min_mm < -depth_thr:
        tags.append("dark_blob_risk")
    if math.isfinite(height_p1_mm) and height_p1_mm < -depth_thr and "dark_blob_risk" not in tags:
        tags.append("dark_blob_risk")

    if math.isfinite(height_grad_p99_mm_per_cell) and height_grad_p99_mm_per_cell > float(halo_grad_p99_threshold_mm_per_cell):
        tags.append("halo_risk")

    if (
        math.isfinite(uv_edge_p99_mm)
        and math.isfinite(uv_p99_mm)
        and uv_edge_p99_mm > float(edge_uv_spike_min_mm)
        and uv_edge_p99_mm > float(edge_uv_spike_ratio) * max(float(uv_p99_mm), 1e-6)
    ):
        tags.append("edge_streak_risk")

    return ";".join(tags)


def _format_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def _write_notes(
    path: Path,
    *,
    save_dir: Path,
    analysis_csv: Path,
    sample_frames: Sequence[int],
    thresholds: Dict[str, float],
) -> None:
    try:
        rel_csv = analysis_csv.relative_to(save_dir)
    except Exception:
        rel_csv = analysis_csv

    lines = [
        "# RGB Compare Intermediate Analysis Notes",
        "",
        f"- save_dir: `{save_dir}`",
        f"- analysis_csv: `{rel_csv}`",
        f"- sampled_frames: `{','.join(str(i) for i in sample_frames)}`",
        "",
        "## Thresholds",
    ]
    for key in sorted(thresholds.keys()):
        lines.append(f"- {key}: `{thresholds[key]}`")
    lines.append("")
    lines.append("## Tags")
    lines.append("- `dark_blob_risk`: height_field_mm has extreme negative outliers (min/p1 below threshold).")
    lines.append("- `halo_risk`: height_field_mm has steep slopes (grad_p99 above threshold).")
    lines.append("- `edge_streak_risk`: uv_disp_mm spikes near image borders (edge_p99 >> p99).")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze rgb_compare intermediate outputs (numpy-only).")
    parser.add_argument("--save-dir", type=str, required=True, help="Output directory created by mpm_fem_rgb_compare.py --save-dir")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: <save-dir>/analysis.csv)")
    parser.add_argument("--sample-per-phase", type=int, default=3, help="Frames sampled per phase (press/slide/hold)")
    parser.add_argument("--frames", type=str, default=None, help="Comma-separated explicit frame ids (overrides sampling)")
    parser.add_argument("--edge-width", type=int, default=3, help="Border width (cells) for edge_uv statistics")
    parser.add_argument("--height-outlier-threshold-mm", type=float, default=5.0, help="Dark-blob risk threshold for extreme negative depth")
    parser.add_argument("--halo-grad-p99-threshold", type=float, default=0.20, help="Halo risk threshold for grad_p99 (mm per cell)")
    parser.add_argument("--edge-uv-spike-ratio", type=float, default=3.0, help="Edge streak ratio threshold: edge_p99 > ratio * p99")
    parser.add_argument("--edge-uv-spike-min-mm", type=float, default=0.20, help="Minimum edge_p99 (mm) to consider edge streak risk")
    parser.add_argument("--no-notes", action="store_true", default=False, help="Do not write analysis_notes.md")

    args = parser.parse_args(argv)
    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        return _fail(f"--save-dir not found: {save_dir}")

    manifest_path = save_dir / "run_manifest.json"
    if not manifest_path.exists():
        return _fail(f"Missing run_manifest.json: {manifest_path}")

    manifest = _read_json(manifest_path)
    traj = manifest.get("trajectory") or {}
    if not isinstance(traj, dict):
        return _fail("run_manifest.json trajectory is not a dict")

    total_frames = _parse_int(traj.get("total_frames"), name="trajectory.total_frames")
    frame_to_phase = traj.get("frame_to_phase") or []
    if not isinstance(frame_to_phase, list) or len(frame_to_phase) != total_frames:
        return _fail("Invalid trajectory.frame_to_phase in run_manifest.json")
    phase_list = [str(p or "") for p in frame_to_phase]

    intermediate_dir = save_dir / "intermediate"
    if not intermediate_dir.exists():
        return _fail(f"Missing intermediate dir: {intermediate_dir}")

    available_frames = _list_intermediate_frames(intermediate_dir)
    if not available_frames:
        return _fail(f"No intermediate frames found under: {intermediate_dir}")
    available_set = set(available_frames)

    if args.frames:
        try:
            requested = [int(x.strip()) for x in str(args.frames).split(",") if x.strip()]
        except Exception as e:
            return _fail(f"Invalid --frames: {args.frames!r} ({e})")
        sample_frames = sorted(set([f for f in requested if f in available_set]))
    else:
        by_phase = _phase_to_frames(available_frames, phase_list)
        sample_frames: List[int] = []
        for phase in ["press", "slide", "hold"]:
            sample_frames.extend(_pick_evenly_spaced(by_phase.get(phase, []), count=int(args.sample_per_phase)))
        sample_frames = sorted(set(sample_frames))

    if len(sample_frames) < 9:
        return _fail(f"Too few sampled frames: {len(sample_frames)} (need >= 9). Have: {sample_frames}")

    out_path = Path(args.out) if args.out else (save_dir / "analysis.csv")
    notes_path = save_dir / "analysis_notes.md"

    thresholds = {
        "height_outlier_threshold_mm": float(args.height_outlier_threshold_mm),
        "halo_grad_p99_threshold_mm_per_cell": float(args.halo_grad_p99_threshold),
        "edge_uv_spike_ratio": float(args.edge_uv_spike_ratio),
        "edge_uv_spike_min_mm": float(args.edge_uv_spike_min_mm),
        "edge_width": float(args.edge_width),
    }

    rows: List[Dict[str, str]] = []
    for frame_id in sample_frames:
        npz_path = intermediate_dir / f"frame_{int(frame_id):04d}.npz"
        if not npz_path.exists():
            return _fail(f"Missing intermediate frame: {npz_path}")

        data = np.load(npz_path)
        if "height_field_mm" not in data or "uv_disp_mm" not in data:
            return _fail(f"Intermediate missing required arrays: {npz_path}")

        height = data["height_field_mm"].astype(np.float32, copy=False)
        uv = data["uv_disp_mm"].astype(np.float32, copy=False)
        h_min, h_p1, h_p99, grad_p99 = _compute_height_stats(height)
        uv_p99, uv_edge_p99 = _compute_uv_stats(uv, edge_width=int(args.edge_width))

        phase = phase_list[int(frame_id)] if int(frame_id) < len(phase_list) else ""
        tags = _classify_phenomena(
            height_min_mm=h_min,
            height_p1_mm=h_p1,
            height_grad_p99_mm_per_cell=grad_p99,
            uv_p99_mm=uv_p99,
            uv_edge_p99_mm=uv_edge_p99,
            height_outlier_threshold_mm=float(args.height_outlier_threshold_mm),
            halo_grad_p99_threshold_mm_per_cell=float(args.halo_grad_p99_threshold),
            edge_uv_spike_ratio=float(args.edge_uv_spike_ratio),
            edge_uv_spike_min_mm=float(args.edge_uv_spike_min_mm),
        )

        rows.append(
            {
                "frame_id": str(int(frame_id)),
                "phase": str(phase),
                "npz": str(npz_path.relative_to(save_dir)),
                "height_min_mm": _format_float(h_min),
                "height_p1_mm": _format_float(h_p1),
                "height_p99_mm": _format_float(h_p99),
                "height_grad_p99_mm_per_cell": _format_float(grad_p99),
                "uv_disp_p99_mm": _format_float(uv_p99),
                "uv_disp_edge_p99_mm": _format_float(uv_edge_p99),
                "phenomena_tags": str(tags),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not bool(args.no_notes):
        _write_notes(
            notes_path,
            save_dir=save_dir,
            analysis_csv=out_path,
            sample_frames=sample_frames,
            thresholds=thresholds,
        )

    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

