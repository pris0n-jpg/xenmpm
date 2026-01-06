"""
Simulate `mpm_height_clip_outliers` effect on saved intermediate height_field_mm.

This script is dependency-light (numpy only) and can run without taichi/ezgl.

It reads `intermediate/frame_XXXX.npz` and compares:
- Original height stats / tags.
- After applying outlier clipping outside contact_mask:
  height_field_mm < -clip_min_mm  =>  NaN

Goal: quantify whether halo_risk (steep height gradients) is reduced on key frames
without rerunning the full render pipeline.
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


def _format_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6g}"


def _percentiles(values: np.ndarray, ps: Sequence[float]) -> List[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [math.nan for _ in ps]
    return [float(x) for x in np.percentile(finite, ps).tolist()]


def _parse_frames_arg(frames: Optional[str]) -> Optional[List[int]]:
    if frames is None:
        return None
    s = str(frames).strip()
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


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


def _classify_phenomena(
    *,
    height_min_mm: float,
    height_p1_mm: float,
    height_grad_p99_mm_per_cell: float,
    height_outlier_threshold_mm: float,
    halo_grad_p99_threshold_mm_per_cell: float,
) -> str:
    tags: List[str] = []
    depth_thr = abs(float(height_outlier_threshold_mm))
    if math.isfinite(height_min_mm) and height_min_mm < -depth_thr:
        tags.append("dark_blob_risk")
    if math.isfinite(height_p1_mm) and height_p1_mm < -depth_thr and "dark_blob_risk" not in tags:
        tags.append("dark_blob_risk")
    if math.isfinite(height_grad_p99_mm_per_cell) and height_grad_p99_mm_per_cell > float(halo_grad_p99_threshold_mm_per_cell):
        tags.append("halo_risk")
    return ";".join(tags)


def _apply_clip_outliers(
    height_field_mm: np.ndarray,
    contact_mask: np.ndarray,
    *,
    clip_min_mm: float,
) -> Tuple[np.ndarray, int, float]:
    floor_mm = -abs(float(clip_min_mm))
    height = height_field_mm.astype(np.float32, copy=True)
    contact = (contact_mask > 0)
    finite = np.isfinite(height)
    outliers = (~contact) & finite & (height < floor_mm)
    outlier_count = int(np.sum(outliers))
    outlier_ratio = float(outlier_count / max(int(height.size), 1))
    if outlier_count > 0:
        height[outliers] = np.nan
    return height, outlier_count, outlier_ratio


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate mpm_height_clip_outliers effect on saved intermediate height_field_mm (numpy-only).")
    parser.add_argument("--save-dir", type=str, required=True, help="Output directory created by mpm_fem_rgb_compare.py --save-dir")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: <save-dir>/height_clip_outliers_effect.csv)")
    parser.add_argument("--out-md", type=str, default=None, help="Output Markdown path (default: <save-dir>/height_clip_outliers_effect.md)")
    parser.add_argument("--no-md", action="store_true", default=False, help="Do not write the Markdown summary note")
    parser.add_argument("--frames", type=str, default="75,80,85", help="Comma-separated frame ids (default: 75,80,85)")
    parser.add_argument("--clip-min-mm", type=float, default=2.0, help="clip_min_mm (values < -clip_min_mm outside contact are set to NaN)")
    parser.add_argument("--height-outlier-threshold-mm", type=float, default=5.0, help="dark_blob_risk threshold for extreme negative depth")
    parser.add_argument("--halo-grad-p99-threshold", type=float, default=0.20, help="halo_risk threshold for grad_p99 (mm per cell)")

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
    frame_to_phase = traj.get("frame_to_phase") or []
    if not isinstance(frame_to_phase, list) or not frame_to_phase:
        return _fail("Invalid trajectory.frame_to_phase in run_manifest.json")
    phase_list = [str(p or "") for p in frame_to_phase]

    intermediate_dir = save_dir / "intermediate"
    if not intermediate_dir.exists():
        return _fail(f"Missing intermediate dir: {intermediate_dir}")

    frames = _parse_frames_arg(args.frames)
    if not frames:
        return _fail("No frames specified")

    out_csv = Path(args.out) if args.out else (save_dir / "height_clip_outliers_effect.csv")
    out_md = Path(args.out_md) if args.out_md else (save_dir / "height_clip_outliers_effect.md")

    rows: List[Dict[str, str]] = []
    md_rows: List[str] = []
    for frame_id in frames:
        npz_path = intermediate_dir / f"frame_{int(frame_id):04d}.npz"
        if not npz_path.exists():
            return _fail(f"Missing intermediate frame: {npz_path}")
        data = np.load(npz_path)
        if "height_field_mm" not in data or "contact_mask" not in data:
            return _fail(f"Intermediate missing required arrays: {npz_path}")
        height = data["height_field_mm"].astype(np.float32, copy=False)
        contact_mask = data["contact_mask"].astype(np.uint8, copy=False)
        phase = phase_list[int(frame_id)] if 0 <= int(frame_id) < len(phase_list) else ""

        h_min, h_p1, h_p99, grad_p99 = _compute_height_stats(height)
        tags = _classify_phenomena(
            height_min_mm=h_min,
            height_p1_mm=h_p1,
            height_grad_p99_mm_per_cell=grad_p99,
            height_outlier_threshold_mm=float(args.height_outlier_threshold_mm),
            halo_grad_p99_threshold_mm_per_cell=float(args.halo_grad_p99_threshold),
        )

        clipped, outlier_count, outlier_ratio = _apply_clip_outliers(
            height,
            contact_mask,
            clip_min_mm=float(args.clip_min_mm),
        )
        c_min, c_p1, c_p99, c_grad_p99 = _compute_height_stats(clipped)
        c_tags = _classify_phenomena(
            height_min_mm=c_min,
            height_p1_mm=c_p1,
            height_grad_p99_mm_per_cell=c_grad_p99,
            height_outlier_threshold_mm=float(args.height_outlier_threshold_mm),
            halo_grad_p99_threshold_mm_per_cell=float(args.halo_grad_p99_threshold),
        )

        rows.append(
            {
                "frame_id": str(int(frame_id)),
                "phase": str(phase),
                "clip_min_mm": _format_float(float(args.clip_min_mm)),
                "outlier_count": str(int(outlier_count)),
                "outlier_ratio": _format_float(outlier_ratio),
                "orig_height_min_mm": _format_float(h_min),
                "orig_height_p1_mm": _format_float(h_p1),
                "orig_height_p99_mm": _format_float(h_p99),
                "orig_grad_p99_mm_per_cell": _format_float(grad_p99),
                "orig_tags": tags,
                "clipped_height_min_mm": _format_float(c_min),
                "clipped_height_p1_mm": _format_float(c_p1),
                "clipped_height_p99_mm": _format_float(c_p99),
                "clipped_grad_p99_mm_per_cell": _format_float(c_grad_p99),
                "clipped_tags": c_tags,
            }
        )

        md_rows.append(
            f"- frame_{int(frame_id):04d} ({phase}): "
            f"grad_p99 {_format_float(grad_p99)} -> {_format_float(c_grad_p99)} | "
            f"tags '{tags}' -> '{c_tags}' | outliers={outlier_count} ({_format_float(outlier_ratio)})"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if not args.no_md:
        lines: List[str] = [
            "# RGB Compare Height Clip Outliers Effect",
            "",
            f"- save_dir: `{save_dir.as_posix()}`",
            f"- frames: `{','.join(str(int(x)) for x in frames)}`",
            "",
            "## Params",
            f"- clip_min_mm: `{float(args.clip_min_mm)}`",
            f"- height_outlier_threshold_mm: `{float(args.height_outlier_threshold_mm)}`",
            f"- halo_grad_p99_threshold_mm_per_cell: `{float(args.halo_grad_p99_threshold)}`",
            "",
            "## Summary",
            *md_rows,
            "",
            "## Notes",
            "",
            "本脚本仅对 intermediate 的 `height_field_mm` 做离线裁剪对比，用于量化 halo_risk 相关统计的变化；",
            "不需要重新运行 taichi/ezgl 渲染链路。",
            "",
        ]
        out_md.write_text("\n".join(lines), encoding="utf-8")

    rel_csv = out_csv.relative_to(save_dir) if out_csv.is_relative_to(save_dir) else out_csv
    print(f"OK: wrote {rel_csv}", flush=True)
    if not args.no_md:
        rel_md = out_md.relative_to(save_dir) if out_md.is_relative_to(save_dir) else out_md
        print(f"OK: wrote {rel_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

