"""
Analyze warp_marker_texture out-of-bounds (OOB) ratio on saved intermediate uv_disp_mm.

This script is meant to be runnable without taichi/ezgl. It reuses the exact
`warp_marker_texture()` implementation in `mpm_fem_rgb_compare.py` (same folder)
to compute OOB statistics for selected frames.
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

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


def _fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    return 1


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6g}"


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


def _parse_hw_arg(hw: Optional[str]) -> Optional[Tuple[int, int]]:
    if hw is None:
        return None
    s = str(hw).strip()
    if not s:
        return None
    if "," in s:
        a, b = s.split(",", 1)
    elif "x" in s:
        a, b = s.split("x", 1)
    else:
        raise ValueError("Invalid --tex-hw, expected 'H,W' or 'HxW'")
    h = int(a.strip())
    w = int(b.strip())
    if h <= 0 or w <= 0:
        raise ValueError("Invalid --tex-hw, H/W must be positive")
    return h, w


def _infer_tex_hw_from_images(save_dir: Path) -> Optional[Tuple[int, int]]:
    if Image is None:
        return None
    for name in ("mpm_0000.png", "fem_0000.png"):
        p = save_dir / name
        if p.exists():
            with Image.open(p) as im:
                w, h = im.size
                return int(h), int(w)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze warp_marker_texture OOB ratio (reuses mpm_fem_rgb_compare.warp_marker_texture).")
    parser.add_argument("--save-dir", type=str, required=True, help="Output directory created by mpm_fem_rgb_compare.py --save-dir")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: <save-dir>/warp_oob_stats.csv)")
    parser.add_argument("--frames", type=str, default="75,80,85", help="Comma-separated frame ids (default: 75,80,85)")
    parser.add_argument("--tex-hw", type=str, default=None, help="Override marker texture size as 'H,W' or 'HxW' (default: infer from mpm_0000.png)")

    args = parser.parse_args(argv)
    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        return _fail(f"--save-dir not found: {save_dir}")

    manifest_path = save_dir / "run_manifest.json"
    if not manifest_path.exists():
        return _fail(f"Missing run_manifest.json: {manifest_path}")
    manifest = _read_json(manifest_path)

    scene_params = manifest.get("scene_params") or {}
    if not isinstance(scene_params, dict):
        return _fail("run_manifest.json scene_params is not a dict")

    gel_size_mm = scene_params.get("gel_size_mm", None)
    if not isinstance(gel_size_mm, list) or len(gel_size_mm) != 2:
        return _fail("Missing/invalid scene_params.gel_size_mm")
    gel_w_mm, gel_h_mm = float(gel_size_mm[0]), float(gel_size_mm[1])
    flip_x = bool(scene_params.get("mpm_warp_flip_x", False))
    flip_y = bool(scene_params.get("mpm_warp_flip_y", False))
    render_flip_x = bool(scene_params.get("mpm_render_flip_x", False))

    traj = manifest.get("trajectory") or {}
    if not isinstance(traj, dict):
        return _fail("run_manifest.json trajectory is not a dict")
    frame_to_phase = traj.get("frame_to_phase") or []
    if not isinstance(frame_to_phase, list) or not frame_to_phase:
        return _fail("Invalid trajectory.frame_to_phase in run_manifest.json")
    phase_list = [str(p or "") for p in frame_to_phase]

    frames = _parse_frames_arg(args.frames)
    if not frames:
        return _fail("No frames specified")

    try:
        tex_hw = _parse_hw_arg(args.tex_hw)
    except Exception as e:
        return _fail(str(e))
    if tex_hw is None:
        tex_hw = _infer_tex_hw_from_images(save_dir)
    if tex_hw is None:
        return _fail("Cannot infer texture size; install Pillow or pass --tex-hw")
    tex_h, tex_w = int(tex_hw[0]), int(tex_hw[1])

    # Dummy base texture: we only need its shape + border color (picked from [0,0]).
    base_tex = np.full((tex_h, tex_w, 3), 255, dtype=np.uint8)

    intermediate_dir = save_dir / "intermediate"
    if not intermediate_dir.exists():
        return _fail(f"Missing intermediate dir: {intermediate_dir}")

    # Local import: this script is under example/, same folder as mpm_fem_rgb_compare.py
    try:
        import mpm_fem_rgb_compare as compare  # type: ignore
    except Exception as e:
        return _fail(f"Failed to import mpm_fem_rgb_compare.py: {e}")
    if not hasattr(compare, "warp_marker_texture"):
        return _fail("mpm_fem_rgb_compare.warp_marker_texture not found")

    out_csv = Path(args.out) if args.out else (save_dir / "warp_oob_stats.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["frame_id", "phase", "tex_w", "tex_h", "oob_px", "oob_ratio"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for frame_id in frames:
            npz_path = intermediate_dir / f"frame_{int(frame_id):04d}.npz"
            if not npz_path.exists():
                return _fail(f"Missing intermediate frame: {npz_path}")
            data = np.load(npz_path)
            if "uv_disp_mm" not in data:
                return _fail(f"Intermediate missing uv_disp_mm: {npz_path}")
            uv = data["uv_disp_mm"].astype(np.float32, copy=False)
            if render_flip_x:
                uv = uv[:, ::-1, :].copy()
            phase = phase_list[int(frame_id)] if 0 <= int(frame_id) < len(phase_list) else ""

            stats: Dict[str, float] = {}
            _ = compare.warp_marker_texture(
                base_tex,
                uv,
                gel_size_mm=(gel_w_mm, gel_h_mm),
                flip_x=flip_x,
                flip_y=flip_y,
                stats_out=stats,
            )
            w.writerow(
                {
                    "frame_id": str(int(frame_id)),
                    "phase": str(phase),
                    "tex_w": str(int(tex_w)),
                    "tex_h": str(int(tex_h)),
                    "oob_px": str(int(stats.get("oob_px", 0.0))),
                    "oob_ratio": _format_float(float(stats.get("oob_ratio", math.nan))),
                }
            )

    rel_csv = out_csv.relative_to(save_dir) if out_csv.is_relative_to(save_dir) else out_csv
    print(f"OK: wrote {rel_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

