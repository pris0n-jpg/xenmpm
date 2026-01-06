"""
Analyze flip/alignment between rendered frames and intermediate fields produced by
`example/mpm_fem_rgb_compare.py --export-intermediate --save-dir ...`.

Goal
----
Provide quantitative evidence for whether the MPM height/UV fields should be
horizontally flipped (x-axis) to match the rendered image convention, and help
diagnose “mirror” artifacts between FEM/MPM outputs.

This script is dependency-light: numpy + Pillow only (no ezgl/taichi/cv2 needed).
It writes a reproducible CSV (`alignment_flip.csv`) under the given `--save-dir`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


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


def _list_frames_by_glob(save_dir: Path, pattern: str) -> List[int]:
    frames: List[int] = []
    for p in save_dir.glob(pattern):
        stem = p.stem  # e.g. fem_0085
        try:
            frame = int(stem.split("_", 1)[1])
        except Exception:
            continue
        frames.append(frame)
    return sorted(set(frames))


def _pick_evenly_spaced(frames: Sequence[int], *, count: int) -> List[int]:
    if count <= 0 or not frames:
        return []
    if len(frames) <= count:
        return list(frames)
    idxs = np.linspace(0, len(frames) - 1, num=count, dtype=int).tolist()
    return [frames[i] for i in sorted(set(int(x) for x in idxs))]


def _weighted_diff_centroid(cur_rgb: np.ndarray, ref_rgb: np.ndarray, *, p: float = 99.0) -> Optional[Tuple[float, float]]:
    if cur_rgb.shape != ref_rgb.shape:
        raise ValueError(f"shape mismatch: {cur_rgb.shape} vs {ref_rgb.shape}")
    diff = np.abs(cur_rgb.astype(np.int16) - ref_rgb.astype(np.int16)).sum(axis=2).astype(np.float32)
    thr = float(np.percentile(diff, p))
    mask = diff >= thr
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    w = diff[mask]
    wsum = float(np.sum(w))
    if wsum <= 1e-9:
        return None
    cx = float(np.sum(xs.astype(np.float64) * w.astype(np.float64)) / wsum)
    cy = float(np.sum(ys.astype(np.float64) * w.astype(np.float64)) / wsum)
    return cx, cy


def _load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im, dtype=np.uint8)


def _contact_centroid_from_height(height_field_mm: np.ndarray, *, contact_thr_mm: float) -> Optional[Tuple[float, float]]:
    if height_field_mm.ndim != 2:
        raise ValueError(f"Unexpected height_field_mm shape: {height_field_mm.shape}")
    mask = np.isfinite(height_field_mm) & (height_field_mm < float(contact_thr_mm))
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    # Use uniform weights to reduce sensitivity to extreme outliers.
    cx = float(np.mean(xs.astype(np.float64)))
    cy = float(np.mean(ys.astype(np.float64)))
    return cx, cy


def _grid_to_pixel_x(col_f: float, *, n_col: int, img_w: int) -> float:
    if n_col <= 1:
        return 0.0
    return float(col_f) / float(n_col - 1) * float(max(img_w - 1, 1))


@dataclass(frozen=True)
class AlignmentRow:
    frame: int
    phase: str
    fem_img_cx: Optional[float]
    mpm_img_cx: Optional[float]
    fem_mirror_cx: Optional[float]
    mpm_vs_fem: str
    uv_px_noflip: Optional[float]
    uv_px_flip: Optional[float]
    uv_best: str
    uv_delta_noflip_px: Optional[float]
    uv_delta_flip_px: Optional[float]


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    if not np.isfinite(float(v)):
        return ""
    return f"{float(v):.3f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze flip/alignment between rendered frames and intermediate fields.")
    parser.add_argument("--save-dir", type=str, required=True, help="Output directory produced by mpm_fem_rgb_compare.py --save-dir")
    parser.add_argument("--frames", type=str, default=None, help="Comma-separated explicit frame ids (e.g. 75,80,85).")
    parser.add_argument("--sample", type=int, default=3, help="Sample N frames from available outputs when --frames is not set.")
    parser.add_argument("--contact-thr-mm", type=float, default=-0.01, help="Contact threshold for height_field_mm (mm).")
    parser.add_argument("--uv-percentile", type=float, default=95.0, help="Percentile for selecting high-magnitude uv_disp_mm region.")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: <save-dir>/alignment_flip.csv)")
    parser.add_argument(
        "--require-mpm-vs-fem",
        type=str,
        choices=["direct", "mirror"],
        default=None,
        help="If set, fail when the majority of sampled frames disagree on whether MPM motion matches FEM directly or via horizontal mirror.",
    )
    parser.add_argument(
        "--require-uv-best",
        type=str,
        choices=["noflip", "flip"],
        default=None,
        help="If set, fail when the majority of sampled frames disagree on whether uv_disp grid X should be treated as flipped to match rendered MPM motion.",
    )
    parser.add_argument(
        "--min-pass-ratio",
        type=float,
        default=0.8,
        help="Minimum ratio of agreeing frames for --require-* checks (default: 0.8).",
    )
    parser.add_argument(
        "--min-known-frames",
        type=int,
        default=2,
        help="Minimum number of frames with a determinate signal for --require-* checks (default: 2).",
    )
    args = parser.parse_args(argv)

    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        return _fail(f"--save-dir not found: {save_dir}")

    manifest_path = save_dir / "run_manifest.json"
    if not manifest_path.exists():
        return _fail(f"Missing run_manifest.json: {manifest_path}")
    manifest = _read_json(manifest_path)

    # Print flip conventions for traceability (so results are “self-explaining” when shared).
    run_context = manifest.get("run_context") or {}
    if isinstance(run_context, dict):
        resolved = run_context.get("resolved") or {}
        if isinstance(resolved, dict):
            conventions = resolved.get("conventions") or {}
            if isinstance(conventions, dict):
                print(
                    "conventions:"
                    f" uv_disp_flip_x={conventions.get('mpm_uv_disp_flip_x')} u_negate={conventions.get('mpm_uv_disp_u_negate')}"
                    f" warp_flip_x={conventions.get('mpm_warp_flip_x')} warp_flip_y={conventions.get('mpm_warp_flip_y')}",
                    flush=True,
                )
    scene_params = manifest.get("scene_params") or {}
    if isinstance(scene_params, dict):
        print(
            "scene_params:"
            f" mpm_render_flip_x={scene_params.get('mpm_render_flip_x')}"
            f" mpm_warp_flip_x={scene_params.get('mpm_warp_flip_x')}"
            f" mpm_warp_flip_y={scene_params.get('mpm_warp_flip_y')}",
            flush=True,
        )
    traj = manifest.get("trajectory") or {}
    if not isinstance(traj, dict):
        return _fail("run_manifest.json trajectory is not a dict")

    total_frames = _parse_int(traj.get("total_frames"), name="trajectory.total_frames")
    frame_to_phase = traj.get("frame_to_phase") or []
    if not isinstance(frame_to_phase, list) or len(frame_to_phase) != total_frames:
        return _fail("Invalid trajectory.frame_to_phase in run_manifest.json")
    phase_list = [str(p or "") for p in frame_to_phase]

    # Determine available frames from intermediate and images.
    intermediate_dir = save_dir / "intermediate"
    if not intermediate_dir.exists():
        return _fail(f"Missing intermediate dir: {intermediate_dir} (need --export-intermediate)")

    available_intermediate = _list_frames_by_glob(intermediate_dir, "frame_*.npz")
    available_fem = _list_frames_by_glob(save_dir, "fem_*.png")
    available_mpm = _list_frames_by_glob(save_dir, "mpm_*.png")
    available = sorted(set(available_intermediate) & set(available_fem) & set(available_mpm))
    if not available:
        return _fail("No common frames found (need fem_*.png/mpm_*.png and intermediate/frame_*.npz)")

    frames = _parse_frames_arg(args.frames)
    if frames is None:
        frames = _pick_evenly_spaced(available, count=int(args.sample))
    else:
        missing = [f for f in frames if f not in available]
        if missing:
            return _fail(f"Frames not available in save-dir: {missing} (available: {available[:10]} ...)")

    fem0_path = save_dir / "fem_0000.png"
    mpm0_path = save_dir / "mpm_0000.png"
    if not fem0_path.exists() or not mpm0_path.exists():
        return _fail("Missing fem_0000.png or mpm_0000.png for diff centroid reference")
    fem0 = _load_rgb(fem0_path)
    mpm0 = _load_rgb(mpm0_path)
    img_h, img_w = int(fem0.shape[0]), int(fem0.shape[1])

    out_csv = Path(args.out) if args.out else (save_dir / "alignment_flip.csv")
    rows_out: List[AlignmentRow] = []

    for frame in frames:
        phase = phase_list[frame] if 0 <= frame < len(phase_list) else ""

        fem_path = save_dir / f"fem_{frame:04d}.png"
        mpm_path = save_dir / f"mpm_{frame:04d}.png"
        npz_path = intermediate_dir / f"frame_{frame:04d}.npz"

        fem = _load_rgb(fem_path)
        mpm = _load_rgb(mpm_path)
        fem_c = _weighted_diff_centroid(fem, fem0)
        mpm_c = _weighted_diff_centroid(mpm, mpm0)
        fem_img_cx = float(fem_c[0]) if fem_c is not None else None
        mpm_img_cx = float(mpm_c[0]) if mpm_c is not None else None
        fem_mirror_cx = (float((img_w - 1) - fem_img_cx) if fem_img_cx is not None else None)
        mpm_vs_fem = "unknown"
        if fem_img_cx is not None and mpm_img_cx is not None:
            direct = abs(mpm_img_cx - fem_img_cx)
            mirror = abs(mpm_img_cx - float((img_w - 1) - fem_img_cx))
            mpm_vs_fem = "mirror" if mirror < direct else "direct"

        data = np.load(npz_path)
        height = data.get("height_field_mm")
        if height is None:
            return _fail(f"Missing height_field_mm in {npz_path}")
        height = height.astype(np.float32, copy=False)
        uv = data.get("uv_disp_mm")
        if uv is None:
            return _fail(f"Missing uv_disp_mm in {npz_path}")
        uv = uv.astype(np.float32, copy=False)

        # Use high-magnitude uv_disp region as a proxy for the active contact/shear area.
        mag = np.sqrt(np.sum(uv * uv, axis=-1))
        thr = float(np.percentile(mag[np.isfinite(mag)], float(args.uv_percentile))) if np.isfinite(mag).any() else float("nan")
        uv_mask = np.isfinite(mag) & (mag >= thr)
        if not uv_mask.any():
            # Fallback: try height-based contact mask
            contact_c = _contact_centroid_from_height(height, contact_thr_mm=float(args.contact_thr_mm))
            uv_c = contact_c
        else:
            ys, xs = np.nonzero(uv_mask)
            uv_c = (float(np.mean(xs.astype(np.float64))), float(np.mean(ys.astype(np.float64))))

        uv_px_noflip = None
        uv_px_flip = None
        uv_best = ""
        uv_delta_noflip = None
        uv_delta_flip = None
        if uv_c is not None:
            col_f = float(uv_c[0])
            n_col = int(uv.shape[1])
            px = _grid_to_pixel_x(col_f, n_col=n_col, img_w=img_w)
            uv_px_noflip = px
            uv_px_flip = float((img_w - 1) - px)

            if mpm_img_cx is not None:
                uv_delta_noflip = abs(mpm_img_cx - uv_px_noflip)
                uv_delta_flip = abs(mpm_img_cx - uv_px_flip)
                uv_best = "noflip" if uv_delta_noflip <= uv_delta_flip else "flip"
            else:
                uv_best = "unknown"
        else:
            uv_best = "no_signal"

        rows_out.append(
            AlignmentRow(
                frame=int(frame),
                phase=phase,
                fem_img_cx=fem_img_cx,
                mpm_img_cx=mpm_img_cx,
                fem_mirror_cx=fem_mirror_cx,
                mpm_vs_fem=mpm_vs_fem,
                uv_px_noflip=uv_px_noflip,
                uv_px_flip=uv_px_flip,
                uv_best=uv_best,
                uv_delta_noflip_px=uv_delta_noflip,
                uv_delta_flip_px=uv_delta_flip,
            )
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame",
                "phase",
                "fem_img_cx",
                "mpm_img_cx",
                "fem_mirror_cx",
                "mpm_vs_fem",
                "uv_px_noflip",
                "uv_px_flip",
                "uv_best",
                "uv_delta_noflip_px",
                "uv_delta_flip_px",
            ]
        )
        for r in rows_out:
            w.writerow(
                [
                    r.frame,
                    r.phase,
                    _fmt(r.fem_img_cx),
                    _fmt(r.mpm_img_cx),
                    _fmt(r.fem_mirror_cx),
                    r.mpm_vs_fem,
                    _fmt(r.uv_px_noflip),
                    _fmt(r.uv_px_flip),
                    r.uv_best,
                    _fmt(r.uv_delta_noflip_px),
                    _fmt(r.uv_delta_flip_px),
                ]
            )

    best_counts: Dict[str, int] = {}
    for r in rows_out:
        best_counts[r.uv_best] = best_counts.get(r.uv_best, 0) + 1
    print(f"OK: save_dir={save_dir} frames={len(rows_out)} out={out_csv}", flush=True)
    print(f"best_counts={best_counts}", flush=True)
    for r in rows_out:
        print(
            f"frame={r.frame:04d} phase={r.phase} "
            f"mpm_vs_fem={r.mpm_vs_fem} mpm_cx={_fmt(r.mpm_img_cx)} fem_cx={_fmt(r.fem_img_cx)} mirror(fem)={_fmt(r.fem_mirror_cx)} "
            f"uv_best={r.uv_best} uv(noflip/flip)={_fmt(r.uv_px_noflip)}/{_fmt(r.uv_px_flip)} "
            f"uv_delta={_fmt(r.uv_delta_noflip_px)}/{_fmt(r.uv_delta_flip_px)}",
            flush=True,
        )

    def _known_values(value: str, *, known: Sequence[str]) -> bool:
        return value in set(known)

    def _validate_majority(*, name: str, values: Sequence[str], expected: str, known: Sequence[str]) -> int:
        picked = [v for v in values if _known_values(v, known=known)]
        if len(picked) < int(args.min_known_frames):
            return _fail(
                f"{name}: insufficient signal (known={len(picked)} < min_known_frames={args.min_known_frames}); "
                f"try specifying --frames to include slide/strong-contact frames"
            )
        ok = sum(1 for v in picked if v == expected)
        ratio = float(ok) / float(len(picked)) if picked else 0.0
        if ratio < float(args.min_pass_ratio):
                return _fail(
                    f"{name}: expected={expected} ok={ok}/{len(picked)} ratio={ratio:.3f} < min_pass_ratio={float(args.min_pass_ratio):.3f}; "
                    "this usually indicates an extra/missing horizontal flip. "
                    "Check: --mpm-render-flip-x / --mpm-warp-flip-x / --mpm-warp-flip-y and run_manifest.json run_context.resolved.conventions / scene_params."
                )
        return 0

    if args.require_mpm_vs_fem:
        rc = _validate_majority(
            name="mpm_vs_fem",
            values=[r.mpm_vs_fem for r in rows_out],
            expected=str(args.require_mpm_vs_fem),
            known=["direct", "mirror"],
        )
        if rc != 0:
            return rc

    if args.require_uv_best:
        rc = _validate_majority(
            name="uv_best",
            values=[r.uv_best for r in rows_out],
            expected=str(args.require_uv_best),
            known=["noflip", "flip"],
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
