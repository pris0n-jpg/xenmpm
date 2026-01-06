"""
Ablate surface field post-processing knobs (fill holes / smooth) from existing rgb_compare outputs.

This script is numpy-only and reads intermediate npz files produced by:
`example/mpm_fem_rgb_compare.py --export-intermediate --save-dir <dir>`

It writes a Markdown report that summarizes:
- uv_disp_mm coverage/spike stats in the contact region
- a halo proxy metric based on height_field_mm gradient p99 (raw vs after box blur smoothing)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SURFACE_KEYS = {
    "mpm_height_fill_holes",
    "mpm_height_fill_holes_iters",
    "mpm_height_smooth",
    "mpm_height_smooth_iters",
}


def _fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    return 1


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6g}"


def _percentile(values: np.ndarray, p: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.percentile(finite, [p]).tolist()[0])


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def _join_argv(argv: object) -> str:
    if isinstance(argv, list) and all(isinstance(x, str) for x in argv):
        return "python " + " ".join(argv)
    if isinstance(argv, str):
        return argv
    return ""


def _list_intermediate_frames(intermediate_dir: Path) -> List[int]:
    frames: List[int] = []
    for path in intermediate_dir.glob("frame_*.npz"):
        stem = path.stem
        if not stem.startswith("frame_"):
            continue
        try:
            frames.append(int(stem.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(frames))


def _pick_default_frames(common_frames: Sequence[int]) -> List[int]:
    # 约定优先：baseline 关键帧 75/80/85；若不可用则选尾部 3 帧（更接近稳定接触/滑移阶段）。
    want = [75, 80, 85]
    available = set(int(x) for x in common_frames)
    if all(x in available for x in want):
        return want
    frames = [int(x) for x in common_frames]
    if len(frames) <= 3:
        return frames
    if len(frames) <= 15:
        return frames[-3:]
    tail = frames[-15:]
    idxs = np.linspace(0, len(tail) - 1, num=3, dtype=int).tolist()
    return [tail[i] for i in sorted(set(int(x) for x in idxs))]


def _phase_ranges_frames(manifest: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    traj = manifest.get("trajectory") or {}
    if not isinstance(traj, dict):
        return {}
    pr = traj.get("phase_ranges_frames") or {}
    if not isinstance(pr, dict):
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for name in ("press", "slide", "hold"):
        rng = pr.get(name) or {}
        if not isinstance(rng, dict):
            continue
        try:
            out[name] = {
                "start_frame": int(rng.get("start_frame")),
                "end_frame": int(rng.get("end_frame")),
            }
        except Exception:
            continue
    return out


def _pick_phase_aligned_frames(manifest: Dict[str, Any]) -> List[int]:
    # 为什么：不同 run 的 record_interval/total_frames 可能不同，直接用相同 frame_id 会导致时间不对齐；
    # 这里改用 phase 对齐：press_end / slide_mid / hold_end。
    pr = _phase_ranges_frames(manifest)
    if not all(k in pr for k in ("press", "slide", "hold")):
        raise ValueError("Missing trajectory.phase_ranges_frames for press/slide/hold")
    press_end = int(pr["press"]["end_frame"])
    slide_start = int(pr["slide"]["start_frame"])
    slide_end = int(pr["slide"]["end_frame"])
    hold_end = int(pr["hold"]["end_frame"])
    slide_mid = int((slide_start + slide_end) // 2)
    return [press_end, slide_mid, hold_end]


def _box_blur_2d(values: np.ndarray, iterations: int = 1) -> np.ndarray:
    # 为什么：复现 MPMSensorScene._box_blur_2d 的 3x3 平滑，用于评估 mpm_height_smooth 对 halo 的影响。
    result = values.astype(np.float32, copy=True)
    for _ in range(max(int(iterations), 0)):
        padded = np.pad(result, ((1, 1), (1, 1)), mode="edge")
        result = (
            padded[0:-2, 0:-2]
            + padded[0:-2, 1:-1]
            + padded[0:-2, 2:]
            + padded[1:-1, 0:-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, 0:-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return result


def _height_grad_p99(height_field_mm: np.ndarray) -> float:
    height = height_field_mm.astype(np.float32, copy=False)
    gy, gx = np.gradient(height)
    grad = np.sqrt(gx * gx + gy * gy)
    return _percentile(grad, 99)


def _contact_boundary_mask(contact_mask: np.ndarray) -> np.ndarray:
    # 为什么：fill holes 关闭时，硬边/空洞更可能让尖峰集中在接触边界（而非整体滑移）。
    m = (contact_mask > 0).astype(np.bool_)
    up = np.roll(m, 1, axis=0)
    down = np.roll(m, -1, axis=0)
    left = np.roll(m, 1, axis=1)
    right = np.roll(m, -1, axis=1)
    interior = up & down & left & right
    boundary = m & (~interior)
    boundary[0, :] = m[0, :]
    boundary[-1, :] = m[-1, :]
    boundary[:, 0] = m[:, 0]
    boundary[:, -1] = m[:, -1]
    return boundary


@dataclass(frozen=True)
class ContactUvStats:
    contact_px: int
    finite_ratio: float
    motion_ratio: float
    u_p50: float
    uv_p50: float
    spike_ratio: float
    spike_boundary_ratio: float


def _compute_contact_uv_stats(
    uv_disp_mm: np.ndarray,
    contact_mask: np.ndarray,
    *,
    motion_eps_mm: float,
    spike_min_mm: float,
) -> ContactUvStats:
    if uv_disp_mm.ndim != 3 or uv_disp_mm.shape[-1] != 2:
        raise ValueError(f"uv_disp_mm must be HxWx2, got shape={uv_disp_mm.shape}")
    if contact_mask.ndim != 2 or contact_mask.shape != uv_disp_mm.shape[:2]:
        raise ValueError(f"contact_mask must be HxW, got shape={contact_mask.shape} vs uv={uv_disp_mm.shape[:2]}")

    contact = (contact_mask > 0).astype(np.bool_)
    contact_px = int(np.sum(contact))
    if contact_px <= 0:
        nan = math.nan
        return ContactUvStats(
            contact_px=0,
            finite_ratio=nan,
            motion_ratio=nan,
            u_p50=nan,
            uv_p50=nan,
            spike_ratio=nan,
            spike_boundary_ratio=0.0,
        )

    uv = uv_disp_mm.astype(np.float32, copy=False)
    uv_contact = uv[contact]  # (N, 2)
    finite = np.isfinite(uv_contact).all(axis=1)
    finite_ratio = float(np.mean(finite)) if uv_contact.shape[0] else math.nan
    uv_contact = uv_contact[finite]
    if uv_contact.size == 0:
        nan = math.nan
        return ContactUvStats(
            contact_px=contact_px,
            finite_ratio=finite_ratio,
            motion_ratio=nan,
            u_p50=nan,
            uv_p50=nan,
            spike_ratio=nan,
            spike_boundary_ratio=0.0,
        )

    u = uv_contact[:, 0]
    mag = np.linalg.norm(uv_contact, axis=1)
    motion_ratio = float(np.mean(mag >= float(motion_eps_mm))) if mag.size else math.nan

    u_p50 = _percentile(u, 50)
    uv_p50 = _percentile(mag, 50)

    # Spike stats computed on full grid to preserve boundary positions.
    mag_grid = np.linalg.norm(uv, axis=2)
    finite_grid = np.isfinite(mag_grid)
    spike = contact & finite_grid & (mag_grid >= float(spike_min_mm))
    spike_count = int(np.sum(spike))
    spike_ratio = float(spike_count / contact_px) if contact_px > 0 else math.nan

    boundary = _contact_boundary_mask(contact_mask)
    spike_boundary_ratio = float(np.mean(boundary[spike])) if spike_count > 0 else 0.0

    return ContactUvStats(
        contact_px=contact_px,
        finite_ratio=finite_ratio,
        motion_ratio=motion_ratio,
        u_p50=u_p50,
        uv_p50=uv_p50,
        spike_ratio=spike_ratio,
        spike_boundary_ratio=spike_boundary_ratio,
    )


@dataclass(frozen=True)
class RunAgg:
    name: str
    save_dir: Path
    fill_holes: bool
    fill_holes_iters: int
    smooth: bool
    smooth_iters: int
    cmd: str
    frames: Tuple[int, ...]
    phases: Tuple[str, ...]
    uv_finite_ratio_avg: float
    uv_motion_ratio_avg: float
    u_p50_avg: float
    uv_p50_avg: float
    uv_spike_ratio_avg: float
    uv_spike_boundary_ratio_avg: float
    height_grad_p99_raw_avg: float
    height_grad_p99_boxblur_avg: float


def _get_scene_params(manifest: Dict[str, Any]) -> Dict[str, Any]:
    sp = manifest.get("scene_params") or {}
    return sp if isinstance(sp, dict) else {}


def _diff_scene_params_except_surface(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    keys = set(a.keys()) | set(b.keys())
    diffs: List[str] = []
    for k in sorted(keys):
        if k in SURFACE_KEYS:
            continue
        if a.get(k) != b.get(k):
            diffs.append(k)
    return diffs


def _phase_list(manifest: Dict[str, Any]) -> List[str]:
    traj = manifest.get("trajectory") or {}
    if not isinstance(traj, dict):
        return []
    ft = traj.get("frame_to_phase") or []
    if not isinstance(ft, list):
        return []
    return [str(x or "") for x in ft]


def _load_npz(save_dir: Path, frame_id: int) -> Dict[str, np.ndarray]:
    p = save_dir / "intermediate" / f"frame_{int(frame_id):04d}.npz"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return dict(np.load(p))


def _summarize_one(
    save_dir: Path,
    *,
    frames: Sequence[int],
    motion_eps_mm: float,
    spike_min_mm: float,
) -> RunAgg:
    manifest_path = save_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    manifest = _read_json(manifest_path)
    sp = _get_scene_params(manifest)
    phase_list = _phase_list(manifest)

    fill_holes = bool(sp.get("mpm_height_fill_holes", False))
    fill_iters = int(sp.get("mpm_height_fill_holes_iters", 10))
    smooth = bool(sp.get("mpm_height_smooth", True))
    smooth_iters = int(sp.get("mpm_height_smooth_iters", 2))
    cmd = _join_argv(manifest.get("argv"))

    uv_finite: List[float] = []
    uv_motion: List[float] = []
    u_p50: List[float] = []
    uv_p50: List[float] = []
    spike_ratio: List[float] = []
    spike_boundary_ratio: List[float] = []
    grad_raw: List[float] = []
    grad_blur: List[float] = []
    phases: List[str] = []

    for f in frames:
        npz = _load_npz(save_dir, int(f))
        if "uv_disp_mm" not in npz or "contact_mask" not in npz or "height_field_mm" not in npz:
            raise KeyError(f"Missing required arrays in intermediate frame_{int(f):04d}.npz under {save_dir}")
        uv_stats = _compute_contact_uv_stats(
            npz["uv_disp_mm"].astype(np.float32, copy=False),
            npz["contact_mask"].astype(np.uint8, copy=False),
            motion_eps_mm=float(motion_eps_mm),
            spike_min_mm=float(spike_min_mm),
        )

        height = npz["height_field_mm"].astype(np.float32, copy=False)
        grad_raw.append(_height_grad_p99(height))
        grad_blur.append(_height_grad_p99(_box_blur_2d(height, iterations=smooth_iters)))

        uv_finite.append(uv_stats.finite_ratio)
        uv_motion.append(uv_stats.motion_ratio)
        u_p50.append(uv_stats.u_p50)
        uv_p50.append(uv_stats.uv_p50)
        spike_ratio.append(uv_stats.spike_ratio)
        spike_boundary_ratio.append(uv_stats.spike_boundary_ratio)

        phases.append(phase_list[int(f)] if 0 <= int(f) < len(phase_list) else "")

    return RunAgg(
        name=save_dir.name,
        save_dir=save_dir,
        fill_holes=fill_holes,
        fill_holes_iters=fill_iters,
        smooth=smooth,
        smooth_iters=smooth_iters,
        cmd=cmd,
        frames=tuple(int(x) for x in frames),
        phases=tuple(phases),
        uv_finite_ratio_avg=_mean(uv_finite),
        uv_motion_ratio_avg=_mean(uv_motion),
        u_p50_avg=_mean(u_p50),
        uv_p50_avg=_mean(uv_p50),
        uv_spike_ratio_avg=_mean(spike_ratio),
        uv_spike_boundary_ratio_avg=_mean(spike_boundary_ratio),
        height_grad_p99_raw_avg=_mean(grad_raw),
        height_grad_p99_boxblur_avg=_mean(grad_blur),
    )


@dataclass(frozen=True)
class ComboRow:
    fill_holes: bool
    fill_holes_iters: int
    smooth: bool
    smooth_iters: int
    derived: bool
    source: Path
    uv_finite_ratio_avg: float
    uv_motion_ratio_avg: float
    uv_spike_ratio_avg: float
    uv_spike_boundary_ratio_avg: float
    height_grad_p99_avg: float


def _make_combo_row(
    run: RunAgg,
    *,
    fill_holes: bool,
    smooth: bool,
    derived: bool,
) -> ComboRow:
    grad = run.height_grad_p99_boxblur_avg if bool(smooth) else run.height_grad_p99_raw_avg
    return ComboRow(
        fill_holes=bool(fill_holes),
        fill_holes_iters=int(run.fill_holes_iters),
        smooth=bool(smooth),
        smooth_iters=int(run.smooth_iters),
        derived=bool(derived),
        source=run.save_dir,
        uv_finite_ratio_avg=float(run.uv_finite_ratio_avg),
        uv_motion_ratio_avg=float(run.uv_motion_ratio_avg),
        uv_spike_ratio_avg=float(run.uv_spike_ratio_avg),
        uv_spike_boundary_ratio_avg=float(run.uv_spike_boundary_ratio_avg),
        height_grad_p99_avg=float(grad),
    )


def _write_report(
    out_path: Path,
    *,
    runs: Sequence[RunAgg],
    frame_slots: Sequence[str],
    motion_eps_mm: float,
    spike_min_mm: float,
    combos: Sequence[ComboRow],
    recommended: ComboRow,
) -> None:
    src_lines: List[str] = []
    for r in runs:
        cmd_line = f"  - cmd: `{r.cmd}`" if r.cmd else ""
        src_lines.extend(
            [
                f"- `{r.save_dir.as_posix()}`",
                f"  - surface: fill_holes=`{r.fill_holes}` (iters=`{r.fill_holes_iters}`), smooth=`{r.smooth}` (iters=`{r.smooth_iters}`)",
                f"  - frames_used: `{','.join(str(int(x)) for x in r.frames)}` phases: `{','.join(r.phases)}`",
                cmd_line,
            ]
        )
        if cmd_line:
            pass
        else:
            src_lines.pop()  # remove empty line

    def _combo_note(c: ComboRow) -> str:
        return "derived" if c.derived else "actual"

    table_lines = [
        "| fill_holes | smooth | note | finite_ratio_avg | motion_ratio_avg | spike_ratio_avg | spike_boundary_ratio_avg | grad_p99_avg | source |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in combos:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    f"{str(bool(c.fill_holes)).lower()} (iters={int(c.fill_holes_iters)})",
                    f"{str(bool(c.smooth)).lower()} (iters={int(c.smooth_iters)})",
                    _combo_note(c),
                    _format_float(c.uv_finite_ratio_avg),
                    _format_float(c.uv_motion_ratio_avg),
                    _format_float(c.uv_spike_ratio_avg),
                    _format_float(c.uv_spike_boundary_ratio_avg),
                    _format_float(c.height_grad_p99_avg),
                    f"`{c.source.as_posix()}`",
                ]
            )
            + " |"
        )

    lines: List[str] = [
        "# Report: Surface Fields Post-Processing Ablation (fill_holes / smooth)",
        "",
        "本报告基于已保存的 `intermediate/frame_*.npz` 离线统计：",
        "",
        "- `finite_ratio_avg/motion_ratio_avg/spike_ratio_avg/spike_boundary_ratio_avg`：来自接触区 `uv_disp_mm`（coverage/尖峰）。",
        "- `grad_p99_avg`：对 `height_field_mm` 的梯度幅值 p99；当 `smooth=true` 时，先按 `mpm_height_smooth_iters` 做 3x3 box blur 再计算（用于近似 halo 风险）。",
        "",
        "## Inputs",
        "",
        f"- frame_slots: `{','.join(str(s) for s in frame_slots)}`",
        f"- motion_eps_mm: `{motion_eps_mm}`",
        f"- spike_min_mm: `{spike_min_mm}`",
        "",
        "### Source runs",
        *src_lines,
        "",
        "## Ablation table (>=4 combinations)",
        "",
        *table_lines,
        "",
        "## Recommendation",
        "",
        f"- recommended: fill_holes=`{recommended.fill_holes}` (iters=`{recommended.fill_holes_iters}`), smooth=`{recommended.smooth}` (iters=`{recommended.smooth_iters}`)",
        "- basis: prefer lower spike_ratio/boundary_ratio (uv stability) + lower grad_p99 (halo proxy), while keeping finite_ratio high.",
        "",
        "## Notes",
        "",
        "- `smooth` 属于渲染前的高度场平滑，不会改变 `intermediate/height_field_mm` 的导出内容；因此这里采用“离线 box blur 后再统计 grad_p99”的方式评估其对 halo 的影响。",
        "- 若需要人工对照帧图，可在对应 `save_dir` 下抽样查看 `mpm_*.png` 与表中指标是否一致。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ablate surface fields post-processing knobs (fill holes / smooth) from existing rgb_compare outputs."
    )
    parser.add_argument(
        "--save-dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more rgb_compare output directories (must contain run_manifest.json and intermediate/frame_*.npz).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Report_surface_fields_ablation.md",
        help="Output Markdown report path (default: Report_surface_fields_ablation.md).",
    )
    parser.add_argument("--frames", type=str, default=None, help="Optional comma-separated explicit frame ids (default: auto)")
    parser.add_argument("--motion-eps-mm", type=float, default=0.01, help="Motion threshold (mm) used for motion_ratio")
    parser.add_argument("--spike-min-mm", type=float, default=0.20, help="Spike threshold (mm) for spike_ratio")

    args = parser.parse_args(argv)
    save_dirs = [Path(p) for p in args.save_dirs]
    for p in save_dirs:
        if not p.exists():
            return _fail(f"--save-dirs not found: {p}")
        if not (p / "run_manifest.json").exists():
            return _fail(f"Missing run_manifest.json: {p}")
        if not (p / "intermediate").exists():
            return _fail(f"Missing intermediate dir: {p}")

    # Validate that only surface field knobs differ across runs (keep physics fixed).
    manifests = [_read_json(p / "run_manifest.json") for p in save_dirs]
    sps = [_get_scene_params(m) for m in manifests]
    base_sp = sps[0]
    for i in range(1, len(sps)):
        diffs = _diff_scene_params_except_surface(base_sp, sps[i])
        if diffs:
            sample = ", ".join(diffs[:12])
            more = "" if len(diffs) <= 12 else f" (+{len(diffs) - 12} more)"
            return _fail(
                "Runs are not comparable: scene_params differ beyond fill/smooth knobs. "
                f"diff_keys(sample)=[{sample}]{more}. "
                "Please select runs where only mpm_height_fill_holes/_iters and/or mpm_height_smooth/_iters differ."
            )

    # Frames: by default align by trajectory phases (press_end/slide_mid/hold_end), so runs may use different frame ids.
    requested_frames: Optional[List[int]] = None
    if args.frames is not None and str(args.frames).strip():
        try:
            requested_frames = sorted({int(x.strip()) for x in str(args.frames).split(",") if x.strip()})
        except Exception:
            return _fail(f"Invalid --frames: {args.frames}")

    runs: List[RunAgg] = []
    for p, manifest in zip(save_dirs, manifests):
        if requested_frames is not None:
            frames = requested_frames
        else:
            try:
                frames = _pick_phase_aligned_frames(manifest)
            except Exception as e:
                # Backward-compatible fallback: pick from available common frames if phase metadata is missing.
                available = _list_intermediate_frames(p / "intermediate")
                if not available:
                    return _fail(f"No intermediate frames found under: {p / 'intermediate'}")
                frames = _pick_default_frames(available)
                print(f"[WARN] phase-aligned frame pick failed for {p.name}: {e}; fallback frames={frames}", flush=True)
        try:
            runs.append(
                _summarize_one(
                    p,
                    frames=frames,
                    motion_eps_mm=float(args.motion_eps_mm),
                    spike_min_mm=float(args.spike_min_mm),
                )
            )
        except Exception as e:
            return _fail(f"Failed to summarize {p}: {e}")

    by_combo: Dict[Tuple[bool, bool], RunAgg] = {}
    by_fill: Dict[bool, RunAgg] = {}
    for r in runs:
        by_combo.setdefault((bool(r.fill_holes), bool(r.smooth)), r)
        by_fill.setdefault(bool(r.fill_holes), r)

    if True not in by_fill or False not in by_fill:
        return _fail("Need at least one run with fill_holes=true and one run with fill_holes=false to build 2x2 ablation.")

    combos: List[ComboRow] = []
    for fill in [True, False]:
        src = by_fill[bool(fill)]
        for smooth in [True, False]:
            actual = by_combo.get((bool(fill), bool(smooth)))
            if actual is not None:
                combos.append(_make_combo_row(actual, fill_holes=bool(fill), smooth=bool(smooth), derived=False))
            else:
                combos.append(_make_combo_row(src, fill_holes=bool(fill), smooth=bool(smooth), derived=True))

    # Simple recommendation heuristic:
    # - prefer fill_holes=true if it doesn't increase spike_ratio (<= 1.05x) and finite_ratio is not worse.
    # - prefer smooth=true if it reduces grad_p99 (<= 0.95x) for the chosen fill_holes value.
    fill_on = next(c for c in combos if c.fill_holes and c.smooth)
    fill_off = next(c for c in combos if (not c.fill_holes) and c.smooth)
    prefer_fill = True
    if (
        math.isfinite(fill_on.uv_spike_ratio_avg)
        and math.isfinite(fill_off.uv_spike_ratio_avg)
        and fill_on.uv_spike_ratio_avg > fill_off.uv_spike_ratio_avg * 1.05
    ):
        prefer_fill = False
    if (
        math.isfinite(fill_on.uv_finite_ratio_avg)
        and math.isfinite(fill_off.uv_finite_ratio_avg)
        and fill_on.uv_finite_ratio_avg + 1e-6 < fill_off.uv_finite_ratio_avg
    ):
        prefer_fill = False

    chosen_fill = bool(prefer_fill)
    c_on = next(c for c in combos if c.fill_holes == chosen_fill and c.smooth)
    c_off = next(c for c in combos if c.fill_holes == chosen_fill and (not c.smooth))
    prefer_smooth = True
    if (
        math.isfinite(c_on.height_grad_p99_avg)
        and math.isfinite(c_off.height_grad_p99_avg)
        and c_on.height_grad_p99_avg > c_off.height_grad_p99_avg * 0.95
    ):
        prefer_smooth = False

    recommended = next(c for c in combos if c.fill_holes == chosen_fill and c.smooth == bool(prefer_smooth))

    out_path = Path(args.out)
    try:
        _write_report(
            out_path,
            runs=runs,
            frame_slots=("press_end", "slide_mid", "hold_end") if requested_frames is None else ("custom",),
            motion_eps_mm=float(args.motion_eps_mm),
            spike_min_mm=float(args.spike_min_mm),
            combos=combos,
            recommended=recommended,
        )
    except Exception as e:
        return _fail(f"Failed to write report: {e}")

    print(f"OK: wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
