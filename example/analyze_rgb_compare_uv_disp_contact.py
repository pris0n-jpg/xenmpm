"""
Analyze uv_disp_mm in the contact region for selected frames.

This script is dependency-light (numpy only) and is intended to run on the
outputs produced by `example/mpm_fem_rgb_compare.py --export-intermediate`.

It writes:
- A per-frame CSV summary (coverage / percentiles / spikes).
- An optional Markdown note with a brief interpretation for "u_p50≈0".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    return 1


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


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


def _pick_default_frames(available_frames: Sequence[int]) -> List[int]:
    # 约定优先：baseline 关键帧 75/80/85；若缺失则退化为“后段 3 帧”等间隔采样。
    want = [75, 80, 85]
    available = set(int(x) for x in available_frames)
    if all(x in available for x in want):
        return want
    if len(available_frames) <= 3:
        return list(available_frames)
    tail = list(available_frames)[max(0, len(available_frames) - 15) :]
    if len(tail) < 3:
        tail = list(available_frames)
    idxs = np.linspace(0, len(tail) - 1, num=3, dtype=int).tolist()
    return [tail[i] for i in sorted(set(int(x) for x in idxs))]


def _contact_boundary_mask(contact_mask: np.ndarray) -> np.ndarray:
    # 为什么：尖峰若集中在接触边界，通常更像后处理/边界伪影而非“整体滑移”。
    # 约束：仅用 numpy；边缘视作“非接触”。
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


def _edge_mask(h: int, w: int, *, edge_width: int) -> np.ndarray:
    ew = max(0, int(edge_width))
    if ew <= 0:
        return np.zeros((h, w), dtype=np.bool_)
    m = np.zeros((h, w), dtype=np.bool_)
    m[:ew, :] = True
    m[-ew:, :] = True
    m[:, :ew] = True
    m[:, -ew:] = True
    return m


@dataclass(frozen=True)
class ContactUvSummary:
    contact_px: int
    finite_ratio: float
    motion_ratio: float
    u_p10: float
    u_p50: float
    u_p90: float
    u_p99: float
    u_min: float
    u_max: float
    uv_p10: float
    uv_p50: float
    uv_p90: float
    uv_p99: float
    uv_max: float
    spike_count: int
    spike_ratio: float
    spike_edge_ratio: float
    spike_boundary_ratio: float
    top_spikes: str


def _compute_contact_uv_summary(
    uv_disp_mm: np.ndarray,
    contact_mask: np.ndarray,
    *,
    motion_eps_mm: float,
    spike_min_mm: float,
    spike_topk: int,
    edge_width: int,
) -> ContactUvSummary:
    if uv_disp_mm.ndim != 3 or uv_disp_mm.shape[-1] != 2:
        raise ValueError(f"uv_disp_mm must be HxWx2, got shape={uv_disp_mm.shape}")
    if contact_mask.ndim != 2 or contact_mask.shape != uv_disp_mm.shape[:2]:
        raise ValueError(f"contact_mask must be HxW, got shape={contact_mask.shape} vs uv={uv_disp_mm.shape[:2]}")

    h, w = int(uv_disp_mm.shape[0]), int(uv_disp_mm.shape[1])
    contact = (contact_mask > 0).astype(np.bool_)
    contact_px = int(np.sum(contact))
    if contact_px <= 0:
        nan = math.nan
        return ContactUvSummary(
            contact_px=0,
            finite_ratio=nan,
            motion_ratio=nan,
            u_p10=nan,
            u_p50=nan,
            u_p90=nan,
            u_p99=nan,
            u_min=nan,
            u_max=nan,
            uv_p10=nan,
            uv_p50=nan,
            uv_p90=nan,
            uv_p99=nan,
            uv_max=nan,
            spike_count=0,
            spike_ratio=nan,
            spike_edge_ratio=nan,
            spike_boundary_ratio=nan,
            top_spikes="",
        )

    uv = uv_disp_mm.astype(np.float32, copy=False)
    uv_contact = uv[contact]  # (N, 2)
    finite = np.isfinite(uv_contact).all(axis=1)
    finite_ratio = float(np.mean(finite)) if uv_contact.shape[0] else math.nan
    uv_contact = uv_contact[finite]
    if uv_contact.size == 0:
        nan = math.nan
        return ContactUvSummary(
            contact_px=contact_px,
            finite_ratio=finite_ratio,
            motion_ratio=nan,
            u_p10=nan,
            u_p50=nan,
            u_p90=nan,
            u_p99=nan,
            u_min=nan,
            u_max=nan,
            uv_p10=nan,
            uv_p50=nan,
            uv_p90=nan,
            uv_p99=nan,
            uv_max=nan,
            spike_count=0,
            spike_ratio=nan,
            spike_edge_ratio=nan,
            spike_boundary_ratio=nan,
            top_spikes="",
        )

    u = uv_contact[:, 0]
    mag = np.linalg.norm(uv_contact, axis=1)
    motion_ratio = float(np.mean(mag >= float(motion_eps_mm))) if mag.size else math.nan

    u_p10, u_p50, u_p90, u_p99 = _percentiles(u, [10, 50, 90, 99])
    u_min = float(np.min(u)) if u.size else math.nan
    u_max = float(np.max(u)) if u.size else math.nan

    uv_p10, uv_p50, uv_p90, uv_p99 = _percentiles(mag, [10, 50, 90, 99])
    uv_max = float(np.max(mag)) if mag.size else math.nan

    # Spike stats computed on full grid to preserve (y, x) coordinates.
    mag_grid = np.linalg.norm(uv, axis=2)
    finite_grid = np.isfinite(mag_grid)
    spike = contact & finite_grid & (mag_grid >= float(spike_min_mm))
    spike_count = int(np.sum(spike))
    spike_ratio = float(spike_count / contact_px) if contact_px > 0 else math.nan

    edge = _edge_mask(h, w, edge_width=int(edge_width))
    boundary = _contact_boundary_mask(contact_mask)
    spike_edge_ratio = float(np.mean(edge[spike])) if spike_count > 0 else 0.0
    spike_boundary_ratio = float(np.mean(boundary[spike])) if spike_count > 0 else 0.0

    # Top-K spikes: y,x,mag_mm (descending). Use argpartition for efficiency.
    top_spikes = ""
    if spike_topk > 0 and spike_count > 0:
        ys, xs = np.where(spike)
        mags = mag_grid[ys, xs].astype(np.float64, copy=False)
        k = min(int(spike_topk), int(mags.size))
        idx = np.argpartition(-mags, kth=k - 1)[:k]
        idx = idx[np.argsort(-mags[idx])]
        parts: List[str] = []
        for i in idx.tolist():
            parts.append(f"{int(ys[i])},{int(xs[i])},{_format_float(float(mags[i]))}")
        top_spikes = ";".join(parts)

    return ContactUvSummary(
        contact_px=contact_px,
        finite_ratio=finite_ratio,
        motion_ratio=motion_ratio,
        u_p10=u_p10,
        u_p50=u_p50,
        u_p90=u_p90,
        u_p99=u_p99,
        u_min=u_min,
        u_max=u_max,
        uv_p10=uv_p10,
        uv_p50=uv_p50,
        uv_p90=uv_p90,
        uv_p99=uv_p99,
        uv_max=uv_max,
        spike_count=spike_count,
        spike_ratio=spike_ratio,
        spike_edge_ratio=spike_edge_ratio,
        spike_boundary_ratio=spike_boundary_ratio,
        top_spikes=top_spikes,
    )


def _write_markdown(
    path: Path,
    *,
    save_dir: Path,
    frames: Sequence[int],
    motion_eps_mm: float,
    spike_min_mm: float,
    edge_width: int,
    rows: Sequence[Tuple[int, str, ContactUvSummary]],
) -> None:
    lines: List[str] = [
        "# RGB Compare UV Disp Contact Diagnostics",
        "",
        f"- save_dir: `{save_dir.as_posix()}`",
        f"- frames: `{','.join(str(int(x)) for x in frames)}`",
        "",
        "## Params",
        f"- motion_eps_mm: `{motion_eps_mm}`",
        f"- spike_min_mm: `{spike_min_mm}`",
        f"- edge_width: `{edge_width}`",
        "",
        "## Per-frame summary",
        "",
    ]
    for frame_id, phase, s in rows:
        lines.extend(
            [
                f"### frame_{frame_id:04d} ({phase})",
                f"- contact_px: `{s.contact_px}`",
                f"- finite_ratio: `{_format_float(s.finite_ratio)}`",
                f"- motion_ratio(|uv|>=eps): `{_format_float(s.motion_ratio)}`",
                f"- u_p50_mm: `{_format_float(s.u_p50)}` (p10/p90=`{_format_float(s.u_p10)}`/`{_format_float(s.u_p90)}`)",
                f"- |uv|_p50_mm: `{_format_float(s.uv_p50)}` (p90/max=`{_format_float(s.uv_p90)}`/`{_format_float(s.uv_max)}`)",
                f"- spikes(|uv|>=min): `{s.spike_count}` (ratio=`{_format_float(s.spike_ratio)}`)",
                f"- spikes_edge_ratio: `{_format_float(s.spike_edge_ratio)}`",
                f"- spikes_boundary_ratio: `{_format_float(s.spike_boundary_ratio)}`",
                "",
            ]
        )
        if s.top_spikes:
            lines.append("- top_spikes(y,x,|uv|_mm):")
            for part in s.top_spikes.split(";"):
                lines.append(f"  - `{part}`")
            lines.append("")

    lines.extend(
        [
            "## Interpretation (why u_p50≈0)",
            "",
            "若 `u_p50≈0` 同时伴随 `motion_ratio(|uv|>=eps)` 很低，说明：在接触区超过 50% 的格点",
            "面内位移幅值都低于阈值 eps，因此中位数会自然接近 0；而“少量抽风”通常对应于 spikes（少数极大值）。",
            "",
            "下一步可用以下信号区分更可能的来源：",
            "",
            "- spikes_edge_ratio 高：更像边缘/warp 出界伪影；",
            "- spikes_boundary_ratio 高：更像接触边界处的提取/填洞/平滑副作用；",
            "- 二者都不高但 motion_ratio 仍低：更像物理 stick（材料/接触/摩擦）导致大面积接触区不滑移。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze uv_disp_mm in contact region for selected frames (numpy-only).")
    parser.add_argument("--save-dir", type=str, required=True, help="Output directory created by mpm_fem_rgb_compare.py --save-dir")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: <save-dir>/uv_disp_contact_stats.csv)")
    parser.add_argument("--out-md", type=str, default=None, help="Output Markdown path (default: <save-dir>/uv_disp_contact_diagnostics.md)")
    parser.add_argument("--no-md", action="store_true", default=False, help="Do not write the Markdown diagnostics note")
    parser.add_argument("--frames", type=str, default=None, help="Comma-separated explicit frame ids (default: 75,80,85 if available)")
    parser.add_argument("--motion-eps-mm", type=float, default=0.01, help="Motion threshold (mm) used for coverage ratio")
    parser.add_argument("--spike-min-mm", type=float, default=0.20, help="Spike threshold (mm) for counting / top-K")
    parser.add_argument("--spike-topk", type=int, default=10, help="Top-K spikes recorded as 'y,x,|uv|_mm' (default: 10)")
    parser.add_argument("--edge-width", type=int, default=3, help="Edge width (pixels/cells) for spikes_edge_ratio")

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
    available_frames = _list_intermediate_frames(intermediate_dir)
    if not available_frames:
        return _fail(f"No intermediate frames found under: {intermediate_dir}")
    available_set = set(available_frames)

    requested = _parse_frames_arg(args.frames)
    frames = requested if requested is not None else _pick_default_frames(available_frames)
    frames = [int(f) for f in frames if int(f) in available_set]
    if not frames:
        return _fail(f"No requested frames found. Available (sample): {available_frames[:10]} ...")

    out_csv = Path(args.out) if args.out else (save_dir / "uv_disp_contact_stats.csv")
    out_md = Path(args.out_md) if args.out_md else (save_dir / "uv_disp_contact_diagnostics.md")

    summaries: List[Tuple[int, str, ContactUvSummary]] = []
    for frame_id in frames:
        npz_path = intermediate_dir / f"frame_{int(frame_id):04d}.npz"
        if not npz_path.exists():
            return _fail(f"Missing intermediate frame: {npz_path}")
        data = np.load(npz_path)
        if "contact_mask" not in data or "uv_disp_mm" not in data:
            return _fail(f"Intermediate missing required arrays: {npz_path}")
        contact_mask = data["contact_mask"].astype(np.uint8, copy=False)
        uv_disp_mm = data["uv_disp_mm"].astype(np.float32, copy=False)
        phase = phase_list[int(frame_id)] if 0 <= int(frame_id) < len(phase_list) else ""
        try:
            summary = _compute_contact_uv_summary(
                uv_disp_mm,
                contact_mask,
                motion_eps_mm=float(args.motion_eps_mm),
                spike_min_mm=float(args.spike_min_mm),
                spike_topk=int(args.spike_topk),
                edge_width=int(args.edge_width),
            )
        except Exception as e:
            return _fail(f"Failed to analyze frame_{int(frame_id):04d}: {e}")
        summaries.append((int(frame_id), str(phase), summary))

    fieldnames = [
        "frame_id",
        "phase",
        "contact_px",
        "finite_ratio",
        "motion_ratio",
        "u_p10_mm",
        "u_p50_mm",
        "u_p90_mm",
        "u_p99_mm",
        "u_min_mm",
        "u_max_mm",
        "uv_p10_mm",
        "uv_p50_mm",
        "uv_p90_mm",
        "uv_p99_mm",
        "uv_max_mm",
        "spike_count",
        "spike_ratio",
        "spike_edge_ratio",
        "spike_boundary_ratio",
        "top_spikes_yx_uvmm",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for frame_id, phase, s in summaries:
            w.writerow(
                {
                    "frame_id": str(int(frame_id)),
                    "phase": str(phase),
                    "contact_px": str(int(s.contact_px)),
                    "finite_ratio": _format_float(s.finite_ratio),
                    "motion_ratio": _format_float(s.motion_ratio),
                    "u_p10_mm": _format_float(s.u_p10),
                    "u_p50_mm": _format_float(s.u_p50),
                    "u_p90_mm": _format_float(s.u_p90),
                    "u_p99_mm": _format_float(s.u_p99),
                    "u_min_mm": _format_float(s.u_min),
                    "u_max_mm": _format_float(s.u_max),
                    "uv_p10_mm": _format_float(s.uv_p10),
                    "uv_p50_mm": _format_float(s.uv_p50),
                    "uv_p90_mm": _format_float(s.uv_p90),
                    "uv_p99_mm": _format_float(s.uv_p99),
                    "uv_max_mm": _format_float(s.uv_max),
                    "spike_count": str(int(s.spike_count)),
                    "spike_ratio": _format_float(s.spike_ratio),
                    "spike_edge_ratio": _format_float(s.spike_edge_ratio),
                    "spike_boundary_ratio": _format_float(s.spike_boundary_ratio),
                    "top_spikes_yx_uvmm": str(s.top_spikes),
                }
            )

    if not args.no_md:
        _write_markdown(
            out_md,
            save_dir=save_dir,
            frames=frames,
            motion_eps_mm=float(args.motion_eps_mm),
            spike_min_mm=float(args.spike_min_mm),
            edge_width=int(args.edge_width),
            rows=summaries,
        )

    rel_csv = out_csv.relative_to(save_dir) if out_csv.is_relative_to(save_dir) else out_csv
    print(f"OK: wrote {rel_csv}", flush=True)
    if not args.no_md:
        rel_md = out_md.relative_to(save_dir) if out_md.is_relative_to(save_dir) else out_md
        print(f"OK: wrote {rel_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

