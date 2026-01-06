"""
Summarize rgb_compare sweep runs from existing output directories.

This script is meant to be lightweight (numpy only) and focuses on extracting:
- key run parameters from run_manifest.json (scene_params / argv)
- contact-region uv_disp_mm statistics on key frames (u_p50, etc.)
- height_field_mm gradient statistic as a proxy for halo_risk (grad_p99)

It writes a Markdown report that can be committed as a “sweep record table”.
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


def _load_npz(save_dir: Path, frame_id: int) -> Dict[str, np.ndarray]:
    p = save_dir / "intermediate" / f"frame_{int(frame_id):04d}.npz"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return dict(np.load(p))


def _contact_uv_stats(npz: Dict[str, np.ndarray]) -> Tuple[float, float, float, float]:
    uv = npz["uv_disp_mm"].astype(np.float32, copy=False)
    contact = (npz["contact_mask"] > 0)
    u = uv[..., 0][contact]
    v = uv[..., 1][contact]
    mag = np.sqrt(u * u + v * v)
    return (
        _percentile(u, 50),
        _percentile(u, 90),
        _percentile(mag, 50),
        _percentile(mag, 90),
    )


def _height_grad_p99(npz: Dict[str, np.ndarray]) -> float:
    height = npz["height_field_mm"].astype(np.float32, copy=False)
    gy, gx = np.gradient(height)
    grad = np.sqrt(gx * gx + gy * gy)
    return _percentile(grad, 99)


@dataclass(frozen=True)
class SweepRow:
    name: str
    save_dir: Path
    press_mm: float
    slide_mm: float
    k_n: float
    k_t: float
    mu_s: float
    mu_k: float
    ogden_mu: str
    ogden_kappa: str
    u_p50_avg: float
    u_p90_avg: float
    uv_p50_avg: float
    uv_p90_avg: float
    grad_p99_avg: float
    cmd: str


def _iter_dirs(globs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pat in globs:
        for p in sorted(Path().glob(pat)):
            if p.is_dir():
                out.append(p)
    # Dedup while keeping order
    seen: set[str] = set()
    uniq: List[Path] = []
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _get_scene_params(manifest: Dict[str, Any]) -> Dict[str, Any]:
    scene_params = manifest.get("scene_params") or {}
    return scene_params if isinstance(scene_params, dict) else {}


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return math.nan


def _safe_list_str(value: object) -> str:
    if isinstance(value, list):
        return "[" + ",".join(str(x) for x in value) + "]"
    return str(value) if value is not None else ""


def _summarize_one(save_dir: Path, *, frames: Sequence[int]) -> SweepRow:
    manifest_path = save_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    manifest = _read_json(manifest_path)
    sp = _get_scene_params(manifest)

    press_mm = _safe_float(sp.get("press_depth_mm"))
    slide_mm = _safe_float(sp.get("slide_distance_mm"))
    k_n = _safe_float(sp.get("mpm_contact_stiffness_normal"))
    k_t = _safe_float(sp.get("mpm_contact_stiffness_tangent"))
    mu_s = _safe_float(sp.get("mpm_mu_s"))
    mu_k = _safe_float(sp.get("mpm_mu_k"))
    ogden_mu = _safe_list_str(sp.get("ogden_mu"))
    ogden_kappa = _format_float(_safe_float(sp.get("ogden_kappa")))
    cmd = _join_argv(manifest.get("argv"))

    u_p50s: List[float] = []
    u_p90s: List[float] = []
    uv_p50s: List[float] = []
    uv_p90s: List[float] = []
    grad_p99s: List[float] = []

    for f in frames:
        npz = _load_npz(save_dir, int(f))
        u_p50, u_p90, uv_p50, uv_p90 = _contact_uv_stats(npz)
        u_p50s.append(u_p50)
        u_p90s.append(u_p90)
        uv_p50s.append(uv_p50)
        uv_p90s.append(uv_p90)
        grad_p99s.append(_height_grad_p99(npz))

    return SweepRow(
        name=save_dir.name,
        save_dir=save_dir,
        press_mm=press_mm,
        slide_mm=slide_mm,
        k_n=k_n,
        k_t=k_t,
        mu_s=mu_s,
        mu_k=mu_k,
        ogden_mu=ogden_mu,
        ogden_kappa=ogden_kappa,
        u_p50_avg=_mean(u_p50s),
        u_p90_avg=_mean(u_p90s),
        uv_p50_avg=_mean(uv_p50s),
        uv_p90_avg=_mean(uv_p90s),
        grad_p99_avg=_mean(grad_p99s),
        cmd=cmd,
    )


def _md_table(rows: Sequence[SweepRow]) -> List[str]:
    lines = [
        "| name | press_mm | slide_mm | k_n | k_t | mu_s | mu_k | ogden_mu | ogden_kappa | u_p50_avg | u_p90_avg | |uv|_p50_avg | |uv|_p90_avg | grad_p99_avg | save_dir |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.name,
                    _format_float(r.press_mm),
                    _format_float(r.slide_mm),
                    _format_float(r.k_n),
                    _format_float(r.k_t),
                    _format_float(r.mu_s),
                    _format_float(r.mu_k),
                    r.ogden_mu,
                    r.ogden_kappa,
                    _format_float(r.u_p50_avg),
                    _format_float(r.u_p90_avg),
                    _format_float(r.uv_p50_avg),
                    _format_float(r.uv_p90_avg),
                    _format_float(r.grad_p99_avg),
                    r.save_dir.as_posix(),
                ]
            )
            + " |"
        )
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a sweep report from existing rgb_compare output directories (numpy-only).")
    parser.add_argument("--baseline-dir", type=str, default="output/rgb_compare/baseline", help="Baseline directory (default: output/rgb_compare/baseline)")
    parser.add_argument("--glob", type=str, action="append", default=["output/rgb_compare/tune_*"], help="Glob(s) of sweep dirs (default: output/rgb_compare/tune_*)")
    parser.add_argument("--frames", type=str, default="75,80,85", help="Frames used for summary (default: 75,80,85)")
    parser.add_argument("--out-md", type=str, default="Report_sweep_press1mm.md", help="Output Markdown path (default: Report_sweep_press1mm.md)")
    args = parser.parse_args(argv)

    frames = [int(x.strip()) for x in str(args.frames).split(",") if x.strip()]
    if not frames:
        return _fail("No frames specified")

    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.exists():
        return _fail(f"baseline_dir not found: {baseline_dir}")

    dirs = [baseline_dir] + _iter_dirs(args.glob)
    rows: List[SweepRow] = []
    for d in dirs:
        try:
            rows.append(_summarize_one(d, frames=frames))
        except Exception as e:
            print(f"WARN: skip {d}: {e}", file=sys.stderr, flush=True)

    if not rows:
        return _fail("No valid dirs found")

    # Baseline is the first row by construction.
    baseline = rows[0]
    candidates = [
        r
        for r in rows[1:]
        if math.isfinite(r.press_mm)
        and math.isfinite(r.slide_mm)
        and abs(r.press_mm - baseline.press_mm) < 1e-6
        and abs(r.slide_mm - baseline.slide_mm) < 1e-6
    ]
    candidates_sorted = sorted(candidates, key=lambda r: float(r.u_p50_avg), reverse=True)

    recommended: Optional[SweepRow] = None
    for r in candidates_sorted:
        # Avoid candidates that significantly worsen height gradient (proxy for halo risk).
        if math.isfinite(r.grad_p99_avg) and math.isfinite(baseline.grad_p99_avg) and r.grad_p99_avg > baseline.grad_p99_avg * 1.05:
            continue
        recommended = r
        break

    out_path = Path(args.out_md)
    lines: List[str] = [
        "# RGB Compare Sweep Report (press=1mm)",
        "",
        "本报告基于已存在的 `output/rgb_compare/*` 目录做离线汇总（无需 taichi/ezgl）。",
        "指标说明：",
        "",
        "- `u_p50_avg/u_p90_avg`：接触区 `uv_disp_mm[...,0]` 分位数（帧 75/80/85 平均）。",
        "- `|uv|_p50_avg/|uv|_p90_avg`：接触区位移幅值分位数（帧 75/80/85 平均）。",
        "- `grad_p99_avg`：`height_field_mm` 梯度幅值的 p99（帧 75/80/85 平均），作为 halo_risk 的代理量。",
        "",
        "## Summary Table",
        "",
        *_md_table([baseline] + candidates_sorted),
        "",
    ]

    if recommended is None:
        lines.extend(
            [
                "## Recommendation",
                "",
                "未在 press/slide 与 baseline 一致的目录中找到“u_p50 明显提升且 grad_p99 不更差”的组合；建议扩大 sweep 或补充其它参数维度（Ogden_mu/kappa）。",
                "",
            ]
        )
    else:
        delta = float(recommended.u_p50_avg) - float(baseline.u_p50_avg)
        lines.extend(
            [
                "## Recommendation",
                "",
                f"- baseline: `{baseline.save_dir.as_posix()}` u_p50_avg=`{_format_float(baseline.u_p50_avg)}` grad_p99_avg=`{_format_float(baseline.grad_p99_avg)}`",
                f"- recommended: `{recommended.save_dir.as_posix()}` u_p50_avg=`{_format_float(recommended.u_p50_avg)}` (Δ=`{_format_float(delta)}`) grad_p99_avg=`{_format_float(recommended.grad_p99_avg)}`",
                "",
                "复跑命令（来自 run_manifest.json argv）：",
                "",
                "```bash",
                recommended.cmd.strip(),
                "```",
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK: wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

