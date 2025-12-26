"""
Lightweight smoke tests for the xenmpm workspace.

This file is intentionally dependency-light: it does not require Taichi or ezgl.
Run:
    python quick_test.py
"""

from __future__ import annotations

import py_compile
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np


def _fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    return 1


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "example" / "mpm_fem_rgb_compare.py"
    if not script_path.exists():
        return _fail(f"Missing script: {script_path}")

    # 1) Syntax compile (no heavy imports)
    for path in [
        script_path,
        repo_root / "xengym" / "fem" / "simulation.py",
        repo_root / "xengym" / "render" / "sensorScene.py",
    ]:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as e:
            return _fail(f"py_compile failed: {path}: {e}")

    # 2) Load script definitions (will print warnings if optional deps missing; that's OK)
    module = runpy.run_path(str(script_path))
    required = [
        "SCENE_PARAMS",
        "_analyze_binary_stl_endfaces_mm",
        "_mpm_flip_x_field",
        "_mpm_flip_x_mm",
        "_compute_rgb_diff_metrics",
        "MPMSensorScene",
    ]
    for name in required:
        if name not in module:
            return _fail(f"Missing symbol in {script_path.name}: {name}")

    scene_params = module["SCENE_PARAMS"]
    for key in ["fem_fric_coef", "mpm_mu_s", "mpm_mu_k", "gel_size_mm"]:
        if key not in scene_params:
            return _fail(f"SCENE_PARAMS missing key: {key}")

    if float(scene_params["fem_fric_coef"]) != 0.4:
        return _fail(f"Unexpected fem_fric_coef default: {scene_params['fem_fric_coef']} (expected 0.4)")
    if float(scene_params["mpm_mu_s"]) != 2.0:
        return _fail(f"Unexpected mpm_mu_s default: {scene_params['mpm_mu_s']} (expected 2.0)")
    if float(scene_params["mpm_mu_k"]) != 1.5:
        return _fail(f"Unexpected mpm_mu_k default: {scene_params['mpm_mu_k']} (expected 1.5)")

    # 3) Coordinate flip invariants
    flip_field = module["_mpm_flip_x_field"]
    flip_x_mm = module["_mpm_flip_x_mm"]
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    flipped = flip_field(x)
    if flipped.tolist() != [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]:
        return _fail("Unexpected _mpm_flip_x_field result")
    if flip_x_mm(1.25) != -1.25:
        return _fail("Unexpected _mpm_flip_x_mm result")

    # 4) Metrics helper invariants
    metrics_fn = module["_compute_rgb_diff_metrics"]
    a = np.zeros((4, 5, 3), dtype=np.uint8)
    b = np.full((4, 5, 3), 10, dtype=np.uint8)
    metrics = metrics_fn(a, b)
    if float(metrics.get("mae", -1.0)) != 10.0 or float(metrics.get("max_abs", -1.0)) != 10.0:
        return _fail(f"Unexpected metrics output: {metrics}")

    # 5) STL end-face sanity check (circle_r4 has ~15mm base vs ~8mm tip)
    stl_path = repo_root / "xengym" / "assets" / "obj" / "circle_r4.STL"
    if not stl_path.exists():
        return _fail(f"Missing STL asset: {stl_path}")
    stl_stats = module["_analyze_binary_stl_endfaces_mm"](stl_path)
    if not stl_stats or "endfaces_mm" not in stl_stats:
        return _fail("STL analysis returned empty stats")
    ymin = stl_stats["endfaces_mm"]["y_min"]
    ymax = stl_stats["endfaces_mm"]["y_max"]
    if abs(float(ymin["size_x_mm"]) - 15.0) > 1.0 or abs(float(ymax["size_x_mm"]) - 8.0) > 1.0:
        return _fail(f"Unexpected STL endface sizes: y_min={ymin}, y_max={ymax}")

    # 6) CLI surface check (ensure key flags are present)
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for flag in ["--fric", "--mpm-depth-tint", "--export-intermediate"]:
        if flag not in help_text:
            return _fail(f"Missing CLI flag in --help output: {flag}")

    print("OK: quick_test", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

