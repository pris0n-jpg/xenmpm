"""
Lightweight smoke tests for the xenmpm workspace.

This file is intentionally dependency-light: it does not require Taichi or ezgl.
Run:
    python quick_test.py
"""

from __future__ import annotations

import csv
import json
import py_compile
import runpy
import subprocess
import sys
import tempfile
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
        "RGBComparisonEngine",
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

    # 4b) Intermediate/metrics file invariants (dependency-light)
    engine_cls = module["RGBComparisonEngine"]
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        engine = engine_cls(
            fem_file="dummy.npz",
            object_file=None,
            mode="raw",
            visible=False,
            save_dir=str(tmp_dir),
        )
        height_field_mm = np.zeros((140, 80), dtype=np.float32)
        height_field_mm[0, 0] = -0.02
        uv_disp_mm = np.zeros((140, 80, 2), dtype=np.float32)
        engine._export_intermediate_frame(
            frame=0,
            mpm_height_field_mm=height_field_mm,
            mpm_uv_disp_mm=uv_disp_mm,
            fem_depth_mm=None,
            fem_marker_disp=None,
            fem_contact_mask_u8=None,
        )
        out_path = tmp_dir / "intermediate" / "frame_0000.npz"
        if not out_path.exists():
            return _fail("Missing intermediate/frame_0000.npz after export")
        try:
            loaded = np.load(out_path)
        except Exception as e:
            return _fail(f"Failed to load exported npz: {e}")
        for key in ["frame", "height_field_mm", "uv_disp_mm", "contact_mask"]:
            if key not in loaded:
                return _fail(f"Missing key in exported npz: {key}")
        if loaded["height_field_mm"].shape != (140, 80):
            return _fail(f"Unexpected height_field_mm shape: {loaded['height_field_mm'].shape}")
        if loaded["uv_disp_mm"].shape != (140, 80, 2):
            return _fail(f"Unexpected uv_disp_mm shape: {loaded['uv_disp_mm'].shape}")
        if loaded["contact_mask"].shape != (140, 80):
            return _fail(f"Unexpected contact_mask shape: {loaded['contact_mask'].shape}")
        if int(loaded["contact_mask"][0, 0]) != 1:
            return _fail("contact_mask[0,0] should be 1 for negative height")
        loaded.close()

        engine._metrics_rows = [
            {
                "frame": 0,
                "phase": "press",
                "mode": "raw",
                "mae": 1.0,
                "mae_r": 1.0,
                "mae_g": 1.0,
                "mae_b": 1.0,
                "max_abs": 2.0,
                "p50": 1.0,
                "p90": 2.0,
                "p99": 2.0,
            }
        ]
        engine._write_metrics_files()
        metrics_csv = tmp_dir / "metrics.csv"
        metrics_json = tmp_dir / "metrics.json"
        if not metrics_csv.exists():
            return _fail("Missing metrics.csv after _write_metrics_files")
        if not metrics_json.exists():
            return _fail("Missing metrics.json after _write_metrics_files")
        try:
            with metrics_csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            return _fail(f"Failed to parse metrics.csv: {e}")
        if len(rows) != 1 or str(rows[0].get("frame")) != "0":
            return _fail(f"Unexpected metrics.csv rows: {rows}")

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
    for flag in ["--fric", "--fem-marker", "--mpm-marker", "--mpm-depth-tint", "--export-intermediate"]:
        if flag not in help_text:
            return _fail(f"Missing CLI flag in --help output: {flag}")   

    # 7) Preflight run manifest should be written even without ezgl/taichi
    #    (keeps outputs auditable in dependency-light environments).
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--mode",
                "raw",
                "--record-interval",
                "5",
                "--save-dir",
                str(tmp_dir),
            ],
            capture_output=True,
            text=True,
        )
        # In dependency-light envs this typically returns non-zero; that's OK.
        _ = proc.returncode
        manifest_path = tmp_dir / "run_manifest.json"
        if not manifest_path.exists():
            return _fail("Missing run_manifest.json in preflight save-dir run")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            return _fail(f"Failed to parse run_manifest.json: {e}")
        for key in ["trajectory", "scene_params", "deps", "run_context"]:
            if key not in manifest:
                return _fail(f"run_manifest.json missing key: {key}")
        traj = manifest.get("trajectory") or {}
        total_frames = int(traj.get("total_frames") or 0)
        if total_frames <= 0:
            return _fail(f"Unexpected total_frames in run_manifest.json: {traj.get('total_frames')}")
        if len(traj.get("frame_to_phase") or []) != total_frames:
            return _fail("run_manifest.json frame_to_phase length mismatch total_frames")
        resolved = (manifest.get("run_context") or {}).get("resolved") or {}
        render = resolved.get("render") or {}
        for key in ["mpm_marker", "mpm_depth_tint", "fem_marker"]:
            if key not in render:
                return _fail(f"run_manifest.json missing resolved.render.{key}")
        conv = resolved.get("conventions") or {}
        for key in [
            "mpm_height_field_flip_x",
            "mpm_uv_disp_flip_x",
            "mpm_uv_disp_u_negate",
            "mpm_warp_flip_x",
            "mpm_warp_flip_y",
            "mpm_overlay_flip_x_mm",
        ]:
            if key not in conv:
                return _fail(f"run_manifest.json missing conventions.{key}")

    # 7c) Marker/tint toggles should be reflected in run_manifest.json.
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--mode",
                "raw",
                "--record-interval",
                "5",
                "--save-dir",
                str(tmp_dir),
                "--fem-marker",
                "off",
                "--mpm-marker",
                "off",
                "--mpm-depth-tint",
                "off",
            ],
            capture_output=True,
            text=True,
        )
        _ = proc.returncode
        manifest_path = tmp_dir / "run_manifest.json"
        if not manifest_path.exists():
            return _fail("Missing run_manifest.json in marker/tint preflight run")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            return _fail(f"Failed to parse run_manifest.json (marker/tint run): {e}")
        resolved = (manifest.get("run_context") or {}).get("resolved") or {}
        render = resolved.get("render") or {}
        if str(render.get("fem_marker") or "") != "off":
            return _fail(f"Unexpected render.fem_marker: {render}")
        if str(render.get("mpm_marker") or "") != "off":
            return _fail(f"Unexpected render.mpm_marker: {render}")
        if render.get("mpm_depth_tint") is not False:
            return _fail(f"Unexpected render.mpm_depth_tint: {render}")

    # 7b) --fric should align FEM fric_coef and MPM mu_s/mu_k and be
    #     reflected in run_manifest.json.
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--mode",
                "raw",
                "--record-interval",
                "5",
                "--save-dir",
                str(tmp_dir),
                "--fric",
                "0.4",
            ],
            capture_output=True,
            text=True,
        )
        _ = proc.returncode
        manifest_path = tmp_dir / "run_manifest.json"
        if not manifest_path.exists():
            return _fail("Missing run_manifest.json in --fric preflight run")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            return _fail(f"Failed to parse run_manifest.json (--fric run): {e}")
        resolved = (manifest.get("run_context") or {}).get("resolved") or {}
        friction = resolved.get("friction") or {}
        if friction.get("aligned") is not True:
            return _fail(f"Expected resolved.friction.aligned=true, got: {friction}")
        for key in ["fem_fric_coef", "mpm_mu_s", "mpm_mu_k"]:
            try:
                if abs(float(friction.get(key)) - 0.4) > 1e-6:
                    return _fail(f"Unexpected {key} in friction: {friction}")
            except Exception:
                return _fail(f"Invalid {key} in friction: {friction}")

    # 8) FEM STL tip/base selection should be auditable via resolved contact face size.
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--mode",
                "raw",
                "--record-interval",
                "5",
                "--save-dir",
                str(tmp_dir),
                "--fem-indenter-geom",
                "stl",
                "--fem-indenter-face",
                "tip",
            ],
            capture_output=True,
            text=True,
        )
        _ = proc.returncode
        manifest_path = tmp_dir / "run_manifest.json"
        if not manifest_path.exists():
            return _fail("Missing run_manifest.json for fem-indenter-face tip run")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            return _fail(f"Failed to parse run_manifest.json (tip run): {e}")
        resolved = (manifest.get("run_context") or {}).get("resolved") or {}
        fem = (resolved.get("indenter") or {}).get("fem") or {}
        face_key = str(fem.get("contact_face_key") or "")
        if face_key != "y_max":
            return _fail(f"Unexpected fem contact_face_key: {face_key} (expected y_max for tip)")
        face_size = fem.get("contact_face_size_mm") or {}
        try:
            size_x = float(face_size.get("size_x_mm") or 0.0)
        except Exception:
            size_x = 0.0
        if abs(size_x - 8.0) > 1.0:
            return _fail(f"Unexpected fem tip contact size_x_mm: {size_x} (expected ~8mm)")

    print("OK: quick_test", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
