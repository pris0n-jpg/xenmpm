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
        "DEFAULT_MPM_MARKER_MODE",
        "SCENE_PARAMS",
        "_analyze_binary_stl_endfaces_mm",
        "_mpm_flip_x_field",
        "_mpm_flip_x_mm",
        "_compute_rgb_diff_metrics",
        "_write_preflight_run_manifest",
        "MPMSensorScene",
        "RGBComparisonEngine",
    ]
    for name in required:
        if name not in module:
            return _fail(f"Missing symbol in {script_path.name}: {name}")

    if str(module.get("DEFAULT_MPM_MARKER_MODE") or "") != "warp":
        return _fail(f"Unexpected DEFAULT_MPM_MARKER_MODE: {module.get('DEFAULT_MPM_MARKER_MODE')} (expected 'warp')")

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
        # Avoid constructing full renderers (ezgl/GL context) here: we only
        # validate file-format invariants of the export helpers.
        engine = engine_cls.__new__(engine_cls)
        engine.save_dir = tmp_dir
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
    for flag in [
        "--fric",
        "--fem-marker",
        "--mpm-marker",
        "--mpm-depth-tint",
        "--mpm-height-fill-holes",
        "--mpm-height-fill-holes-iters",
        "--mpm-height-smooth",
        "--mpm-height-smooth-iters",
        "--mpm-height-reference-edge",
        "--mpm-height-clamp-indenter",
        "--mpm-height-clip-outliers",
        "--mpm-height-clip-outliers-min-mm",
        "--export-intermediate",
    ]:
        if flag not in help_text:
            return _fail(f"Missing CLI flag in --help output: {flag}")

    # 7) Preflight run manifest helper invariants (no UI / no blocking).
    write_preflight = module["_write_preflight_run_manifest"]
    default_conventions = {
        "mpm_height_field_flip_x": True,
        "mpm_uv_disp_flip_x": True,
        "mpm_uv_disp_u_negate": False,
        "mpm_warp_flip_x": True,
        "mpm_warp_flip_y": True,
        "mpm_overlay_flip_x_mm": True,
        "mpm_height_fill_holes": False,
        "mpm_height_fill_holes_iters": 10,
        "mpm_height_smooth": True,
        "mpm_height_smooth_iters": 2,
        "mpm_height_reference_edge": True,
        "mpm_height_clamp_indenter": True,
        "mpm_height_clip_outliers": False,
        "mpm_height_clip_outliers_min_mm": 5.0,
    }

    def _read_manifest(manifest_path: Path) -> Dict[str, object]:
        if not manifest_path.exists():
            raise FileNotFoundError(str(manifest_path))
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _assert_manifest_common(manifest: Dict[str, object]) -> None:
        for key in ["trajectory", "scene_params", "deps", "run_context"]:
            if key not in manifest:
                raise ValueError(f"run_manifest.json missing key: {key}")
        traj = manifest.get("trajectory") or {}
        total_frames = int(traj.get("total_frames") or 0)
        if total_frames <= 0:
            raise ValueError(f"Unexpected total_frames in run_manifest.json: {traj.get('total_frames')}")
        if len(traj.get("frame_to_phase") or []) != total_frames:
            raise ValueError("run_manifest.json frame_to_phase length mismatch total_frames")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        run_context = {
            "args": {
                # Ensure save_dir does not leak into run_manifest.json (argv already records it).
                "save_dir": str(tmp_dir),
                "mpm_show_indenter": True,
            },
            "resolved": {
                "conventions": dict(default_conventions),
                "render": {
                    "fem_marker": "on",
                    "mpm_marker": "warp",
                    "mpm_depth_tint": True,
                },
                "friction": {
                    "fem_fric_coef": 0.4,
                    "mpm_mu_s": 0.4,
                    "mpm_mu_k": 0.4,
                    "aligned": True,
                },
                "indenter": {
                    "fem": {"geom": "stl", "face": "tip", "contact_face_key": "y_max"},
                    "mpm": {"type": "box"},
                },
            }
        }
        try:
            write_preflight(
                tmp_dir,
                record_interval=5,
                total_frames=3,
                run_context=run_context,
                reason="quick_test",
            )
        except Exception as e:
            return _fail(f"_write_preflight_run_manifest failed: {e}")

        try:
            manifest = _read_manifest(tmp_dir / "run_manifest.json")
            _assert_manifest_common(manifest)
        except Exception as e:
            return _fail(f"Invalid run_manifest.json (default): {e}")

        if not (tmp_dir / "tuning_notes.md").exists():
            return _fail("Missing tuning_notes.md after _write_preflight_run_manifest")

        args_out = ((manifest.get("run_context") or {}).get("args") or {})
        if isinstance(args_out, dict):
            if "save_dir" in args_out:
                return _fail(f"run_manifest.json should not contain args.save_dir: {args_out}")
            if args_out.get("mpm_show_indenter") is not True:
                return _fail(f"Unexpected run_manifest.json args.mpm_show_indenter: {args_out}")

        resolved = (manifest.get("run_context") or {}).get("resolved") or {}    
        render = resolved.get("render") or {}
        for key in ["mpm_marker", "mpm_depth_tint", "fem_marker"]:
            if key not in render:
                return _fail(f"run_manifest.json missing resolved.render.{key}")
        conv = resolved.get("conventions") or {}
        for key, expected in default_conventions.items():
            if key not in conv:
                return _fail(f"run_manifest.json missing conventions.{key}")
            if conv.get(key) != expected:
                return _fail(f"Unexpected conventions.{key}: {conv}")

        friction = resolved.get("friction") or {}
        if friction.get("aligned") is not True:
            return _fail(f"Expected resolved.friction.aligned=true, got: {friction}")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        run_context = {
            "resolved": {
                "conventions": {
                    **dict(default_conventions),
                    "mpm_height_fill_holes": True,
                    "mpm_height_fill_holes_iters": 7,
                    "mpm_height_smooth": False,
                    "mpm_height_smooth_iters": 0,
                },
                "render": {
                    "fem_marker": "off",
                    "mpm_marker": "off",
                    "mpm_depth_tint": False,
                },
                "friction": {
                    "fem_fric_coef": 0.4,
                    "mpm_mu_s": 0.4,
                    "mpm_mu_k": 0.4,
                    "aligned": True,
                },
            }
        }
        try:
            write_preflight(
                tmp_dir,
                record_interval=5,
                total_frames=3,
                run_context=run_context,
                reason="quick_test-variant",
            )
        except Exception as e:
            return _fail(f"_write_preflight_run_manifest failed (variant): {e}")

        try:
            manifest = _read_manifest(tmp_dir / "run_manifest.json")
            _assert_manifest_common(manifest)
        except Exception as e:
            return _fail(f"Invalid run_manifest.json (variant): {e}")

        resolved = (manifest.get("run_context") or {}).get("resolved") or {}
        conv = resolved.get("conventions") or {}
        if conv.get("mpm_height_fill_holes") is not True:
            return _fail(f"Unexpected conventions.mpm_height_fill_holes: {conv}")
        if int(conv.get("mpm_height_fill_holes_iters") or 0) != 7:
            return _fail(f"Unexpected conventions.mpm_height_fill_holes_iters: {conv}")
        if conv.get("mpm_height_smooth") is not False:
            return _fail(f"Unexpected conventions.mpm_height_smooth: {conv}")
        try:
            if int(conv.get("mpm_height_smooth_iters")) != 0:
                return _fail(f"Unexpected conventions.mpm_height_smooth_iters: {conv}")
        except Exception:
            return _fail(f"Unexpected conventions.mpm_height_smooth_iters: {conv}")

        render = resolved.get("render") or {}
        if str(render.get("fem_marker") or "") != "off":
            return _fail(f"Unexpected render.fem_marker: {render}")
        if str(render.get("mpm_marker") or "") != "off":
            return _fail(f"Unexpected render.mpm_marker: {render}")
        if render.get("mpm_depth_tint") is not False:
            return _fail(f"Unexpected render.mpm_depth_tint: {render}")

    # 8) Offline mirror/flip sentinel fixture (no simulation, numpy+PIL only).
    fixture_dir = repo_root / "example" / "testdata" / "mirror_sentinel_noflip"
    sentinel_script = repo_root / "example" / "analyze_rgb_compare_flip_alignment.py"
    if not fixture_dir.exists() or not sentinel_script.exists():
        return _fail("mirror sentinel fixture missing (need example/testdata/mirror_sentinel_noflip)")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        out_csv = Path(tmp_dir_str) / "alignment_flip.csv"
        proc = subprocess.run(
            [
                sys.executable,
                str(sentinel_script),
                "--save-dir",
                str(fixture_dir),
                "--sample",
                "5",
                "--require-mpm-vs-fem",
                "direct",
                "--require-uv-best",
                "noflip",
                "--min-pass-ratio",
                "0.8",
                "--min-known-frames",
                "2",
                "--out",
                str(out_csv),
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            if proc.stdout:
                print(proc.stdout, flush=True)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, flush=True)
            return _fail("mirror sentinel fixture failed (see output above)")

    print("OK: quick_test", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
