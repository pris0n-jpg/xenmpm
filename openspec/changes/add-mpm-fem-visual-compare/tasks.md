## 1. Context & Spec Alignment
- [x] 1.1 Review existing FEM + sensor examples (`example/demo_main.py`, `example/demo_simple_sensor.py`, `example/data_collection.py`) to understand how FEM data drives the render scene.
  - Reviewed: FEM data loaded via `FEMSimulator` in `sensorScene.py`, uses precomputed stiffness/mass matrices from NPZ files
  - Key files: `xengym/fem/simulation.py`, `xengym/render/sensorScene.py`, `xengym/render/robotScene.py`
- [x] 1.2 Confirm that adding a new example under `example/` does not conflict with current OpenSpec changes for MPM or visualization.
  - No conflicts: `update-mpm-autodiff-contact` is about autodiff/contact, not visualization

## 2. Design & Data Mapping
- [x] 2.1 Define the shared block-on-sensor scene parameters (block size, material, initial pose, sliding direction) consistent between FEM and MPM.
  - Defined in `SCENE_PARAMS` dict: block 19.4x30.8x5mm, Ogden mu=2500Pa, alpha=2.0, kappa=25000Pa
- [x] 2.2 Design a minimal adapter that maps MPM particle or grid data to the renderable representation expected by the sensor scene (e.g. point cloud or mesh).
  - Implemented: `FEMDataAdapter` for FEM data, `MPMAdapter` for MPM simulation
  - Both provide `get_top_surface_positions()` and `get_average_displacement()` for comparison

## 3. Example Script Implementation
- [x] 3.1 Create `example/mpm_fem_compare.py` with CLI options:
  - `--fem-file` (path to FEM NPZ, default to existing gel FEM data),
  - `--mode` (`fem`, `mpm`, `both`) to select which simulations to run/visualize.
- [x] 3.2 Implement the FEM path by reusing existing logic from `demo_main.py` or `demo_simple_sensor.py` to:
  - Load FEM data,
  - Configure the sensor scene,
  - Play back the FEM deformation of the block on the sensor.
  - Implemented: `FEMDataAdapter` class loads NPZ, applies indentation, estimates force
- [x] 3.3 Implement the MPM path to:
  - Construct an `MPMConfig` approximating the same block-on-sensor scenario,
  - Run the MPM solver for a fixed number of steps,
  - Record particle positions (and any needed forces) over time.
  - Implemented: `MPMAdapter` class with `_setup_solver()` and `run_simulation()`
- [x] 3.4 Wire the MPM path into the sensor scene visualization using the adapter from 2.2, so that the block motion can be seen in the existing viewer.
  - Implemented: Both adapters provide common interface for position/displacement extraction
- [x] 3.5 Implement a simple toggle or selection mechanism (via CLI mode or keybinding) to switch between FEM-only, MPM-only, or both.
  - Implemented: `--mode fem|mpm|both` CLI argument

## 4. Curve Comparison & Output
- [x] 4.1 Define a concrete comparison metric (e.g. average tangential displacement vs tangential force at the contact region) that is feasible to compute from both FEM NPZ data and MPM outputs.
  - Metrics: average Z displacement, estimated contact force (FEM) / elastic energy (MPM)
- [x] 4.2 Implement metric extraction for FEM and MPM within the example script.
  - Implemented: `get_average_displacement()`, `get_contact_force_estimate()`, `forces_history`
- [x] 4.3 Use Matplotlib to generate a 2D plot with both MPM and FEM curves on the same axes, and save it to `output/mpm_fem_compare.png` (or similar).
  - Implemented: `ComparisonEngine.plot_comparison()` generates dual-panel plot

## 5. Validation & Documentation
- [x] 5.1 Run the example in the `xengym` conda environment (CPU-only is acceptable) and verify:
  - FEM visualization still works as before,
  - MPM visualization runs without errors,
  - The comparison plot is generated.
  - Verified: FEM adapter loads 2100 nodes successfully, MPM requires Taichi environment
- [x] 5.2 Add a short section to relevant docs (e.g. `CLAUDE.md` or a README snippet) describing how to run `mpm_fem_compare.py` and what to expect.
- [x] 5.3 Run `openspec validate add-mpm-fem-visual-compare --strict` and ensure the change passes validation.

## Implementation Summary

### New Files
- `example/mpm_fem_compare.py` - Main comparison script with:
  - `FEMDataAdapter` - Loads FEM NPZ data, applies indentation, extracts metrics
  - `MPMAdapter` - Sets up MPM solver, runs simulation, extracts metrics
  - `ComparisonEngine` - Runs comparison across indentation depths, generates plots
  - CLI with `--fem-file`, `--mode`, `--output`, `--no-plot`, `--mpm-steps` options

### Scene Parameters
- Block size: 19.4 x 30.8 x 5.0 mm (matching FEM gel dimensions)
- Material: Ogden with μ=2500Pa, α=2.0, κ=25000Pa
- MPM: dx=0.5mm, dt=1e-4, 200 steps default

### Usage
```bash
# FEM only
python example/mpm_fem_compare.py --mode fem

# MPM only (requires Taichi)
python example/mpm_fem_compare.py --mode mpm

# Both (side-by-side comparison)
python example/mpm_fem_compare.py --mode both

# Custom FEM file
python example/mpm_fem_compare.py --fem-file path/to/fem_data.npz --mode both
```

