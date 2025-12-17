## Context

Existing components:
- FEM side:
  - `example/demo_main.py`, `example/demo_simple_sensor.py`, and `example/data_collection.py` already load FEM data (e.g. `fem_data_gel_2035.npz`) and drive the sensor scene.
  - The `render/` subpackage contains sensor and scene utilities used by these examples.
- MPM side:
  - `xengym/mpm` contains the VHE-MLS-MPM solver, configuration, CLI, and validation scripts.
  - MPM currently has its own CLI entry point and validation scenes, but no direct link to the main sensor + FEM visualization pipeline.

User need:
- A single “MPM vs FEM” scene for a small block sliding on the sensor, with:
  - 3D visual comparison (toggle or side-by-side).
  - Simple slice-based curve comparison (e.g. tangential force vs displacement).

## Goals / Non-Goals

Goals:
- Reuse as much of the existing **FEM visualization and sensor scene scaffolding** as possible.
- Add a new example that:
  - Uses FEM data exactly as in existing scripts.
  - Creates an MPM configuration that approximates the same scenario (block size, material stiffness, contact).
  - Provides a minimal toggle between FEM and MPM views in the same scene (e.g. keyboard key or command-line choice).
  - Produces a simple numeric comparison curve for a chosen observable (e.g. average block displacement vs time, or tangential force vs displacement).

Non-Goals:
- No large architectural refactors of `render/` or FEM code.
- No generic framework to “register arbitrary solvers” into the UI; this is a single, targeted comparison script.

## High-Level Design

### 1. New Example Script: `example/mpm_fem_compare.py`

Responsibilities:
- Parse command-line arguments:
  - `--fem-file`: path to FEM NPZ file (default to existing gel FEM data).
  - `--mode`: `"fem"`, `"mpm"`, or `"both"` (to control which simulation(s) to run).
  - Optional toggles for plotting vs interactive only.
- FEM path:
  - Reuse existing logic from `demo_main.py` / `demo_simple_sensor.py` to:
    - Load FEM displacement/stress data from `fem_file`.
    - Configure and spawn the sensor scene renderer.
    - Play back FEM deformation of the block on the sensor.
- MPM path:
  - Construct an `MPMConfig` approximating the same block-on-sensor setup:
    - Grid size and `dx` chosen so the block has similar physical dimensions to the FEM mesh.
    - Material parameters (Ogden/Maxwell) chosen for a reasonably similar stiffness (exact matching is out of scope).
    - Contact parameters configured to mimic the sensor contact in a simplified way.
  - Run the MPM simulation for a fixed number of steps and:
    - Record particle positions (and optionally stress/energy) over time.
    - Map particle data to a renderable object in the sensor scene (e.g. using a mesh or point cloud overlay).
- Visualization:
  - Use the same sensor scene (`render/`) as the FEM path.
  - Provide a simple toggle (e.g. keyboard key or initial `--mode`) to choose:
    - FEM-only playback.
    - MPM-only playback.
    - Optionally alternate or overlay modes if feasible with minimal changes.

### 2. Numerical Comparison (Curves)

- Define a simple metric extracted consistently from both FEM and MPM data, e.g.:
  - Average tangential displacement of the block’s contact surface vs time.
  - Or average normal displacement vs tangential friction force.
- For FEM:
  - Use existing FEM NPZ fields (positions/forces) to compute the chosen metric per frame.
- For MPM:
  - Use particle positions and contact/force information (or approximations) to compute the same metric per frame.
- Plotting:
  - Use Matplotlib to generate a 2D plot with both curves:
    - X-axis: displacement or time.
    - Y-axis: force or stress proxy.
  - Save the plot and/or show it interactively when the example exits.

### 3. Data Flow and Integration

- FEM data flow (existing):
  - NPZ → `example/demo_main.py` → render scene.
- MPM data flow (new):
  - `MPMConfig` + MPM solver → particle positions (and optionally stress/energy) → adapter → render scene.
- The new script will:
  - Instantiate/render the sensor scene once.
  - Provide helper functions to:
    - “Attach” a FEM-driven block visual.
    - “Attach” an MPM-driven block visual.
  - The toggle logic will decide which one is currently active.

## Risks / Trade-offs

- **Risk:** MPM and FEM may not visually match well with a simple parameter guess.
  - Mitigation: Document that the example is qualitative and uses approximate parameter matching; include a note in the script and/or README.
- **Risk:** Reusing the existing render pipeline might require some glue code to map MPM particles to the expected mesh format.
  - Mitigation: Start with the simplest possible visual (e.g. point cloud or coarse proxy mesh), and only add complexity if needed.
- **Risk:** Running both FEM and MPM in the same script may be heavy for some machines.
  - Mitigation: Offer `--mode fem` and `--mode mpm` options to run them separately if desired.

## Migration Plan

1. Audit existing FEM and sensor scene examples (`example/demo_main.py`, `example/demo_simple_sensor.py`, `example/data_collection.py`) to identify reusable entry points.
2. Design a minimal adapter from MPM particle data to the sensor scene visual representation.
3. Implement `example/mpm_fem_compare.py` using the existing FEM path and the new MPM path.
4. Add simple Matplotlib-based curve plotting for the chosen metric.
5. Test the example in the `xengym` conda environment (CPU-only is acceptable; GPU if available).

## Open Questions

- Final choice of metric (force–displacement vs displacement–time) and whether to expose it as a CLI option.
- Whether to support saving per-frame snapshots (images) in addition to real-time viewing.

