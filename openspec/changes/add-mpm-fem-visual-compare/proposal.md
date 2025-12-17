## Summary

Add a new example in the `xengym` conda environment that visually and numerically compares the existing FEM simulation with the Taichi-based MPM solver on a **single shared scene**: a small soft block sliding over a sensor.

The example will:
- Reuse the existing IsaacGym/render pipeline (as used in `example/demo_main.py` / `demo_simple_sensor.py`) to show an interactive/animated 3D view.
- Run both FEM and MPM versions of the “block on sensor” scene and allow switching between them in the same viewer.
- Produce a basic numerical comparison:
  - Deformation shape / displacement field snapshots.
  - A simple section-based stress–strain (or force–displacement) curve for a chosen region.

This proposal only defines the spec/design; implementation will follow after approval.

## Motivation

Users working in the `xengym` conda environment want an **intuitive, visual** way to understand:
- How the new MPM solver behaves relative to the existing FEM simulation on the same physical scenario.
- Where MPM and FEM differ in deformation shape and stress/force response.

Currently:
- FEM is used via `example/demo_main.py` / `data_collection.py`, which drive the sensor scene based on precomputed FEM data (`fem_data_*.npz`).
- MPM is implemented in `xengym/mpm`, but there is no shared visual comparison scene directly linking MPM and FEM behaviour.

Providing a single side-by-side or toggleable visualization will:
- Help validate the MPM model against FEM intuition.
- Serve as a practical starting point for future calibration/validation work.

## Goals

- Add a **single, focused example script** under `example/` that:
  - Runs a “small block sliding on sensor” scene in both FEM and MPM.
  - Uses the existing render/IsaacGym scene graph to display both, with a simple toggle (e.g. keypress or command-line flag) between FEM and MPM.
  - Outputs at least one section-based curve (e.g. tangential force vs tangential displacement, or average displacement vs time) for both methods on the same plot.
- Keep the implementation as lightweight as possible:
  - Reuse existing FEM data and render scaffolding.
  - Avoid inventing an entirely new UI system.

## Non-Goals

- Do not build a full-blown GUI comparison tool with complex controls; minimal interactions (play/pause/toggle) are sufficient.
- Do not perform rigorous quantitative validation of MPM vs FEM; this example is primarily **visual and qualitative**, with simple supporting curves.
- Do not redesign the FEM or MPM solvers themselves; changes should be limited to wiring them into a common scene and exporting comparison data.

## Scope

In scope:
- A new example script under `example/` (e.g. `example/mpm_fem_compare.py`) which:
  - Loads existing FEM data (from `assets/data/fem_data_*.npz`) and drives the sensor scene as in `demo_main.py`.
  - Sets up and runs an equivalent MPM scene using `xengym.mpm`, with matching geometry/material/boundary conditions as closely as practical.
  - Connects both to a common visualization path (existing `render/` scene) and to simple Matplotlib plots.
- Small additions to existing capabilities/specs if needed (e.g. validation/visualization).

Out of scope:
- Major changes to the rendering system or new external dependencies beyond Matplotlib and what is already in the project.
- New FEM solvers or MPM material models.

## Open Questions

- How exactly to define the “section” or metric for the stress–strain / force–displacement curve:
  - Top-layer average displacement vs tangential force?
  - A line or patch within the contact region?
- How tightly MPM parameters (e.g. Ogden/Maxwell) should be tuned to mimic the existing FEM configuration for this example (likely we will keep a simple, robust default).  

