## ADDED Requirements

### Requirement: MPM vs FEM Visual Comparison Scene
The system SHALL provide an example that visually compares MPM and FEM on a single, shared physical scenario: a small block sliding on a sensor.

- The example SHALL:
  - Run both FEM and MPM versions of the scene (block on sensor) using a common geometry and approximate material parameters.
  - Use the existing sensor/render scene to display the block and sensor contact.
  - Allow the user to choose between FEM-only, MPM-only, or both (e.g. via command-line flag or toggle).

#### Scenario: Example script execution
- **WHEN** the user runs `python example/mpm_fem_compare.py` in the `xengym` conda environment,
- **THEN** the script SHALL:
  - Load FEM data from a configured NPZ (defaulting to an existing `fem_data_*.npz` file),
  - Construct and run an MPM simulation of a similarly sized block sliding on the sensor,
  - And present a 3D visualization where the user can view at least one of FEM or MPM deformation sequences.

### Requirement: Shared Sensor Scene Visualization
The visual comparison example SHALL reuse the existing sensor/render scene infrastructure (as used by `demo_main.py` / `demo_simple_sensor.py`) rather than introducing a completely new viewer.

- It MAY add minimal glue code or adapters to map MPM data into the existing rendering abstractions.

#### Scenario: Reused render pipeline
- **WHEN** the example script starts the visualization,
- **THEN** it SHALL instantiate the sensor scene via existing render/IsaacGym utilities,
- AND it SHALL render either FEM-driven or MPM-driven block motion in that scene, without creating a parallel, unrelated UI system.

### Requirement: Section-Based Curve Comparison
The example SHALL produce at least one simple numeric comparison between MPM and FEM for the block-on-sensor scene.

- This MAY be:
  - Tangential force vs tangential displacement at the block–sensor contact,
  - Or average displacement vs time for a chosen region.
- The comparison SHALL be output as a 2D plot (e.g. via Matplotlib) showing both curves on the same axes.

#### Scenario: Curve output
- **WHEN** the example simulation completes (FEM + MPM),
- **THEN** it SHALL compute the chosen metric for both methods over time and generate a plot file (e.g. PNG) and/or show the plot interactively,
- AND the plot SHALL contain clearly labelled curves distinguishing MPM from FEM.

