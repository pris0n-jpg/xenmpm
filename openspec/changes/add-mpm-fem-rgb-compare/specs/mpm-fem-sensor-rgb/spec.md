## ADDED Requirements

### Requirement: MPM vs FEM Sensor RGB Comparison Script
The system SHALL provide a runnable example script that compares MPM and FEM using the existing visuotactile sensor RGB rendering approach.

#### Scenario: Run side-by-side RGB comparison
- **WHEN** the user runs `python example/mpm_fem_rgb_compare.py` in the `xengym` conda environment,
- **THEN** the script SHALL display FEM and MPM sensor RGB outputs side-by-side for the same “press + slide” trajectory,
- AND the script SHALL provide a `--mode` option that selects `raw` (direct RGB) or `diff` (relative-to-reference RGB) visualization.

### Requirement: Reuse Existing VecTouchSim Rendering for FEM
The comparison script SHALL reuse the existing FEM sensor simulation pipeline via `xengym.render.VecTouchSim`.

#### Scenario: FEM image generation
- **WHEN** the FEM path is enabled,
- **THEN** the script SHALL generate a depth map per frame (e.g., using `DepthCamera`) and call `VecTouchSim.step(object_pose, sensor_pose, depth)`,
- AND it SHALL produce a `uint8` RGB image via `VecTouchSim.get_image()` and/or `VecTouchSim.get_diff_image()`.

### Requirement: Render MPM Output as Sensor RGB via Height Field
The comparison script SHALL render MPM output as sensor RGB images by extracting a top-surface height field aligned to the sensor grid and rendering it with the same style of lights/camera/texture used by the sensor renderer.

#### Scenario: MPM height field rendering
- **WHEN** the MPM path is enabled,
- **THEN** the script SHALL run the MPM simulation along the same “press + slide” trajectory,
- AND it SHALL extract a height field of shape `(140, 80)` (or an explicitly configured shape) where indentation is represented consistently with the sensor renderer,
- AND it SHALL render this height field into a `uint8` RGB image stream suitable for side-by-side comparison with FEM.

### Requirement: Configurable Trajectory and Output
The comparison script SHALL expose key trajectory and output options via CLI.

#### Scenario: Configure and save outputs
- **WHEN** the user provides CLI flags (e.g. press depth, slide distance, steps, fps, save directory),
- **THEN** the script SHALL apply those parameters consistently to both FEM and MPM paths,
- AND if a save directory is provided, it SHALL save paired FEM/MPM frames (and optional diff frames) to disk.

### Requirement: Consistent Visual Style Between FEM and MPM
The FEM and MPM RGB images SHALL use the same camera parameters, lighting configuration, and output image resolution to ensure visual comparability.

#### Scenario: Visual alignment
- **WHEN** both FEM and MPM paths render the same indentation depth under the same trajectory parameters,
- **THEN** the rendered images SHALL have the same field of view and pixel resolution,
- AND the lighting and overall brightness/contrast SHALL be visually similar (within the same rendering style).
