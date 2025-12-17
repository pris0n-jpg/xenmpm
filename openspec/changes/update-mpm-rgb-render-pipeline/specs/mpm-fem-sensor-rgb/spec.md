## ADDED Requirements

### Requirement: MPM Marker Layer MUST Follow Tangential Motion
When rendering MPM sensor RGB, the marker layer that represents printed markers on the sensor surface SHALL move consistently with the simulated surface tangential displacement during press and slip.

#### Scenario: Marker translates during slip
- **WHEN** the MPM trajectory enters the “slide” phase with non-zero tangential motion,
- **THEN** the rendered MPM marker pattern SHALL translate/warp in the direction consistent with the surface tangential displacement field,
- AND the translation/warp magnitude SHALL scale monotonically with the configured slide distance (all else equal).

#### Scenario: Marker remains stable without tangential motion
- **WHEN** the MPM trajectory is purely normal indentation (no slide distance, or slide phase disabled),
- **THEN** the marker pattern SHALL remain visually stable (no systematic translation), aside from lighting changes due to normal deformation.

### Requirement: MPM RGB Renderer SHALL Expose Marker Mode Controls
The MPM RGB rendering pipeline SHALL expose user-facing controls to select how markers are rendered.

#### Scenario: Select marker rendering mode
- **WHEN** the user selects a marker mode via CLI (e.g., `off`, `static`, `warp`),
- **THEN** the MPM rendering SHALL apply the selected mode without affecting the FEM path,
- AND the default mode SHALL preserve current behavior for backward compatibility.

### Requirement: MPM RGB Renderer SHALL Visualize Indenter Pose (Optional)
The MPM RGB rendering pipeline SHALL provide an optional visualization of the indenter pose used by the MPM simulation to make motion and alignment diagnosable.

#### Scenario: Indenter overlay enabled
- **WHEN** the user enables MPM indenter visualization,
- **THEN** the MPM view SHALL include an overlay (2D or 3D) that moves consistently with the obstacle/indenter trajectory,
- AND the overlay SHALL remain aligned to the same sensor field-of-view used for the RGB rendering.

### Requirement: MPM RGB Renderer SHALL Provide Debug Overlays
The MPM RGB rendering pipeline SHALL provide optional debug overlays to verify coordinate alignment and displacement-field correctness.

#### Scenario: Enable uv/warp debug overlay
- **WHEN** the user enables a debug overlay mode (e.g., showing uv-displacement magnitude or warp offsets),
- **THEN** the MPM view SHALL overlay the diagnostic visualization on top of the RGB image,
- AND the overlay SHALL not change the underlying height-field deformation rendering.
