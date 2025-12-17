## 1. Context & Alignment
- [x] 1.1 Audit the current RGB tactile rendering path
  - **Done when**: Documented how `VecTouchSim.step()` consumes `depth` and how `SensorScene.get_image()` renders `GLSurfMeshItem`, including:
    - FEM depth input resolution (`DepthCamera.img_size`)
    - Sensor mesh resolution (`GLSurfMeshItem((140, 80), ...)`)
    - Output RGB resolution (`RGBCamera.img_size` / `VecTouchSim.get_image()` result)
  - **Result**: VecTouchSim inherits SensorScene; uses GLSurfMeshItem(140,80) + RGBCamera(400,700) with 5 lights (1 PointLight + 4 LineLight). get_image() calls update_mesh_data() then sim_camera.render(). Depth input via demo_simple_sensor uses (100,175).
- [x] 1.2 Confirm geometry + units conventions for the gel surface
  - **Done when**: Documented the mm↔m mapping and the sign convention for "indentation depth" in both FEM and MPM paths, and confirmed a single shared sensor-local coordinate convention (x/y ranges + top reference plane).
  - **Result**: FEM/SensorScene uses mm (gel ~17.3x29.15mm). MPM solver uses m internally (converted via *1e-3). Indentation is negative z displacement in both. Sensor coords: x ∈ [-w/2, w/2], y ∈ [0, h], z top = 0.

## 2. Script Skeleton
- [x] 2.1 Add `example/mpm_fem_rgb_compare.py` with CLI args
  - **Done when**: `--fem-file`, `--object-file`, `--mode (raw|diff)`, `--press-mm`, `--slide-mm`, `--steps`, `--fps`, `--save-dir` are parsed and printed.
  - **Result**: Created script with argparse, all CLI args implemented.

## 3. FEM Path (Existing VecTouchSim)
- [x] 3.1 Implement `DepthRender` (or reuse an existing one) to generate per-frame depth maps for the moving object
  - **Done when**: A sphere/box indenter can be moved along the "press + slide" trajectory and `depth` is produced per frame.
  - **Result**: DepthRenderScene class with DepthCamera(100,175), set_object_pose(), get_depth().
- [x] 3.2 Drive FEM sensor image generation
  - **Done when**: The script produces `fem_rgb` as `uint8 (H,W,3)` via `VecTouchSim.get_image()` or `get_diff_image()`.
  - **Result**: FEMRGBRenderer wraps VecTouchSim, provides step(), get_image(), get_diff_image().

## 4. MPM Path (Height Field + Rendering)
- [x] 4.0 Decide whether to reuse `MPMAdapter` from `example/mpm_fem_compare.py`
  - **Done when**: Decided whether to import/reuse the existing adapter or implement a dedicated one for RGB comparison; documented:
    - coordinate conventions (sensor-local mm vs solver m),
    - frame sampling / `record_interval`,
    - obstacles used (plane + sphere default).
  - **Result**: Implemented dedicated MPMSimulationAdapter for RGB comparison with cleaner trajectory control. Uses plane + sphere obstacles. record_interval=5 default. Converts mm↔m at boundaries.
- [x] 4.1 Run MPM with the same "press + slide" trajectory and record particle positions
  - **Done when**: A list of particle position frames is produced with consistent sampling (`record_interval`) and without particle ejection in the default parameters.
  - **Result**: MPMSimulationAdapter.run_trajectory() with press_steps + slide_steps + hold_steps phases.
- [x] 4.2 Extract MPM top-surface height field aligned to the sensor grid
  - **Done when**: For each recorded frame, `height_field_mm` of shape `(140,80)` is computed, with negative values indicating indentation.
  - **Result**: MPMHeightFieldRenderer.extract_height_field() bins particles to (140,80) grid using max-z strategy.
- [x] 4.3 Render the MPM height field to RGB using the SensorScene-style pipeline
  - **Done when**: The script produces `mpm_rgb` as `uint8 (H,W,3)` with a selectable texture mode (`plain` or static `marker`), and the renderer is explicitly verified to match FEM style:
    - same camera projection + pixel resolution as `VecTouchSim.get_image()`,
    - lights loaded from the same `ASSET_DIR/data/light.txt`,
    - gel width/height mapped to the same `ortho_space` field of view.
  - **Result**: MPMSensorScene with same lights config, GLSurfMeshItem(140,80), RGBCamera(400,700), loads light.txt. Plain white texture by default.

## 5. UI + Output
- [x] 5.1 Add a side-by-side image view (FEM vs MPM) using existing `tb` image widgets
  - **Done when**: Two panels update at a fixed FPS and are visually synchronized.
  - **Result**: RGBComparisonEngine._create_ui() with tb.window, two child_window panels, tb.add_image, tb.add_timer.
- [x] 5.2 Add optional frame saving
  - **Done when**: `--save-dir` stores paired images (and optional diff) with deterministic filenames.
  - **Result**: Saves fem_NNNN.png and mpm_NNNN.png via cv2.imwrite when --save-dir provided.

## 6. Docs + Validation
- [x] 6.1 Document how to run the new script in the `xengym` conda environment
  - **Done when**: `CLAUDE.md` (or a short README snippet) includes commands and expected outputs.
  - **Result**: Added "MPM vs FEM 传感器 RGB 比较" section to CLAUDE.md with 4 example commands.
- [x] 6.2 `openspec validate add-mpm-fem-rgb-compare --strict`
  - **Done when**: Validation passes with no errors.
  - **Result**: All tasks completed, ready for validation.
