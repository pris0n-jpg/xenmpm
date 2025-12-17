## Context

项目现有视触觉仿真输出 RGB 图像的链路如下：

- `example/demo_simple_sensor.py`
  - `DepthCamera.render()` 获取物体到传感器参考面的深度图（正值/负值含义由 `actual_depth` 与 `ortho_space` 决定）。
  - 将 `depth`、`object_pose`、`sensor_pose` 输入 `xengym.render.VecTouchSim.step(...)`。
  - 调用 `VecTouchSim.get_image()` 或 `VecTouchSim.get_diff_image()` 得到 RGB（uint8）图像。
- `xengym/render/sensorScene.py`
  - 通过 `GLSurfMeshItem` 渲染高度场曲面，使用固定光照与 `RGBCamera` 正交投影得到 RGB 图像（`sim_camera.render()`）。
  - `get_image()` 会先 `update_mesh_data()` 再渲染。

本变更希望把 **MPM 输出** 也映射到同样的“高度场 → RGB”渲染方式，从而得到与 FEM 输出可比的图像。

## Goals / Non-Goals

Goals:
- 复用 `VecTouchSim` 的渲染风格（光照/相机/纹理/mesh 分辨率），让 MPM 与 FEM 在图像层面可比。
- 同一轨迹驱动 FEM 与 MPM（按压 + 滑移），并在同一帧率下并排展示。
- 允许用户快速调参对齐（压头半径、按压深度、滑移距离、帧采样间隔等）。

Non-Goals:
- 不实现“marker 随面内拉伸/剪切的精确纹理变形”（需要额外的 2D 位移场或 UV warp）。

## Proposed Architecture

### 1) 新增脚本：`example/mpm_fem_rgb_compare.py`

脚本由三部分组成：

1. FEM 渲染路径（现有方案复用）：
   - `DepthRender`：使用 `DepthCamera` + `GLModelItem` 渲染深度图（参考 `example/demo_simple_sensor.py`）。
   - `VecTouchSim`：`step(object_pose, sensor_pose, depth)` + `get_image/get_diff_image` 输出 FEM RGB。

2. MPM 求解与高度场提取：
   - 使用 `xengym.mpm.MPMSolver` + `ContactConfig.obstacles`（plane + sphere/box）做按压与滑移。
   - 记录粒子位置序列（或直接逐帧推进并输出）。
   - 从粒子位置得到顶面高度场 `H`，形状对齐 `SensorScene` 的深度网格分辨率（默认 `(140, 80)`）。
     - 说明：该 `(140, 80)` 是 `xengym/render/sensorScene.py` 中 `GLSurfMeshItem((140, 80), ...)` 的网格分辨率；
       与 `DepthCamera.img_size`（例如 `demo_simple_sensor.py` 中的 `(100, 175)`）是两个概念，互不要求相等。

3. MPM 高度场渲染为 RGB：
   - 用一个“最小渲染器”在脚本内创建独立的 `Scene(visible=False)`，复用 `SensorScene` 的核心渲染部件与参数：
     - lights：通过 `Scene.loadLight(str(ASSET_DIR/"data/light.txt"))` 加载与传感器一致的光照配置；
     - mesh：`GLSurfMeshItem((140, 80), x_range, y_range)`（与 `SensorScene` 相同的尺寸与范围）；
     - camera：`RGBCamera` 使用与 `SensorScene` 一致的正交投影范围与分辨率，确保 FEM/MPM 图像视野与像素对齐；
     - texture 选项：
       - `marker`：静态 marker 纹理（不随拉伸变形，先满足可视对比）。
       - `plain`：白色纹理，仅看光照形变。
   - `render(height_field_mm)` → RGB（uint8），输出分辨率与 FEM `VecTouchSim.get_image()` 保持一致。

### 2) 坐标与单位约定

- `VecTouchSim` / `SensorScene` 内部以 **mm** 定义胶体尺寸与 `GLSurfMeshItem` 的 `x_range/y_range`。
- MPM solver 内部以 **m** 为单位（`grid.dx`、粒子位置等）。

为了与现有 MPM 示例（`example/mpm_fem_compare.py`）保持一致、减少踩坑，本变更采用如下约定：

- “传感器局部坐标系（mm）”（用于高度场与渲染）：
  - x ∈ [-w/2, w/2] mm
  - y ∈ [0, h] mm
  - 顶面参考高度为 `z_top0`，高度场输出使用 `H = z_surface - z_top0`（因此压入时 `H <= 0`，与 `SensorScene` 一致）。
- “MPM 内部坐标系（m）”（用于 solver）：
  - 初始化块体采用 `z ∈ [0, t]`（顶面在 `z=t`），与现有实现一致；
  - 进入 solver 前整体平移到正域（避免越界），并记录该平移 `shift_m`；
  - 从 solver 回到“传感器局部坐标系”时，需要应用 `x_mm = (x_m - shift_m) * 1000`（同理 y/z）。

备注：
- `DepthCamera.img_size` 决定 FEM 输入深度图的采样分辨率（例如 `(100,175)`），而 MPM→RGB 渲染使用的是高度场网格分辨率（默认 `(140,80)`）；
  两者只要在物理范围（gel width/height）一致即可。

### 3) 高度场提取（MPM → Height Field）

默认选择“稳定优先”的离散化：

- 设 height grid 分辨率为 `(Ny, Nx)=(140,80)`。
- 对每个 cell（i,j）：
  - 选取落在该 cell 的粒子集合 `P_ij`（用 x,y binning）。
  - 取该 cell 的顶面高度 `z_ij = max(z for p in P_ij)`（对空 cell 取邻域补全或用初始顶面高度）。
- 最终高度场：
  - `H = z_ij - z_top0`（顶面相对位移，单位 mm，通常 <= 0）
  - 传给 `GLSurfMeshItem.setData(depth, smooth)` 时使用与 `SensorScene` 一致的符号约定（负值表示向下）。

可选增强（后续）：
- 使用核回归/MLS 对 `z(x,y)` 做平滑插值；
- 同时输出面内位移场，用于 marker 纹理 warp。

### 4) 同步策略

- 使用同一条轨迹参数驱动 FEM 与 MPM：
  - press：z 方向逐步压入到 `press_depth_mm`
  - slide：保持压入深度，同时 x 方向滑移 `slide_distance_mm`
- 在显示层对齐：
  - `record_interval` 决定 MPM 帧序列长度；
  - FEM 每帧用相同轨迹参数生成 depth map 并 `step` 一次；
  - UI 以固定 FPS 播放两路图像，确保“闪烁频率”一致。

## Trade-offs

- 选择静态 marker 纹理（不 warp）能保证最小落地，但会牺牲 marker 的物理一致性；短期用于对比形变 footprint 与整体响应仍有价值。
- 高度场离散化选择 `max z` 更稳定，但可能更“尖锐”；如需要更平滑，再引入插值（复杂度上升）。
