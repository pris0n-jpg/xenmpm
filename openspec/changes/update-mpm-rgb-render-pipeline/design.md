## 背景：当前 MPM RGB 渲染的“缺口”

现状（`example/mpm_fem_rgb_compare.py`）：

- MPM → RGB 采用 `height_field_mm (140x80)` 驱动 `GLSurfMeshItem` 的 zmap，得到 RGB。
- marker 是静态生成的纹理贴图（`Texture2D(marker_tex_np)`），每帧仅做深度着色 tint（不包含任何 x/y 位移）。
- MPM 求解器的压头作为 SDF obstacle 在物理里移动，但 RGB 渲染里未渲染压头几何体，因此用户会感知“压头不动”。

这导致：
- 滑移过程中 marker 不会移动/扭曲（违背“印在表面”的物理直觉）。
- 很难从 RGB 画面确认轨迹是否正确、坐标系是否对齐。

## 目标：以最小复杂度补齐“可比且可诊断”的 MPM RGB 管线

设计原则（遵循 guardrails）：
- 先做 **最小闭环**：高度场 + 面内位移场 + 纹理 warp + 压头可视化（可选）。
- 避免引入新大型依赖；优先复用现有 `numpy/cv2/xensesdk.ezgl`。
- 不修改 MPM 求解器；渲染侧只消费粒子状态与 obstacle 位姿。

## 提案架构（模块化但仍可留在脚本内）

### 1) 顶面运动学提取：`MPMSurfaceKinematicsExtractor`

输入：
- `positions_m: (N,3)`：当前粒子位置
- `initial_positions_m: (N,3)`：初始粒子位置（或用于定义参考系）
- `surface_indices: (Ns,)`：顶面/近顶面粒子索引（复用已有 `surface_band` 逻辑）

输出（与 height grid 对齐）：
- `height_field_mm: (Ny,Nx)`：与当前一致，使用 `max(z_disp)` 或类似策略
- `uv_disp_mm: (Ny,Nx,2)`：面内位移场（u=dx, v=dy，单位 mm）

核心策略：
- 使用与 height_field 相同的 binning（按 x/y 落格到 `(Ny,Nx)`）。
- 对每个 cell 内粒子：
  - `z_disp = z - z0` 取 max（顶面高度）
  - `dx = x - x0`、`dy = y - y0` 取均值或加权均值（避免噪声）
- 对空 cell：使用 0 填充，并提供可选的轻量平滑（已有 `_box_blur_2d` 可复用扩展到 2 通道）。

注意：坐标翻转/范围对齐
- 当前 `GLSurfMeshItem` 的 `x_range=(w/2,-w/2)` 与 `y_range=(h,0)` 会引入左右/上下翻转。
- 现有 height_field 虽然视觉上“看起来对”，但 uv 位移用于纹理 warp 时必须与纹理坐标系保持一致。
- 本变更将把“grid→texture→mesh”的映射写成显式函数，并在 debug overlay 里可视化方向（例如在右滑时 marker 向右移动）。

### 2) Marker 纹理 warp：`MPMMarkerWarper`

目标：给定 base marker 纹理（静态点阵），以及 `uv_disp_mm`，生成 per-frame 的 warped texture（同尺寸，如 320x560），使 marker 随面内位移移动/扭曲。

实现方式（优先顺序）：
1. 若 `cv2` 可用：使用 `cv2.remap`（快且实现简洁）
   - 输出纹理每个像素 `(x,y)`，从输入 base 纹理采样 `(x - dx_px, y - dy_px)`（逆向映射，避免空洞）。
2. 无 `cv2`：使用 numpy 进行近邻或双线性采样（作为 fallback，性能次之但可用）。

位移换算：
- `dx_px = dx_mm / gel_w_mm * tex_w`
- `dy_px = dy_mm / gel_h_mm * tex_h`
并根据实际 texcoords 的翻转情况调整符号（在 debug overlay 中验证）。

输出：
- `warped_marker_tex_np: (tex_h, tex_w, 3) uint8`
并通过 `Texture2D.setTexture(...)` 更新到 GPU。

### 3) 压头可视化：`MPMIndenterVisualizer`（可选）

两种策略（用户可选/后续可切换）：
- **2D overlay（优先落地）**：在输出 RGB 上画压头投影轮廓（circle/box），直接在 numpy 图像上 `cv2.circle/rectangle`。
  - 优点：不需要额外 GL 对象、遮挡关系不敏感、实现快。
  - 缺点：不是真实 3D 渲染，光照不一致。
- **3D 渲染**：在 `MPMSensorScene` 中加入 `GLModelItem`（例如 `circle_r4.STL` 或用户指定 STL），并用 obstacle center 更新其 transform。
  - 优点：与 FEM 的深度渲染几何更一致。
  - 缺点：需要处理坐标系/尺度、以及与 gel 表面的相对位置/遮挡。

本提案建议：默认先实现 2D overlay（最小复杂度），并保留扩展点到 3D 渲染。

### 4) Debug Overlay（强烈建议）

为避免“看起来不对但不知道哪错了”，增加调试叠加选项：
- `uv`：显示 `|u|,|v|` 强度热力图（叠加在 RGB 上）
- `warp`：显示 warp 像素偏移量统计（max/mean）与方向箭头（例如在固定位置画一个小箭头）
- 记录关键日志：每帧 `press/slide`、`uv_disp_mm` 的 min/max、height_field 的中心-of-mass（已有打印可复用但需要更结构化）

## 与现有 FEM 管线的对齐点

- 视觉风格（相机、光照、输出尺寸）：继续复用当前 `MPMSensorScene` 的 lights/camera 参数，使其与 `xengym/render/SensorScene` 对齐。
- marker 语义：FEM 侧 marker 的 warp 来自真实表面网格（`vis_fem_mesh`）+ `MarkerTextureCamera` 的投影渲染；MPM 侧用“位移场驱动的纹理 warp”去逼近同一语义。
- 轨迹对齐：继续复用 `MPMSimulationAdapter.frame_controls` 驱动 FEM 与 MPM 的同帧控制信号，避免“同 index 不同位姿”的错觉。

## 风险与权衡

- 仅用顶面粒子估计 uv 位移会有噪声；需轻量平滑与空洞填补。
- 坐标翻转是最常见 bug 源；必须用 debug overlay 验证“右滑→marker 向右移动”这种一眼能看出的断言。
- 2D overlay 的压头投影需要与 gel 的投影参数一致；否则会出现“压头位置不对”的新困惑，因此必须基于同一 `ortho_space` 做映射。

