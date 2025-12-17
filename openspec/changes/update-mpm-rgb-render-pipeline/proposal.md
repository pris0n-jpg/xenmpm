# Summary

完善 `example/mpm_fem_rgb_compare.py` 中 **MPM 的 RGB 渲染管线**，使其在“按压 + 滑移”交互下的可视化行为更接近 FEM/真实传感器直觉：

- **压头可视化与运动一致**：MPM 侧在 RGB 画面中能明确看到压头位置，并随轨迹移动。
- **Marker 图层随接触/滑移移动**：MPM 侧 marker 不再是“静态贴图”，而是根据 MPM 顶面 **面内位移场（u,v）** 产生纹理 warp，模拟“印在表面”的 marker 随材料点运动。
- **对齐与可诊断**：增加可选调试输出（例如显示位移场/warp 强度），便于定位坐标翻转、尺度、符号等问题。

该变更为“渲染/可视化增强”，不改变 MPM 求解器物理方程，也不要求 FEM 与 MPM 完全数值一致。

## Motivation

当前对比窗口中，MPM 侧存在用户可见的差异与误导：

1. “压头不动”：MPM 求解器中压头（SDF obstacle）实际在动，但 RGB 渲染链路没有渲染压头几何体，导致用户无法从图像中确认压头位姿。
2. “marker 不动”：MPM 侧 marker 纹理是静态生成（贴在表面网格上），每帧只更新了 z 向高度场与简单着色，因此滑移/剪切不会反映为 marker 的平移/扭曲。
3. 对比诊断困难：缺乏显示位移场/坐标系对齐的手段，问题定位依赖打印日志或主观观察。

## Goals

- 在 **不引入重型依赖** 的前提下，为 MPM 侧补齐：
  - 顶面 **面内位移场（u,v）** 的提取（与现有 height field 同网格对齐，如 `(140,80)`）。
  - marker 纹理 warp（优先使用 `cv2.remap`，无 `cv2` 时提供 numpy fallback）。
  - 可选压头几何体渲染（与 FEM 的 DepthRenderScene/对象 STL 风格一致）。
- 保持脚本可运行性与 CLI 兼容：默认行为不破坏既有使用方式；新增功能通过可选参数启用。
- 提供明确的调试/验证方式：便于验证“marker 确实随滑移移动”和“压头位置/位移场方向正确”。

## Non-Goals

- 不实现完整的“真实相机/光学”数据驱动成像（CNN/NeRF 等学习方法）。
- 不将 MPM 输出升级为完整三维表面网格（x/y/z 顶点），本次优先以 **高度场 + 位移场 + 纹理 warp** 达到“可比且直观”的效果。
- 不承诺 FEM 与 MPM 的压痕轮廓形状完全一致（材参、接触刚度、摩擦与边界条件仍需单独标定/对齐）。

## Scope

In scope:
- 修改 `example/mpm_fem_rgb_compare.py` 的 MPM 渲染实现：
  - 增加 per-frame 的面内位移场提取（顶面粒子 → (u,v) grid）。
  - 增加 marker warp，并提供开关参数（`off/static/warp`）。
  - 增加压头渲染（开关参数）。
  - 增加调试可视化（开关参数）。
- 补充 `CLAUDE.md` 的运行示例与推荐参数（如果 CLI 有新增）。
- 增加一个轻量回归脚本或测试（建议放在 `example/test_*.py` 或 `quick_test.py` 扩展），用于非交互验证输出的基本一致性（例如保存数帧并检查像素变化/位移场统计）。

Out of scope:
- 修改 `xengym/render/SensorScene` 的 FEM 渲染链路（除非为复用工具做极小改动且不会影响现有功能）。
- 修改 `xengym/mpm` 求解器核心（接触/本构/时间积分）。

## Proposed CLI Extensions (Draft)

- `--mpm-marker {off,static,warp}`：关闭/静态/warp marker（默认：`static`，与当前行为一致）。
- `--mpm-show-indenter`：在 MPM 画面渲染压头几何体（默认：关闭）。
- `--mpm-debug-overlay {off,uv,warp}`：叠加显示位移场强度、warp 诊断等（默认：关闭）。
- `--mpm-uv-source {surface_band,top_k}`：位移场采样策略（默认：`surface_band`，复用现有“顶面带宽粒子”）。

## Open Questions (Need Your Confirmation)

1. 你期望的 marker 行为是“**随表面平移/剪切**（warp）”即可，还是必须体现“**拉伸/压缩导致点阵间距变化**”（需要更精确的位移梯度/应变映射）？
2. 你希望 MPM 压头的可视化是：
   - A) 仅在 RGB 画面叠加轮廓/投影（2D overlay，最快最稳），还是
   - B) 真正渲染 3D STL/几何体（更一致但依赖 OpenGL/遮挡关系）？
3. 你用于对比的压头/物体默认是哪个 STL（`circle_r4.STL`、`square_d6.STL` 等）？这会影响我们如何让 MPM 的压头形状与 FEM 侧保持一致。

