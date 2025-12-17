# Summary

新增一个独立对比脚本 `example/mpm_fem_rgb_compare.py`，复用项目现有的 **视触觉传感器 RGB 渲染链路**（`xengym.render.VecTouchSim` / `SensorScene.get_image()`）来对比 **MPM 与 FEM** 在同一“按压 + 滑移”场景下的传感器输出图像。

脚本将：
- FEM：沿用现有方案（`DepthCamera` 渲染深度图 → `VecTouchSim.step()` → `get_image()` / `get_diff_image()`）。
- MPM：运行 Taichi MPM，提取胶体顶面高度场（height field），并用与 `VecTouchSim` 同风格的光照/纹理/相机设置渲染为 RGB 图像（用于与 FEM 图像并排对比）。
- 输出：交互式双图像视窗（FEM vs MPM），支持 raw/diff 切换与帧保存。

本提案仅定义需求与实现计划；在批准后进入实现阶段。

## Motivation

当前已有 `example/mpm_fem_compare.py` 做 3D 点云层面的对比，但用户的核心需求是对比 **视触觉传感器的 RGB 输出**（即下游算法真实使用的数据形态）。

复用 `VecTouchSim` 的 RGB 渲染链路可以：
- 用同一套光照与相机参数得到可比的图像；
- 更直观地观察按压形状、滑移摩擦引起的形变扩散、以及响应随时间的变化；
- 为后续的校准/验证（例如对齐 FEM 与 MPM 的 footprint、边界条件、材参）提供可视化依据。

## Goals

- 在 `example/` 下新增一个脚本：
  - 在 `xengym` conda 环境中可运行；
  - 同时驱动 FEM 与 MPM，生成 **并排** 的 RGB 图像（raw 与 diff 两种视图）；
  - 支持“按压 + 滑移”轨迹，轨迹参数可通过 CLI 配置；
  - 支持将关键帧/序列输出到 `output/`（或用户指定目录）。
- 最大程度复用现有渲染路径：
  - FEM 路径直接使用 `xengym.render.VecTouchSim`；
  - MPM 路径复用 `SensorScene` 同风格的 `GLSurfMeshItem + RGBCamera` 设置（尽量保持同一视觉风格）。

## Non-Goals

- 不追求 FEM 与 MPM 完全数值等价（本脚本首要目标是“可比的传感器图像输出”与“可调参对齐”）。
- 不在本次变更中实现 MPM 的 marker 纹理随面内拉伸的精确 warp（可作为后续增强项）。
- 不引入新的 UI 框架或新的大型外部依赖。

## Scope

In scope:
- 新增：`example/mpm_fem_rgb_compare.py`。
- 可能新增：少量渲染复用工具（如“height-field → RGB”渲染器），优先写在脚本内以控制范围。
- 文档：在 `CLAUDE.md` 或 `example/` 内补充运行说明。

Out of scope:
- 修改 FEM 数据格式或 FEM 求解器实现；
- 给 MPM solver 新增 mesh-SDF 或任意 STL 碰撞（本次优先使用 sphere/box 这类已有 SDF 障碍体，保证落地）。

## Open Questions

- 默认对比视图选择：
  - 建议默认 `raw`（直接 `get_image()`），`diff` 作为可选项（相对参考帧，增强按压可见性）。
- 默认压头几何：
  - 建议默认 sphere（对应 `circle_r*.STL`），因为 MPM 已有 sphere SDF 支持；box 保留为可选项。
- MPM → height field 的提取策略：
  - 建议默认 “每格取最大 z”（稳定优先），插值作为后续可选增强项。
