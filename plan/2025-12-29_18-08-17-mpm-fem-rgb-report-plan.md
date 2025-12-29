---
mode: plan
cwd: "F:/workspace/xenmpm"
task: "根据 Report.md 生成可执行计划：MPM vs FEM RGB 对齐与伪影收敛"
complexity: medium
tool: mcp__sequential-thinking__sequentialthinking
total_thoughts: 0
created_at: "2025-12-29T18:08:17+08:00"
---

# Plan: MPM vs FEM RGB 对齐与伪影收敛（基于 Report.md）

🎯 任务概述

当前 MPM vs FEM 的 RGB 观感差异混入了多类“非力学因素”（摩擦/尺度/深度语义/坐标翻转/渲染叠色/marker 语义），以及 MPM 高度场重建带来的暗盘与 halo 伪影。目标是在不引入过度复杂度的前提下：先固化可复现基线与审计输出，再按优先级逐项对齐/收敛，使最终差异可被可靠归因到参数或物理模型本身。

> 备注：当前 Codex CLI 环境未暴露 `mcp__sequential-thinking__sequentialthinking` 调用入口，因此 `total_thoughts=0`；本计划基于 `Report.md` 与代码静态审阅手工拆解。

📋 执行计划

1. **建立“可运行环境”与基线产物目录（先保证可复现）**
   - 动作：在可运行环境（建议 conda + Python 3.9，具备 `taichi` 与 `xensesdk.ezgl`）执行基线命令，固定 `--save-dir` 输出。
   - 推荐基线：`python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/baseline`
   - 产出：`output/rgb_compare/baseline/run_manifest.json`、`metrics.csv/json`、`intermediate/frame_*.npz`（若开启）。
   - 验收：同机连续运行 2 次，`run_manifest.json` 中 `trajectory.total_frames` 与 `frame_to_phase` 长度一致；`metrics.csv` 可被 `Import-Csv`/Excel 正常解析。

2. **对齐检查：参数/尺度/压头接触面（消除“看起来不一样但不一定物理错”）**
   - 动作：核对启动日志与 manifest：
     - 摩擦：确认 `aligned=true`（或显式使用 `--fric`/`--fem-fric`/`--mpm-mu-s`/`--mpm-mu-k`）。
     - 尺度：确认 `Scale ... consistent=true`；若为 false，先对齐 `cam_view_*` 与 `gel_size_mm` 再继续对比。
     - 压头：明确 FEM `--fem-indenter-face tip|base` 与 MPM `--indenter-type`；必要时开启 `--mpm-show-indenter` 作为 2D 对齐证据。
   - 产出：记录一条“对齐清单”（可追加到 `run_manifest.json` 的 `run_context.resolved` 或单独写 `baseline_checklist.md`）。
   - 验收：对齐项全部为 true/一致后，再进入伪影收敛阶段。

3. **复现并定位伪影类型（用中间量闭环归因）**
   - 动作：从 press/slide/hold 三个 phase 各选 3 帧，结合 `intermediate/frame_*.npz` 统计：
     - `height_field_mm` 的 min/p1/p99（是否出现“远超几何允许的异常深值”）
     - `uv_disp_mm` 的幅值分位（是否与 marker warp 可见性相符）
   - 产出：一个小表格（frame_id → phase → 指标 → 是否出现 dark blob/halo/边缘拉丝）。
   - 验收：能把现象至少归到以下一类：高度场异常深值 / 高度场台阶+灯光 / marker warp 翻转与出界 / 参数未对齐。

4. **高度场收敛：优先抑制 dark blob，再收敛 halo（KISS：先调参后改算法）**
   - 动作（按优先级）：
     1) 保持 `--mpm-height-clamp-indenter on`（footprint 内 clamp 到压头表面）
     2) 调 `--mpm-height-fill-holes-iters`（先增大）与 `--mpm-height-smooth-iters`（必要时增大）
     3) 若 footprint 外仍有极端负值：新增“离群值裁剪/稳健 inpaint”开关（避免静默改变默认）
   - 产出：同一批抽样帧的 before/after；并把有效参数写入 manifest（或写入 `output/.../tuning_notes.md`）。
   - 验收：暗盘显著消失；halo 收敛到接触边界且强度下降；中间量显示台阶/极端负值减少。

5. **marker 收敛：确保 warp 语义正确，并消除“翻两次/边缘拉丝”**
   - 动作：
     - 使用 `--mpm-marker warp --mpm-debug-overlay uv|warp` 做方向与量级自检；
     - 梳理翻转链路（高度场 flip、UV flip+u negate、warp flip_x/flip_y），把“同一轴修正”收敛到单一层（提供显式开关与日志/manifest 回显）。
     - 将 cv2 remap 边界策略与 numpy fallback 对齐（避免 `BORDER_REFLECT101` 造成边缘拖影干扰归因）。
   - 产出：至少一组“压头向左滑 → uv/warp 叠加显示也向左”的证据帧；以及对齐后的 flip 配置记录。
   - 验收：marker 不再出现“反向/不动/局部抽风”；边缘拉丝不再由 remap 出界放大。

6. **审计与回归：把“对齐项/伪影项”固化为轻量测试与文档**
   - 动作：
     - 扩展 `quick_test.py`：新增对关键默认值/CLI flag/manifest 字段的断言（保持无 ezgl/taichi 依赖）。
     - 文档：在 `Report.md` 增补“推荐基线命令 + 排查顺序 + 常见现象→开关/中间量”的最终版本。
   - 产出：`python quick_test.py` 通过；文档可直接指导复现/定位。
   - 验收：后续改动（高度场/warp/flip）若引入回退，会在 quick_test 或 manifest 差异中被快速发现。

⚠️ 风险与注意事项

- **环境风险**：Python 版本与 `xensesdk` 二进制兼容性（建议 Python 3.9）；OpenGL 上下文/驱动差异会影响渲染可复现性。
- **数据体积**：开启 `--export-intermediate` 可能产生大量 npz；需要抽样导出策略（例如 `--export-intermediate-every`）。
- **向后兼容**：翻转链路与 remap 边界策略调整应通过开关引入，避免静默改变旧 demo 的观感与依赖假设。
- **归因顺序**：未完成 A 类对齐前，不应把差异直接归因到 MPM 物理 bug。

📎 参考

- `Report.md:1`
- `example/mpm_fem_rgb_compare.py:195`（FEM/MPM 摩擦默认值）
- `example/mpm_fem_rgb_compare.py:154`（`mpm_height_clamp_indenter` 默认值）
- `example/mpm_fem_rgb_compare.py:831`（DepthCamera 正交视野设置）
- `example/mpm_fem_rgb_compare.py:1078`（clamp 到压头表面逻辑入口）
- `example/mpm_fem_rgb_compare.py:460`（`warp_marker_texture` 语义与逆向映射）
- `example/mpm_fem_rgb_compare.py:523`（`cv2.remap` 边界模式）
- `example/mpm_fem_rgb_compare.py:1481`（UV flip + u negate）
- `example/mpm_fem_rgb_compare.py:1591`（高度场 flip）
- `example/mpm_fem_rgb_compare.py:2932`（启动日志：aligned/scale 检查）
- `example/mpm_fem_rgb_compare.py:2626`（UI 帧循环取模）
- `xengym/fem/simulation.py:463`（FEM 深度缩放 `depth *= 0.4`）
- `xengym/render/sensorScene.py:243`（FEM 摩擦系数注入点）
- `xengym/assets/data/light.txt:1`（多色灯光配置）

