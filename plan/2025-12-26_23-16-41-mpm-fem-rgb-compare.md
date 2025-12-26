---
mode: plan
cwd: "F:/workspace/xenmpm"
task: "根据 Report.md 生成执行计划（MPM vs FEM RGB 对齐与归因收敛）"
complexity: medium
tool: mcp__sequential-thinking__sequentialthinking
tool_status: unavailable
total_thoughts: 7
created_at: "2025-12-26T23:19:08.8670716+08:00"
source: "Report.md"
---

# Plan: MPM vs FEM RGB 对比对齐与差异归因收敛

🎯 任务概述
当前“MPM vs FEM RGB 差异”混入了大量非力学因素（几何/轨迹/坐标/渲染策略不对齐），因此差异不具备指向性。
本计划按优先级逐项对齐这些变量，并补齐可观测的中间产物与日志，把问题收敛到可归因、可回归的单点。

📋 执行计划
1. 基线复现与参数固化：固化一套 CLI 参数（包含 indenter 类型/尺寸、摩擦、marker 模式、tint 开关等）与输出目录，并记录对比截图对应的 `frame_id`。
   - 验收：重复运行帧数/阶段分段一致（`430/5≈86` 帧），关键日志一致。
2. FEM 压头“接触面/朝向”验证：验证 `circle_r4.STL` 当前朝向是否让 15mm 方形底座接触，并提供 180° 翻转的最小验证开关（先验证事实，再决定资产/代码修复）。
   - 验收：FEM 侧方框轮廓可通过开关稳定复现/消除；确定“正确接触面”的朝向。
3. DepthRenderScene 变换链稳定：检查构造期 `.rotate/.translate` 是否被每帧 `setTransform()` 覆盖；若是，将“固定旋转/偏移”与“每帧 pose”合并为单一路径，避免隐式状态。
   - 验收：压头位姿可解释且稳定（同一帧相同输入得到相同世界变换）。
4. 压头几何体一致化（MPM vs FEM）：两边统一几何与尺寸（优先先统一为 box；若走 STL，则两边都用同一 tip STL/同一尺寸来源），并在启动日志打印最终生效配置。
   - 验收：同帧接触轮廓形状一致，几何因素不再污染对比。
5. 摩擦参数对齐：对齐 FEM `fric_coef` 与 MPM `mu_s/mu_k`（至少数量级一致），并将对齐值作为显式配置（避免默认值在两边漂移）。
   - 验收：滑移阶段的剪切带/粘滑表现不再被摩擦量级差异主导。
6. 坐标系与滑移正方向统一：把 dx_slide 的符号约定集中到单一坐标变换函数，清理散落在物理/渲染/overlay 的补丁式取反，并增加同帧 pose 对照日志。
   - 输出：每帧记录 MPM `obstacle_centers[1]` 与 FEM `set_object_pose(x,y,z)`；必要时保存带压头 overlay 的图用于目视确认。
   - 验收：同帧两侧压头 x 方向单调性一致，“强边缘在相反侧”消失或可被日志解释。
7. 渲染策略对齐：对齐/可控化 MPM depth tint（关闭/开启）与 marker 语义（static/warp），并核对投影尺度（`gel_size_mm` vs `cam_view_width/height`）一致性。
   - 验收：关闭 tint、统一 marker 语义后，RGB 差异显著收敛。
8. 中间产物输出与差异度量：同时导出 MPM/FEM 的 height_field、uv_disp、接触 mask（或近似），并计算简单统计量（MAE/峰值/分位数）+ 差异图，作为归因依据。
   - 验收：可以把差异主要来源归因到（几何/摩擦/坐标/渲染）中的某一类，并给出量化证据。
9. 回归验证与文档沉淀：补充轻量回归脚本/测试，锁定对齐后的关键不变量（几何类型/尺寸、摩擦、坐标方向、渲染开关），并把推荐运行命令写入文档。
   - 验收：`quick_test.py` 或 `example/test_*.py` 能快速发现关键不变量回退并给出明确失败信息。

⚠️ 风险与注意事项
- 默认值变更会影响现有 demo：优先用新增 CLI 开关/显式配置推进，避免静默改变默认。
- 若需要新增/裁切 STL：需记录资产来源与几何意图，并控制体积（避免提交不可追溯大文件）。
- 坐标/渲染链路跨度大：先用“同帧 pose 日志 + overlay”闭环验证，再做结构性重构，避免一次改动过大导致难回滚。

📎 参考
- `Report.md:1`
- `example/mpm_fem_rgb_compare.py:443`
- `example/mpm_fem_rgb_compare.py:987`
- `example/mpm_fem_rgb_compare.py:1035`
- `example/mpm_fem_rgb_compare.py:1207`
- `example/mpm_fem_rgb_compare.py:1229`
- `example/mpm_fem_rgb_compare.py:1363`
- `xengym/fem/simulation.py:233`
- `xensesdk/xensesdk/ezgl/GLGraphicsItem.py:191`
