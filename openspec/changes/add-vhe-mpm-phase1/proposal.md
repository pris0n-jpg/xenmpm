# Change: 构建 Phase1 可微 VHE-MLS-MPM 求解器

## Why
- 落地《方案 v2.6》中针对 Taichi 的 VHE-MLS-MPM 需求，提供热力学一致、支持自动微分的基线，实现后续论文与 Real2Sim 的核心数值骨架。
- 当前仓库缺少对应本构、能量统计与接触摩擦的实现，无法支撑验证场景与能量一致性分析。

## What Changes
- 增加 3D 显式 MLS-MPM/APIC 求解流程（clear_grid → p2g → grid_op → g2p → update_F_and_internal → reduce_energies → cleanup_ut），支撑 Ogden + 广义 Maxwell + 可选 Kelvin-Voigt 体粘性。
- 实现投影能量修正统计（ΔE_proj_step、E_proj_cum）与粘性耗散分解，保持粒子级增量与全局累积输出。
- 引入正则化弹塑性摩擦 + SDF penalty 接触，含 grid_ut 迟滞清理与切向弹簧内变量管理，保障 stick–slip 可微行为。
- 提供 Taichi 自动微分封装，对材料参数、初始状态、外部控制暴露梯度接口，允许自定义 loss。
- 统一配置/输出（dataclass/JSON/YAML），包含能量时间序列、切向力-位移曲线、网格收敛数据，并补齐参数稳定性（Drucker-type）与时间步约束检查。
- 交付验证场景：单轴拉伸、纯剪切+客观性、应力松弛/Δt 收敛、能量守恒与投影修正对比、块-板 stick–slip、GelSlim incipient slip、接触/弹性球网格收敛。

## Impact
- 影响规格：`specs/vhe-mpm-solver/spec.md`，`specs/vhe-validation/spec.md`。
- 影响代码：新增/重构 `mpm_solver.py`, `constitutive.py`, `contact.py`, `decomp.py`, `fields.py`, `config.py`, `autodiff_wrapper.py`, `scenes/`, `validation/`，以及 CLI 入口（如 `main.py`）和输出脚本。当前阶段不引入外部资产文件。
