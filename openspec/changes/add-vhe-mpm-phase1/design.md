## Context
- 依据《方案 v2.6》Phase 1 要求，需在 Taichi 中实现热力学一致的 3D 显式 MLS-MPM/APIC 求解器，支持 Ogden+广义 Maxwell 粘弹性、可选 Kelvin-Voigt 体粘性，正则化弹塑性接触，及投影能量修正统计，且需可微。
- 目标覆盖科研验证（GelSlim/摩擦/能量一致性）而非量产性能优化；时间积分为一阶算子分裂方案，允许可控投影误差。

## Goals / Non-Goals
- Goals: 明确分层架构、字段布局、能量记账、接触摩擦迟滞策略、自动微分封装、稳定性检查与验证输出格式；给出实现顺序和风险缓解。
- Non-Goals: Phase 2 的 barrier-type 零穿透接触、热-力耦合、ROM/高阶结构保持积分、资产/GUI 打磨。

## Decisions
- **分层架构**：核心数值层（mpm_solver/constitutive/contact/decomp）、基础设施层（fields/config/autodiff_wrapper）、场景+验证层（scenes/validation）、应用入口层（main.py/CLI）。上层仅依赖抽象接口（SDF、配置、线代封装）。
- **字段布局**：粒子字段含 x/v/F/C/mass/volume/b_bar_e[k]/delta_E_viscous_step/delta_E_proj_step；网格字段 grid_m/grid_v/grid_ut/grid_contact_mask/grid_nocontact_age；全局标量 E_kin/E_elastic/E_viscous_step/E_viscous_cum/E_proj_step/E_proj_cum。grid_ut 为跨步内变量，clear_grid 不清空。
- **求解流程**：step 顺序固定：clear_grid → p2g → grid_op → g2p → update_F_and_internal → reduce_energies → cleanup_ut。reduce_energies 紧随本构更新以保证能量统计与状态一致；cleanup_ut 用迟滞计数器避免 ghost friction。
- **本构积分**：Ogden 偏差 + Barrier 体积 + Maxwell 分支（上对流→松弛→SPD+等容投影）。投影阶段记录 ΔE_proj_step，粒子级累积后再 reduce；可选 Kelvin-Voigt 体粘性受配置驱动且默认关闭。
- **线代抽象**：decomp.py 提供 polar_decompose / eig_sym_3x3（封装 ti.svd/ti.sym_eig）。如需 SafeSVD/自定义梯度，仅在该层替换。
- **接触与摩擦**：SDF penalty 法向 + 正则化弹塑性摩擦（切向弹簧 u_t，tanh 过渡静/动摩擦）；grid_contact_mask 标记接触，grid_nocontact_age ≥ K_clear 时清理 u_t。
- **自动微分**：autodiff_wrapper 通过 ti.ad.Tape 包装 step/run，允许对材料参数、初始粒子状态、外部驱动求导；loss_fn 由用户自定义，暴露梯度读取接口。
- **配置与输出**：dataclass/JSON/YAML 载入网格/时间步/材料/接触/输出选项；输出粒子状态、能量序列、切向力-位移、收敛数据，支持后处理绘图脚本。
- **稳定性检查**：启动时执行 Ogden 参数 Drucker-type 检查（符号一致、路径扫描正定性）与 Δt 约束（三类：弹性 CFL、粘性时间尺度、接触刚度）。J 裁剪与 SPD 投影避免 NaN。

## Risks / Trade-offs
- 投影能量修正：需控制 |ΔE_proj_cum|/E_viscous_cum ≲ 1%，并随 Δt 一阶收敛；风险通过验证场景量化。
- 性能与可微稳定性的权衡：SafeSVD/Eigen 替换可能影响速度；初期采用 Taichi 内置，必要时再行优化。
- 接触穿透与摩擦滞留：Penalty 法存在有限穿透，grid_ut 迟滞窗口需调参避免黏连与过度清理。

## Migration Plan
- 先建立 config/fields/decomp 骨架 → mpm_solver 流程 & 能量记账 → constitutive（含 ΔE_proj）→ contact + cleanup_ut → autodiff_wrapper 接口 → 场景/验证脚本与输出 → 参数稳定性与 Δt 检查 → 运行验证并记录基线。

## Open Questions
- GelSlim/块板场景的具体几何与参数（SDF 定义、压入/拖动轨迹）是否已有数据源？若无需在实现阶段提供可修改默认。
- 输出格式是否需与现有 quick_test/例程保持兼容（CSV/npz）？目前假设 CSV+图像即可。
