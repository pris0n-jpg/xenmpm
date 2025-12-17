## ADDED Requirements

### Requirement: Phase1 VHE-MLS-MPM 求解流程
系统必须（SHALL）提供 3D 显式 MLS-MPM/APIC 求解器，默认初始态为应力自由且无内变量历史：粒子 v=0、F=I、C=0、b_bar_e[k]=I，网格 grid_m/grid_v=0，grid_ut=0 且 clear_grid 不清空 grid_ut 与 grid_nocontact_age。单步执行顺序固定为 clear_grid → p2g → grid_op → g2p → update_F_and_internal → reduce_energies → cleanup_ut，并支持均匀规则网格与二次 B-Spline 核。

#### Scenario: 固定步进序执行
- **WHEN** 运行 step 时
- **THEN** 按固定顺序调用各 kernel，grid_ut 在 clear_grid 之后仍保留，上一步累计的 grid_nocontact_age 仅在 cleanup_ut 内更新。

### Requirement: VHE 本构积分与投影能量修正
系统必须（SHALL）实现 Barrier 型体积势能、修正主伸长的多分支 Ogden 偏差超弹性、广义 Maxwell 粘弹性（上对流→松弛→SPD+等容投影）与可选 Kelvin-Voigt 体粘性；每个分支在投影时记录 ΔE_proj_step，并在粒子级累积 delta_E_proj_step、delta_E_viscous_step 后写回 tau 供下一步 p2g 使用。

#### Scenario: 本构更新含能量记账
- **WHEN** update_F_and_internal 更新粒子 F 与内变量
- **THEN** 对每个 Maxwell 分支执行上对流、松弛、SPD+等容投影，累积 delta_E_viscous_step 与 delta_E_proj_step，并输出包含体积+Ogden+粘性+可选体粘性的 Kirchhoff 应力。

### Requirement: 能量分解与输出
系统必须（SHALL）在每步 reduce_energies 中将粒子级 delta_E_viscous_step/delta_E_proj_step 归并为 E_viscous_step/E_proj_step，并累积 E_viscous_cum/E_proj_cum；同时计算 E_kin 与 E_elastic（体积+Ogden）。输出接口须提供 E_kin、E_elastic、E_viscous_step、E_viscous_cum、E_proj_step、E_proj_cum 的时间序列供验证。

#### Scenario: 能量序列可导出
- **WHEN** 仿真完成且启用能量输出
- **THEN** 生成包含上述能量分量的时间序列（CSV/等），且 E_proj_cum 与 E_viscous_cum 可用于绘制比值与 Δt 收敛曲线。

### Requirement: 正则化弹塑性接触与摩擦
系统必须（SHALL）支持基于 SDF 的 penalty 法向接触与正则化弹塑性切向摩擦：每个接触节点维护 grid_ut，使用 tanh 过渡静/动摩擦，grid_contact_mask 标记接触，grid_nocontact_age 连续超过 K_clear 时清空 grid_ut，防止 ghost friction。

#### Scenario: Stick–Slip 历程
- **WHEN** 接触节点出现切向加载并达到屈服
- **THEN** 切向力从弹性区平滑过渡到动摩擦平台，grid_ut 在接触解除且等待 K_clear 步后被清理，切向力-位移曲线显示 stick–slip 行为。

### Requirement: 自动微分接口
系统必须（SHALL）提供基于 Ti Tape 的自动微分封装，支持对材料参数（μ0、μ_p、α_p、G_k、τ_k、ζ_bulk 等）、初始粒子状态、边界/接触控制参数求梯度，并允许用户传入自定义 loss（如能量比值、末态位移）。接口返回标量 loss 与参数梯度查询入口。

#### Scenario: 自定义损失可反传
- **WHEN** 用户通过 run_sim_and_compute_loss(cfg, scene_builder, loss_fn, requires_grad=True) 运行仿真
- **THEN** loss_fn 中组合的能量/位移指标可正确反传到配置中标记为可微的参数，且梯度在调用后可读取。

### Requirement: 配置与输出约束
系统必须（SHALL）支持 dataclass/JSON/YAML 配置，包含网格分辨率、时间步 Δt、粒子数量上限、Ogden/Maxwell 参数、可选体粘性、接触系数、输出选项（能量、粒子状态、切向力-位移、收敛数据）。默认初始态为无应力且需允许场景覆盖以支持预加载。输出须包含粒子位置/速度用于可视化。

#### Scenario: 配置驱动仿真
- **WHEN** 用户提供完整配置文件并选择场景
- **THEN** 求解器按配置初始化字段、材料参数与接触参数，输出启用的能量与场景特定数据（如切向力曲线、光流/位移场）。

### Requirement: 稳定性与安全检查
系统必须（SHALL）在仿真启动前执行 Ogden 参数 Drucker-type 检查（符号一致、路径扫描正定），并在配置加载时给出 Δt 约束提示（弹性 CFL、粘性时间尺度、接触刚度）；本构计算须对 J 设下限并使用 SPD 投影避免 NaN。

#### Scenario: 非稳定参数阻断
- **WHEN** 配置中出现违反符号一致或路径扫描失败的 Ogden 参数
- **THEN** 仿真启动时给出错误或显著警告并阻止继续运行，提示用户修正参数或降低 Δt。

### Requirement: 性能与可维护性基线
系统必须（SHALL）在 GPU 环境下支持至少 ~100k 粒子的一阶显式步进，模块化拆分（mpm_solver/constitutive/contact/decomp/fields/autodiff），关键 kernel 保持中文意图注释，提供单元/脚本级测试覆盖能量分解与摩擦循环。

#### Scenario: 基线性能与测试
- **WHEN** 在目标 GPU 上运行 ~100k 粒子场景
- **THEN** 单步耗时保持在可接受范围（随硬件而定），能量/摩擦测试脚本能够无错误完成并生成预期曲线。
