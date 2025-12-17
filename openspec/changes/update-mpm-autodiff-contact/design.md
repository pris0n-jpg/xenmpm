## Context
- 现状问题：autodiff 使用 numpy + Tape(loss=None)，梯度不可用；Ogden 本构硬编码 2 项，配置可能越界；接触法向/切向刚度耦合；验证与 CLI 未覆盖粘弹性/摩擦。
- 目标：提供可微的 Taichi 计算图、可配置 Ogden 1–4 项、可分法向/切向摩擦刚度与兼容旧配置、补充验证与 CLI 场景选项。

## Goals / Non-Goals
- Goals: 可用梯度输出；Ogden 支持 1–4 项且超限报错/截断；接触参数拆分并在求解器中生效；新增验证/CLI 选项覆盖粘弹性与摩擦。
- Non-Goals: 性能优化、Barrier 接触、热-力耦合。

## Decisions
- Autodiff：loss 存储在 Taichi scalar field（needs_grad=True），在构造 solver 时创建；使用 `ti.ad.Tape(loss=loss_field)` 包裹整个前向（step/run + compute_loss kernel）；对粒子状态 `x/v/F/b_bar_e` 与材料参数 `ogden_mu/ogden_alpha/maxwell_G/maxwell_tau` 设置 `needs_grad=True`，未求导的中间量显式 `ti.stop_grad`；提供预定义可微 loss kernel（位置/力/能量匹配，通过 `ti.template()` 或独立 kernel 选择），禁止 numpy 参与；SPD 投影不可微，AD 模式下采用 straight-through 估计或关闭投影分支（示例：`b_proj = ti.stop_grad(b_proj) + b_relaxed - ti.stop_grad(b_relaxed)`）；Maxwell 内变量梯度需显式开启且注意显存占用。
- Ogden：使用固定长度(4) taichi field/列表，静态循环展开，n_terms 由配置控制；from_dict 校验长度并报错/截断，<=4。
- 接触：配置拆分 normal/tangent stiffness（可保留旧 contact_stiffness 兼容）；compute_contact_force 接收独立刚度；grid_op 传递切向刚度。
- 验证/CLI：新增 Maxwell 松弛/摩擦验证脚本或模式；CLI 增加 --arch (cpu/gpu) 和场景枚举，生成能量/摩擦曲线输出。

## Risks / Trade-offs
- Taichi autodiff 需保证 loss 可微且无 numpy 参与；可能限制自定义 loss 形态。
- SPD 投影不可微，AD 模式需 straight-through 或关闭投影，可能影响梯度准确性；需在文档与代码中显式切换。
- Maxwell 内变量 b_bar_e 开启 needs_grad 会增加显存/算力，必要时可在 AD 模式下裁剪步数或关闭部分分支。
- 固定 4 项 Ogden 在需求超出时需报错/截断，需文档提醒。
- 接触参数拆分可能改变默认行为，需向后兼容。

## Plan / Migration
- 调整 config.from_dict 增加长度校验与旧字段兼容；更新 constitutive/solver 使用 general Ogden。
- 重构 autodiff_wrapper：needs_grad 标记、Taichi kernel loss、梯度导出；在 AD 模式下提供切换开关（SPD STE/关闭投影、max_grad_steps 控制长度，默认 50 步）并显式创建/重置 loss field。
- 更新 contact/solver 使用切向刚度；验证兼容旧配置。
- 扩展 validation/cli 场景与 arch 选项；输出能量/摩擦曲线。

## Open Questions
- 是否需要切向阻尼/独立摩擦平滑参数？（暂仅刚度拆分）
- 验证输出格式（CSV/图像）是否需对齐现有 pipeline？当前假设 CSV。
