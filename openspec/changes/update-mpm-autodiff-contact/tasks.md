## 1. 实施
- [x] 1.1 配置兼容（Ogden 项数、接触刚度拆分，旧配置自动映射）。
- [x] 1.2 本构（Ogden 1–4 项通用版本，solver 使用通用接口）。
- [x] 1.3 接触（法/切向刚度分离，grid_op 传递，迟滞保留）。
- [⚠️] 1.4 Autodiff 重构 **（基础设施完成，但存在Taichi AD限制）**
  - [x] 1.4.1 创建 loss scalar field（needs_grad=True），在 solver/init 阶段构造并支持重置。
  - [x] 1.4.2 材料参数/粒子场 needs_grad：ogden_mu/ogden_alpha/maxwell_G/maxwell_tau、x/v/F/b_bar_e；未求导中间量（如中间 SVD/特征分解结果）使用 stop_grad，确保梯度仅流向目标参数/状态。
  - [x] 1.4.3 SPD 投影策略：AD 模式下提供开关（STE 或关闭投影），默认 STE；使用 ti.static() 统一路径，make_spd_ste() 实现。
  - [x] 1.4.4 预定义可微 loss kernel（位置/velocity/能量/COM 匹配，使用独立 kernel），禁止 numpy 参与。添加 target validation。
  - [x] 1.4.5 Tape 流程：with ti.ad.Tape(loss) 包裹 step/run+compute_loss；导出梯度到 numpy。
  - [x] 1.4.6 显存控制：max_grad_steps（默认 50）或裁剪开关，避免 b_bar_e grads 过大。
  - [x] 1.4.7 Kernel 分离：reduce_energies 分为 clear_energy_fields + reduce_energies 以符合 Taichi AD 规则。
  - [x] 1.4.8 **Codex审查修复** (2025-12-01): 所有5个问题已解决
    - [x] 严重: AD Tape阻断 - 添加显式NotImplementedError阻止运行时崩溃
    - [x] 严重: 文档缺失 - 移动TAICHI_AUTODIFF_LIMITATIONS.md到xengym/mpm/
    - [x] 高: Maxwell参数字段不匹配 - CLI使用maxwell_branches列表
    - [x] 高: Contact刚度字段名错误 - 统一为contact_stiffness_normal/tangent
    - [x] 中: 双核路径验证 - 确认update_F_and_internal_ste已删除，ti.static()统一
  - [❌] **BLOCKED**: P2G/G2P 包含 atomic scatter/gather 操作，Taichi AD v1.7.4 不支持。梯度无法通过仿真步反向传播。
    - **影响**: 无法计算 loss 对材料参数/初始状态的梯度（通过仿真步）。
    - **缓解**: 已添加文档 (xengym/mpm/TAICHI_AUTODIFF_LIMITATIONS.md) 和运行时阻断异常。
    - **未来**: 需要手动伴随方法或 Taichi 版本升级支持。
- [x] 1.5 验证与 CLI：新增/扩展验证场景涵盖 Maxwell 松弛与摩擦曲线；CLI 支持 --arch cpu/gpu、场景选择，输出能量/摩擦数据。
  - [x] Maxwell 松弛测试场景 (cli.py:run_maxwell_relaxation_test)
  - [x] 接触摩擦测试场景 (cli.py:run_friction_test)
  - [x] CLI --arch 参数支持 (cpu/gpu/cuda/vulkan)
  - [x] 场景选择 --scene drop/maxwell/friction
  - [x] CSV 格式输出 (maxwell_relaxation.csv, friction_curve.csv)
- [x] 1.6 文档与示例：更新 README/实现状态说明，示例配置展示新接触参数与 Ogden 多项。
  - [x] README.md 更新 autodiff 限制说明
  - [x] README.md 添加 CLI 使用文档 (场景/架构选项)
  - [x] README.md 更新接触配置说明 (normal/tangent stiffness)
  - [x] 创建示例配置 (config_maxwell_demo.json, config_friction_demo.json)
  - [x] 创建高级用法示例 (examples/advanced_usage.py)

## 2. 验证
- [x] 2.1 配置加载测试：覆盖合法/超限 Ogden、旧/新接触参数映射。
- [ ] 2.2 数值验证：运行 Maxwell 松弛 & 摩擦验证脚本，检查能量/曲线输出。
- [❌] 2.3 Autodiff 验证：对材料参数和初始状态求梯度 - **BLOCKED by Taichi AD limitation (see 1.4 above)**。

## 3. Autodiff 技术债务 (未来工作)
- [ ] 3.1 调研 Taichi 新版本 (>v1.7.4) 是否改进 atomic operations AD 支持。
- [ ] 3.2 评估手动伴随方法实现的工作量（为 P2G/G2P 手写反向传播）。
- [ ] 3.3 考虑外部 AD 框架 (JAX/PyTorch) 包装方案。
- [ ] 3.4 简化物理模型（仅用于参数优化，非完整 MPM）作为临时方案。

> 注：1.1-1.3 已在前 4 轮 codex review 中完成并验证。1.4 autodiff 基础设施已完成，但受 Taichi AD 限制阻塞。
