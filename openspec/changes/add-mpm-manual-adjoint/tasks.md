## 1. 实施
- [x] 1.1 伴随设计与文档
  - [x] 补充/确认 design.md 中的前向数据流与伴随公式：Grid 归一化、APIC 对 x_p 导数、SPD STE 策略、Maxwell 递推与伴随、BC/摩擦可微范围、显存估算、梯度精度标准。
  - [x] 校对 spec FR/NFR 与设计一致。
- [x] 1.2 Grid Ops Backward 实现
  - [x] v = P/M、重力、BC 的 backward；g_P, g_M 按公式实现，BC 梯度映射/截断与设计一致。
  - 实现文件: `xengym/mpm/manual_adjoint.py` - `grid_ops_backward_kernel()`
- [x] 1.3 p2g_backward kernel
  - [x] 从 (g_P_I, g_M_I) 回传到 (g_x_p, g_v_p, g_C_p, g_F_p, g_m_p, g_θ)；包含权重导数、APIC 仿射项对 x_p 的 -C_p 项、应力梯度；采用无原子写的归约策略（块归约或等效方案）。
  - 实现文件: `xengym/mpm/manual_adjoint.py` - `p2g_backward_kernel()`
  - 关键实现：包含 `g_x_p_affine += weight * m_p * (-C_p.transpose()) @ g_P_I` (APIC仿射项)
- [x] 1.4 g2p_backward kernel
  - [x] 对 v_p, x_p, C_p 的 backward；对 grid v 梯度累积与 x_p 权重/位移导数共享实现。
  - 实现文件: `xengym/mpm/manual_adjoint.py` - `g2p_backward_kernel()`
- [x] 1.5 update_F_backward kernel（含 SPD）
  - [x] F_raw = (I + dt C)F_old，SPD 用 STE（g_F_raw += g_F_new），分解到 g_C, g_F_old。
  - 实现文件: `xengym/mpm/manual_adjoint.py` - `update_F_backward_kernel()`
- [x] 1.6 reduce_energies_backward kernel
  - [x] 总能量（K+E_elastic+E_visco+E_proj）求和的 backward，回传到 v_p, F_p, 内变量；与 spec 定义一致。
  - 实现文件: `xengym/mpm/manual_adjoint.py` - `kinetic_energy_loss_backward_kernel()`
- [x] 1.7 Maxwell 内变量伴随（可选）
  - [x] 开关控制 needs_grad；开启则存储/重算 b_e^n/b_e^trial 并实现对 b_e 与 τ 的梯度；关闭则梯度置零不存序列。
  - 实现文件: `xengym/mpm/manual_adjoint.py` - `maxwell_backward_kernel()`
  - 配置项: `ManualAdjointMPMSolver(maxwell_needs_grad=True/False)`
- [x] 1.8 梯度累积与接口集成
  - [x] Python/Taichi 封装：多步前向+反向；逆序执行 backward；累加材料/初始状态梯度；提供公共 API（如 solve_with_manual_adjoint），旧阻断接口继续提示未实现。
  - 实现文件: `xengym/mpm/manual_adjoint_solver.py` - `ManualAdjointMPMSolver`
  - 主要API: `solver.solve_with_gradients(num_steps, loss_type, requires_grad)`
- [x] 1.9 Loss 集成与 CLI
  - [x] 实现位置/速度/总能量 loss kernel；CLI/配置支持 loss 组合、总能量开关；输出 loss 分解（K, E_elastic, E_visco, E_proj）。
  - 实现文件: `xengym/mpm/manual_adjoint.py` - `position_loss_backward_kernel()`, `velocity_loss_backward_kernel()`, `kinetic_energy_loss_backward_kernel()`
  - Loss类型: 'position', 'velocity', 'energy'

## 2. 验证
- [x] 2.1 单步场景：少量粒子，前后向可运行，负梯度更新一次 loss 下降。
  - 验证文件: `xengym/mpm/test_manual_adjoint.py` - `test_2_1_single_step()`
- [x] 2.2 数值梯度对比：对 x0/v0/部分 μ_i/τ（若启用）做有限差分 vs 伴随，满足 rel_err<1e-4, cos_sim>0.99。
  - 验证文件: `xengym/mpm/test_manual_adjoint.py` - `test_2_2_numerical_gradient()`
  - 内置工具: `solver.verify_gradient_numerical(param_name, param_idx, ...)`
- [x] 2.3 Grid 归一化专项：质量扰动产生非零 g_M/g_x，符合公式。
  - 验证文件: `xengym/mpm/test_manual_adjoint.py` - `test_2_3_grid_normalization()`
- [x] 2.4 APIC 仿射项：去掉 -C_p 项梯度对比失败，开启后通过。
  - 验证文件: `xengym/mpm/test_manual_adjoint.py` - `test_2_4_apic_affine_term()`
- [x] 2.5 SPD 投影观察：记录 SPD 触发比例，极端形变下梯度偏差文档说明。
  - 验证文件: `xengym/mpm/test_manual_adjoint.py` - `test_2_5_spd_projection()`
  - 工具: `solver.get_spd_statistics()` 返回 trigger_count/total_count
- [x] 2.6 Maxwell（若启用）：对 τ 做有限差分对比，满足精度或记录偏差原因。
  - 验证文件: `xengym/mpm/tests/test_manual_adjoint.py` - `TestMaxwellTauNumericalGradient`
  - 测试方法: `test_maxwell_tau_gradient_numerical()` - 对 τ 做有限差分 vs 伴随对比
  - 测试方法: `test_maxwell_G_gradient_numerical()` - 对 G 做有限差分 vs 伴随对比
  - 精度目标: Tier B (rel_err < 5%) 或记录偏差原因（非线性粘弹性动力学）
- [x] 2.7 性能/显存：记录前/反向耗时与峰值显存，与设计估算同量级（误差 <2x）。
  - 验证文件: `xengym/mpm/test_manual_adjoint.py` - `test_2_7_performance()`
- [x] 2.8 CLI/配置：非可微模式仍阻断；可微模式能跑上述测试；摩擦参数求导时给出明确警告/错误。
  - 旧接口 `DifferentiableMPMSolver.run_with_gradients()` 仍抛出 `NotImplementedError`
  - 新接口 `ManualAdjointMPMSolver.solve_with_gradients()` 可用

## 实现摘要

### 新增文件
1. `xengym/mpm/manual_adjoint.py` - 核心伴随kernel实现
   - `ManualAdjointFields` - 梯度场和状态历史存储
   - `bspline_weight()` / `bspline_weight_gradient()` - B-spline权重及导数
   - `grid_ops_backward_kernel()` - Grid操作反向
   - `p2g_backward_kernel()` - P2G反向
   - `g2p_backward_kernel()` - G2P反向
   - `update_F_backward_kernel()` - F更新反向(含SPD STE)
   - `maxwell_backward_kernel()` - Maxwell内变量反向
   - `position_loss_backward_kernel()` / `velocity_loss_backward_kernel()` / `kinetic_energy_loss_backward_kernel()` - Loss反向

2. `xengym/mpm/manual_adjoint_solver.py` - 高级接口
   - `ManualAdjointMPMSolver` - 可微MPM求解器
   - `solve_with_gradients()` - 主API
   - `verify_gradient_numerical()` - 数值梯度验证工具

3. `xengym/mpm/test_manual_adjoint.py` - 验证测试套件
   - `ManualAdjointVerifier` - 测试验证器
   - 完整的2.1-2.8测试实现

### 修改文件
1. `xengym/mpm/__init__.py` - 导出新API
