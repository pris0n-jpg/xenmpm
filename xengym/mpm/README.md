# VHE-MLS-MPM Solver

可微的粘弹性超弹性 Material Point Method (MPM) 求解器，基于 Taichi 实现。

## 功能特性

- **3D 显式 MLS-MPM/APIC 求解器**：支持高阶粒子-网格传输
- **Ogden 超弹性模型**：多项式形式的超弹性本构
- **广义 Maxwell 粘弹性**：支持多个 Maxwell 分支
- **可选 Kelvin-Voigt 体粘性**：额外的体积粘性阻尼
- **SDF Penalty 接触**：基于符号距离场的接触处理
- **正则化弹塑性摩擦**：tanh 过渡的静/动摩擦模型
- **能量统计**：完整的能量分解和投影修正追踪
- **自动微分**：支持对材料参数和初始状态求导

## 安装

```bash
# 安装依赖
pip install -r xengym/mpm/requirements.txt

# 或者直接安装 Taichi
pip install taichi>=1.6.0
```

## 快速开始

### 基本使用

```python
import numpy as np
import taichi as ti
from xengym.mpm import MPMConfig, MPMSolver

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 创建配置
config = MPMConfig()
config.grid.grid_size = (64, 64, 64)
config.grid.dx = 0.01
config.time.dt = 1e-4
config.time.num_steps = 1000

# 创建粒子
n_particles = 1000
positions = np.random.rand(n_particles, 3).astype(np.float32) * 0.3

# 创建求解器
solver = MPMSolver(config, n_particles)
solver.initialize_particles(positions)

# 运行仿真
for step in range(config.time.num_steps):
    solver.step()

    if step % 100 == 0:
        energy_data = solver.get_energy_data()
        print(f"Step {step}: E_kin={energy_data['E_kin']:.3e}")
```

### 使用配置文件

```python
from xengym.mpm import MPMConfig

# 从 JSON 加载
config = MPMConfig.from_json("config.json")

# 从 YAML 加载
config = MPMConfig.from_yaml("config.yaml")

# 保存配置
config.save_json("output_config.json")
```

### 自动微分

```python
from xengym.mpm import DifferentiableMPMSolver

# 创建可微求解器 (带 STE 和内存控制)
diff_solver = DifferentiableMPMSolver(
    config,
    n_particles,
    use_spd_ste=True,      # 启用 SPD 投影的 Straight-Through Estimator
    max_grad_steps=50      # 最大梯度步数 (控制显存)
)
diff_solver.initialize_particles(positions)

# 设置目标位置
target_positions = positions + 0.01
diff_solver.set_target_positions(target_positions)

# 计算梯度 (支持多种 loss 类型)
results = diff_solver.run_with_gradients(
    num_steps=50,
    loss_type='position',  # 'position', 'velocity', 'energy', 'com'
    requires_grad={'ogden_mu': True, 'ogden_alpha': True}
)

# Loss 类型说明:
# - 'position': 位置匹配
# - 'velocity': 速度匹配
# - 'energy': 动能匹配 (仅 E_kin = 0.5*m*v^2, 不含弹性/粘性/投影能量)
# - 'com': 质心匹配
# 注: 'energy' 默认为动能。如需总能量匹配，请参考文档实现自定义 loss

print("Loss:", results['loss'])
print("Gradient w.r.t. Ogden mu:", results['grad_ogden_mu'])
print("Gradient w.r.t. Ogden alpha:", results['grad_ogden_alpha'])
```

**⚠️ Autodiff 限制 (Taichi AD v1.7.4)**:

P2G/G2P kernels 包含 atomic scatter/gather 操作，Taichi autodiff **不支持**。`run_with_gradients()` 已被明确阻断以防止运行时崩溃。当前实现提供了完整的基础设施（loss field, needs_grad, tape flow），等待 Taichi 版本改进后即可启用。详见 `xengym/mpm/TAICHI_AUTODIFF_LIMITATIONS.md`。

可用功能:
- ✅ 所有 loss 类型的计算 (position/velocity/energy/com)
- ✅ Target validation 和清晰的错误信息
- ✅ SPD projection STE 支持
- ✅ Memory control (max_grad_steps)

阻塞功能 (需要 Taichi 版本升级或手动伴随方法):
- ❌ 梯度通过 P2G/G2P 反向传播
- ❌ 计算 loss 对材料参数的梯度
- ❌ 计算 loss 对初始状态的梯度

### Manual Adjoint 方法 (推荐)

为绕过 Taichi AD 的 atomic ops 限制，我们实现了手写伴随 (manual adjoint) 方法：

```python
from xengym.mpm import ManualAdjointMPMSolver, configure_gradient_mode

# [重要] 配置梯度模式 - 必须在 ti.init() 之前调用
configure_gradient_mode(
    use_numerical=True,       # True: 数值差分 (准确), False: 解析近似 (快但不完整)
    eps=1e-4,                 # 有限差分步长
    experimental_p_total=False,  # 实验性: 启用 P_total 数值梯度 (Maxwell/体粘性)
    max_particles=5000,       # 实验模式粒子数上限
    max_steps=500             # 实验模式步数上限
)

import taichi as ti
ti.init(arch=ti.gpu)

# 创建手动伴随求解器
solver = ManualAdjointMPMSolver(
    config,
    n_particles,
    max_grad_steps=100,      # 最大梯度追踪步数
    maxwell_needs_grad=False # 是否计算 Maxwell 参数梯度
)
solver.initialize_particles(positions)

# 设置目标位置
target_positions = final_positions
solver.set_target_positions(target_positions)

# 运行前向仿真 + 反向传播
results = solver.solve_with_gradients(
    num_steps=50,
    loss_type='position',  # 'position', 'velocity', 'kinetic_energy'
    requires_grad={
        'ogden_mu': True,
        'ogden_alpha': True,
        'initial_x': True,
        'initial_v': True
    }
)

print("Loss:", results['loss'])
print("∂L/∂μ:", results['grad_ogden_mu'])
print("∂L/∂α:", results['grad_ogden_alpha'])
print("∂L/∂x₀:", results['grad_initial_x'])
```

**支持矩阵:**

| 梯度类型 | 标准模式 | 实验 P_total 模式 | 备注 |
|---------|---------|------------------|------|
| **Ogden 参数** (μ, α) | ✅ 完整 | ✅ 完整 | 解析+数值验证 |
| **Ogden 体模量** (κ) | ✅ 完整 | ✅ 完整 | 解析梯度 |
| **初始状态** (x₀, v₀) | ✅ 完整 | ✅ 完整 | BPTT 传播 |
| **变形梯度** g_F | ⚠️ 仅 Ogden | ✅ 完整 P_total | 含 Maxwell/体粘性贡献 |
| **Maxwell 参数** (G, τ) | ❌ 需实验模式 | ✅ 完整 | G 从应力路径, τ 从内部变量更新 |
| **体粘性参数** (η_bulk) | ⚠️ 需实验模式 | ✅ 完整 | 从应力路径计算 |
| **规模限制** | 无 | 5000 粒子, 500 步 | 可配置 |

**⚠️ 重要**：含 Maxwell 分支的配置**必须启用实验模式**才能计算梯度：
```python
configure_gradient_mode(experimental_p_total=True)  # 在 ti.init() 之前调用
```

**⚠️ FR-4 合规状态**: 当前实现**部分满足** FR-4 要求
- ✅ Ogden 材料参数可微
- ✅ 初始状态可微
- ⚠️ **Maxwell 分支参数 (G, τ) 需启用实验模式** (有规模限制)
- ⚠️ **体粘性参数 (η_bulk) 需启用实验模式** (有规模限制)

**⚠️ Manual Adjoint 限制**:

1. **材料模型限制 (标准模式)**: 仅支持纯 Ogden 弹性模型的梯度
   - ❌ Maxwell 粘弹性分支 → `ValueError` (严格模式)
   - ❌ 体粘性 (bulk viscosity) → `ValueError` (严格模式)
   - 原因: 数值 `dP/dF` 仅对 Ogden 应力做差分，不含粘性贡献

2. **实验性 P_total 模式**: 完整支持 Maxwell/体粘性的 **g_F 梯度**
   - 启用: `configure_gradient_mode(experimental_p_total=True)`
   - **g_F 计算**: 对完整 P_total = P_ogden + P_maxwell + P_visc 数值差分 ✅
   - **Maxwell 参数梯度**: G 从应力路径计算, τ 从内部变量更新路径计算 ✅
   - **体粘性参数梯度**: η_bulk 从应力路径计算 ✅
   - **⚠️ 重要限制**: g_mu/g_alpha 仍**仅对 Ogden 部分**计算
   - 规模限制: 默认最大 5000 粒子、500 步 (可配置，solver 层强制拦截)
   - 性能: 计算开销大 (每粒子每步 18 次完整应力评估)，建议仅用于小规模验证

3. **配置时机**: `configure_gradient_mode()` 必须在 `ti.init()` 之前调用
   - Taichi 使用编译时常量 (`ti.static()`)，运行时无法更改

4. **性能开销**:
   - 标准模式: 每粒子每步 18 次 Ogden 应力评估
   - 实验模式: 每粒子每步 18 次完整 P_total 评估 (含 Maxwell 分支循环)

5. **Jacobian 钳位**: 固定为 (0.5, 2.0)，匹配前向传播，不可配置

**验证梯度**:

```python
# 使用有限差分验证解析梯度
verify_result = solver.verify_gradient_numerical(
    param_name='ogden_mu',
    param_idx=0,
    num_steps=10,
    eps=1e-4
)
print(f"解析: {verify_result['analytic']:.6f}")
print(f"数值: {verify_result['numerical']:.6f}")
print(f"相对误差: {verify_result['rel_error']:.2%}")
```

## 配置说明

### 网格配置 (GridConfig)

- `grid_size`: 网格分辨率 (nx, ny, nz)
- `dx`: 网格间距 (米)
- `origin`: 网格原点坐标

### 时间配置 (TimeConfig)

- `dt`: 时间步长 (秒)
- `num_steps`: 仿真步数
- `substeps`: 每步的子步数

### 材料配置 (MaterialConfig)

- `density`: 材料密度 (kg/m³)
- `ogden`: Ogden 超弹性参数
  - `mu`: 剪切模量列表
  - `alpha`: 指数列表
  - `kappa`: 体积模量
- `maxwell_branches`: Maxwell 分支列表
  - `G`: 剪切模量
  - `tau`: 松弛时间
- `enable_bulk_viscosity`: 是否启用体粘性
- `bulk_viscosity`: 体粘性系数

### 接触配置 (ContactConfig)

- `enable_contact`: 是否启用接触
- `contact_stiffness_normal`: 法向接触刚度
- `contact_stiffness_tangent`: 切向摩擦刚度
- `contact_stiffness`: 旧接触刚度 (向后兼容,会自动映射到 normal/tangent)
- `mu_s`: 静摩擦系数
- `mu_k`: 动摩擦系数
- `friction_transition_vel`: 摩擦过渡速度
- `K_clear`: 切向位移清理阈值 (摩擦迟滞计数器)
- `obstacles`: SDF 障碍物列表 (**新** - 支持 plane/sphere/box)

### SDF 障碍物配置 (SDFConfig) (**新**)

支持三种类型的 SDF 障碍物：

```python
# 平面 (默认地面)
{
    "sdf_type": "plane",
    "center": [0.0, 0.0, 0.0],      # 平面上的点
    "normal": [0.0, 0.0, 1.0],      # 法向量 (向外)
    "half_extents": [0.0, 0.0, 0.0] # 忽略
}

# 球体 (⚠️ 注意: half_extents[0] = 半径)
{
    "sdf_type": "sphere",
    "center": [0.5, 0.5, 0.5],      # 球心
    "normal": [0.0, 0.0, 1.0],      # 忽略
    "half_extents": [0.1, 0.0, 0.0] # half_extents[0] 是半径!
}

# 立方体 (轴对齐)
{
    "sdf_type": "box",
    "center": [0.3, 0.3, 0.1],      # 中心点
    "normal": [0.0, 0.0, 1.0],      # 忽略
    "half_extents": [0.1, 0.1, 0.05] # 各轴半长
}
```

**参数使用表：**
| 类型 | center | normal | half_extents |
|------|--------|--------|--------------|
| plane | 平面上一点 | 法向量 (单位) | 忽略 |
| sphere | 球心 | 忽略 | `[radius, 0, 0]` |
| box | 中心 | 忽略 | `[hx, hy, hz]` |

**向后兼容**：如果 `obstacles` 列表为空，默认使用 z=0 地面平面。

## CLI 使用

### 基本命令

```bash
# 查看帮助
python -m xengym.mpm.cli --help

# 运行下落测试 (CPU)
python -m xengym.mpm.cli --scene drop --arch cpu --output output_drop

# 运行 Maxwell 松弛测试 (GPU)
python -m xengym.mpm.cli --scene maxwell --arch gpu --output output_maxwell

# 运行摩擦测试
python -m xengym.mpm.cli --scene friction --arch gpu --output output_friction

# 使用自定义配置
python -m xengym.mpm.cli --scene drop --config my_config.json --output output_custom
```

### 可用场景

**基本场景:**

1. **drop** - 基本下落测试
   - 粒子立方体在重力下落地
   - 输出: `energy.csv`, `particles_*.npz`

2. **maxwell** - Maxwell 粘弹性松弛测试
   - 初始拉伸后的应力松弛
   - 验证 Maxwell 分支的粘弹性行为
   - 输出: `maxwell_relaxation.csv` (stretch_x, E_elastic, E_viscous_cum)

3. **friction** - 接触摩擦测试
   - 粒子块在地面上滑动
   - 验证法向/切向刚度和摩擦迟滞
   - 输出: `friction_curve.csv` (tangent_disp, tangent_vel, E_kin, E_elastic, E_proj_step, E_proj_cum)

**验证场景 (FR-5 覆盖):**

4. **uniaxial** - 单轴拉伸测试
   - 应力-应变响应 vs Ogden 理论曲线
   - 输出: `uniaxial_tension.csv`

5. **shear** - 纯剪切测试
   - 剪切响应验证
   - 输出: `pure_shear.csv`

6. **objectivity** - 客观性测试 (**新**)
   - 验证叠加刚体旋转后应力不变性
   - 输出: `objectivity.csv` (stress_no_rot, stress_rot, rel_diff)

7. **energy** - 能量守恒测试
   - 弹性仿真的能量守恒误差追踪
   - 输出: `energy_conservation.csv`

8. **energy_conv** - 能量收敛测试 (**新**)
   - 含投影能量追踪: E_kin, E_elastic, E_viscous_step/cum, E_proj_step, E_proj_cum
   - 验证能量平衡和 ΔE_proj_step vs E_viscous 关系
   - 输出: `energy_convergence.csv`

9. **gelslim** - GelSlim 黏滑测试 (**新**)
   - 切向力-位移曲线
   - stick-slip 与 incipient slip 行为观察
   - 输出: `gelslim_slip.csv` (统一格式: tangent_disp, tangent_vel, E_kin, E_elastic, E_proj_step, E_proj_cum)

10. **hertz** - Hertz 接触测试 (**新**)
    - 弹性球撞击: 压痕 vs 时间
    - 误差-步长/网格收敛验证
    - 输出: `hertz_contact.csv`

11. **all_validation** - 运行所有验证场景

```bash
# 运行单轴拉伸测试
python -m xengym.mpm.cli --scene uniaxial --output results/

# 运行所有验证场景
python -m xengym.mpm.cli --scene all_validation --output validation_output/

# 禁用严格模式 (允许不稳定配置继续运行)
python -m xengym.mpm.cli --scene drop --no-strict
```

### 绘图脚本 (**新**)

生成验证结果的标准曲线:

```bash
# 绘制所有 CSV 结果
python -m xengym.mpm.scripts.plot_validation --input validation_output/

# 绘制特定场景
python -m xengym.mpm.scripts.plot_validation --input validation_output/ --scene energy_conv
```

生成的图像:
- `uniaxial_tension.png`: 应力-应变 vs 理论
- `objectivity.png`: 旋转不变性验证
- `energy_convergence.png`: E_kin/E_elastic/E_viscous/E_proj 分解
- `gelslim_slip.png`: 切向力-位移曲线
- `hertz_contact.png`: 压痕深度 vs 时间

### 架构选项 (**新**)

- `--arch cpu`: CPU 后端
- `--arch gpu`: 自动选择 GPU (默认)
- `--arch cuda`: NVIDIA CUDA
- `--arch vulkan`: Vulkan (跨平台)

### 输出文件

所有场景输出 CSV 格式的能量/物理量曲线:

```csv
step,time,E_kin,E_elastic,...
0,0.0,1.23e-5,4.56e-3,...
10,1.0e-3,2.34e-5,4.55e-3,...
```

可用于后处理绘图 (matplotlib, pandas, etc.)

## 示例代码

查看 `xengym/mpm/examples/` 目录获取更多示例:

### 配置文件示例

**快速验证配置 (推荐用于冒烟测试):**

- `config_quick_validation.json` - 快速基础验证 (~100 粒子, ~100 步, 秒级运行)
- `config_quick_maxwell.json` - 快速 Maxwell 验证 (~50 粒子, ~100 步)

**完整演示配置:**

- `config_maxwell_demo.json` - Maxwell 粘弹性配置
- `config_friction_demo.json` - 接触摩擦配置
- `config_gelslim.json` - GelSlim 黏滑测试配置

### Python 示例

```bash
# 运行高级用法示例
python xengym/mpm/examples/advanced_usage.py
```

包含以下示例:
1. **Multi-term Ogden**: 4 项 Ogden 超弹性模型
2. **Maxwell Viscoelasticity**: 多分支 Maxwell 粘弹性
3. **Contact Friction**: 法向/切向刚度分离的接触摩擦
4. **Energy Analysis**: 完整的能量追踪和分析

## 项目结构

```
xengym/mpm/
├── __init__.py           # 模块导出
├── config.py             # 配置管理
├── fields.py             # 粒子和网格字段
├── decomp.py             # 线性代数分解
├── constitutive.py       # 本构模型
├── constitutive_gradients.py  # 本构梯度 (manual adjoint, 含实验性 P_total 模式)
├── contact.py            # 接触和摩擦
├── mpm_solver.py         # 主求解器
├── autodiff_wrapper.py   # 自动微分封装
├── manual_adjoint.py     # 手动伴随核心 kernels
├── manual_adjoint_solver.py  # 手动伴随求解器
├── stability.py          # 稳定性检查 (Drucker, 时间步)
├── validation.py         # 验证场景 (FR-5 覆盖)
├── cli.py                # CLI 入口 (含严格模式)
├── examples/             # 示例代码和配置
│   ├── advanced_usage.py
│   ├── manual_adjoint_example.py  # Manual Adjoint 示例
│   ├── config_quick_validation.json  # 快速冒烟测试
│   ├── config_quick_maxwell.json     # 快速 Maxwell 测试
│   ├── config_gelslim.json
│   ├── config_maxwell_demo.json
│   └── config_friction_demo.json
├── scripts/              # 后处理脚本
│   ├── plot_validation.py  # 绘图脚本 (生成标准曲线)
│   └── plot_dt_convergence.py  # dt 收敛验证 (FR-5 E_proj/E_viscous)
├── gradient_mode_tests.py # 梯度模式单元测试 (手动运行避免 pytest 收集)
├── requirements.txt      # 依赖列表
└── README.md             # 本文档
```

## 求解流程

MPM 求解器的每一步包含以下阶段:

1. **clear_grid**: 清空网格字段 (保留 grid_ut)
2. **p2g**: 粒子到网格传输
3. **grid_op**: 网格操作 (施加力、边界条件、接触)
4. **g2p**: 网格到粒子传输
5. **update_F_and_internal**: 更新变形梯度和内变量
   - 使用 `ti.static()` 统一 STE/非STE 路径，编译时分支选择
   - STE (Straight-Through Estimator) 用于 autodiff 模式下的 SPD 投影
   - 默认路径: 非STE (标准 SPD 投影)
6. **reduce_energies**: 归约能量统计
7. **cleanup_ut**: 清理切向位移 (基于迟滞计数器)

## 能量统计

求解器追踪以下能量:

- `E_kin`: 动能
- `E_elastic`: 弹性能
- `E_viscous_step`: 本步粘性耗散
- `E_viscous_cum`: 累积粘性耗散
- `E_proj_step`: 本步投影修正
- `E_proj_cum`: 累积投影修正

## FR-5 验证需求对照表

| FR-5 需求 | 验证场景 | CLI 命令 | 输出文件 | 状态 |
|-----------|---------|----------|---------|------|
| 单轴拉伸 vs Ogden 理论 | `uniaxial` | `--scene uniaxial` | `uniaxial_tension.csv` | ✅ |
| 纯剪切响应 | `shear` | `--scene shear` | `pure_shear.csv` | ✅ |
| 应力客观性 (刚体旋转不变) | `objectivity` | `--scene objectivity` | `objectivity.csv` | ✅ |
| 能量守恒 (弹性仿真) | `energy` | `--scene energy` | `energy_conservation.csv` | ✅ |
| 能量分解 (含 ΔE_proj) | `energy_conv` | `--scene energy_conv` | `energy_convergence.csv` | ✅ |
| Maxwell 应力松弛 | `maxwell` | `--scene maxwell` | `maxwell_relaxation.csv` | ✅ |
| Stick-Slip 摩擦 | `friction` | `--scene friction` | `friction_curve.csv` | ✅ |
| GelSlim 黏滑 | `gelslim` | `--scene gelslim` | `gelslim_slip.csv` | ✅ |
| Hertz 接触收敛 | `hertz` | `--scene hertz` | `hertz_contact.csv` | ✅ |
| **dt 收敛验证** | 专用脚本 | 见下方 | `dt_convergence.csv/png` | ✅ |

### dt 收敛验证 (FR-5 关键指标)

验证 `|E_proj_cum| / E_viscous_cum` 随 dt 的一阶收敛性：

```bash
# 运行 dt 收敛研究
python -m xengym.mpm.scripts.plot_dt_convergence --output results/

# 自定义 dt 因子
python -m xengym.mpm.scripts.plot_dt_convergence --dt-factors 1.0 0.5 0.25 0.125
```

**期望结果**：
- log-log 图中斜率 ≈ 1.0 (一阶收敛)
- 比值 `|E_proj_cum| / E_viscous_cum` < 1% (对于合理的 dt)

**⚠️ 前提条件**：
- **必须启用 Maxwell 粘弹性分支**，否则 `E_viscous_cum ≈ 0`，收敛比无法计算
- 仿真时长需足够长，让粘性耗散累积到可测量水平
- 如果脚本报告"数据无效"，检查材料配置中是否包含 `maxwell_branches`

**输出**：
- `dt_convergence.png`: log-log 收敛曲线
- `dt_convergence.csv`: 各 dt 下的能量数据

## 注意事项

1. **时间步长约束**：需要满足 CFL 条件和粘性时间尺度约束
2. **网格分辨率**：粒子间距应为网格间距的 0.5 倍左右
3. **材料参数**：Ogden 参数需满足 Drucker 稳定性条件
4. **GPU 内存**：大规模仿真需要足够的 GPU 内存

## 参考文献

- Jiang et al., "The Material Point Method for Simulating Continuum Materials", SIGGRAPH Course Notes 2016
- Stomakhin et al., "A Material Point Method for Snow Simulation", SIGGRAPH 2013
- Ogden, "Non-Linear Elastic Deformations", 1984

## 许可证

本项目遵循与 xenmpm 主项目相同的许可证。
