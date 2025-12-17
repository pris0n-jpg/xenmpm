## ADDED Requirements

### Requirement: 手工伴随支持可微 MPM
系统必须（SHALL）为 P2G/G2P/GridOps/F 更新/Maxwell/能量求和提供前向+手工伴随 backward，实现材料参数与初始状态的梯度，并绕过 Taichi 对原子操作反向的限制。

#### Scenario: 可微模式运行
- **WHEN** 用户在可微模式下运行 MPM 并调用梯度接口
- **THEN** 系统使用手工伴随路径返回材料参数/初始状态的非零梯度，且不会触发 Taichi “Not supported” 错误。

### Requirement: Grid 归一化反向 (v = P/M)
系统必须（SHALL）在 Grid Ops backward 中显式实现 v=P/M 的反向：g_P += g_v / M，g_M += -(g_v · v) / M，确保质量分布对 loss 的影响被捕获。

#### Scenario: GridOps Backward
- **WHEN** 反向计算中存在对网格速度的梯度 g_v
- **THEN** 系统按上述公式将梯度分配到 g_P、g_M，再交由 P2G backward。

### Requirement: APIC 仿射项对位置的导数
系统必须（SHALL）在 P2G backward 中包含 APIC 仿射项 C_p(x_I - x_p) 对 x_p 的导数（-C_p），除权重导数之外，避免粒子位置梯度缺失。

#### Scenario: APIC 梯度
- **WHEN** 开启 APIC/MLS-MPM 仿射动量
- **THEN** P2G backward 必须包含 -C_p 对 x_p 的贡献，粒子位置梯度不为 0。

### Requirement: SPD 投影策略 (STE)
系统必须（SHALL）在形变梯度的 SPD 投影中采用直通估计（STE）：前向执行 SPD 保证稳定，反向视 SPD 为恒等，g_F_raw += g_F_spd，并提供 SPD 触发统计与梯度偏差风险说明。

#### Scenario: SPD 开启
- **WHEN** 粒子触发 SPD 投影
- **THEN** 反向按 STE 传递梯度，并可记录投影触发比例用于调试。

### Requirement: Maxwell 内变量伴随（可配置）
系统必须（SHALL）为 Maxwell 内变量提供可配置的伴随：配置开启时存储/重算 b_e 状态并回传 b_e、τ 梯度；关闭时不存储时间序列且相关梯度为 0。

#### Scenario: Maxwell 梯度开启
- **WHEN** 用户启用 Maxwell 梯度
- **THEN** backward 正确传递 b_e 与 τ 梯度；若关闭则不存储状态、梯度为 0。

### Requirement: Loss 定义与一致性
系统必须（SHALL）提供位置、速度、总能量（动能+弹性+粘性+投影）等可微 loss；默认能量 loss 语义需在接口/文档中声明，与实现一致；总能量 loss 可选或提供自定义指引。

#### Scenario: 总能量 loss
- **WHEN** 用户选择总能量 loss
- **THEN** 系统按定义计算总能量并传递梯度；选择动能 loss 时需在文档中明确范围。

### Requirement: 接口与模式防护
系统必须（SHALL）区分非可微模式与手工伴随可微模式：可微模式走手工伴随；非可微模式继续阻断并提示不可用，禁止直接用 Tape 包 step。

#### Scenario: 模式切换
- **WHEN** 用户切换可微模式
- **THEN** 系统使用对应路径；不支持场景下给出清晰错误与替代方案。

### Requirement: 验证与梯度精度
系统必须（SHALL）提供数值梯度对比（有限差分）与专项测试，满足精度阈值：相对误差 < 1e-4，方向余弦 > 0.99（小规模场景）；记录性能/显存与设计估算同量级（误差<2x）。

#### Scenario: 梯度对比
- **WHEN** 对 x0/v0/材料参数（及 τ 若启用）进行有限差分对比
- **THEN** 手工伴随梯度满足上述精度阈值，否则视为缺陷。
