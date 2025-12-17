## ADDED Requirements

### Requirement: 本构与客观性验证
系统必须（SHALL）提供可运行脚本以验证 Ogden/VHE 模型：单轴拉伸应力-应变对比解析解，纯剪切响应与叠加刚体旋转的客观性检查，应力松弛的 Δt 收敛（Lie-Trotter + Projection 一阶）。

#### Scenario: 拉伸/剪切/松弛验证
- **WHEN** 运行对应验证脚本（拉伸、纯剪切+刚体旋转、step strain 松弛）
- **THEN** 输出曲线显示：拉伸应力接近解析解，纯剪切叠加旋转后主应力不变（客观性），松弛曲线随 Δt 减小呈一阶收敛。

### Requirement: 能量分解与投影修正对比
系统必须（SHALL）在能量验证脚本中输出 E_kin、E_elastic、E_viscous_cum、E_proj_step、E_proj_cum，并提供图表对比 |ΔE_proj_step| 与每步粘性耗散、|E_proj_cum|/E_viscous_cum 随时间与 Δt 的曲线，展示投影误差可控且一阶收敛。

#### Scenario: 能量一致性报告
- **WHEN** 运行能量验证脚本并开启能量输出
- **THEN** 生成时间序列与图表，|E_proj_cum|/E_viscous_cum 保持在 ~1% 量级并随 Δt 减小趋近 0，闭合系统能量不增。

### Requirement: Stick–Slip 与 incipient slip 验证
系统必须（SHALL）提供块-板 stick–slip 场景与 GelSlim 风格 incipient slip 场景：输出切向力-位移曲线、接触区光流/位移场，呈现静摩擦线性区、峰值后动摩擦平台及局部微滑移再到宏观滑移的模式。

#### Scenario: 摩擦曲线与光流模式
- **WHEN** 运行块-板与 GelSlim 场景脚本
- **THEN** 切向力曲线出现峰值后下降至动摩擦平台，光流/位移场展示 incipient slip 的边缘微滑移与宏观滑移阶段。

### Requirement: 空间网格收敛性
系统必须（SHALL）包含 Hertz 接触或弹性球撞击等无粘性算例，使用 Δx、Δx/2、Δx/4 三组网格，输出接触力-时间或位移场 L2 误差，在双对数坐标下呈现收敛阶（内部近二阶、接触邻域可降阶但保持趋势）。

#### Scenario: 收敛曲线生成
- **WHEN** 运行网格收敛脚本
- **THEN** 生成误差-网格尺寸曲线，显示预期收敛阶，且脚本可复现实验参数（关闭粘性以匹配理论）。
