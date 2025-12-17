## ADDED Requirements

### Requirement: 粘弹性与摩擦验证场景
系统必须（SHALL）提供可运行的验证用例涵盖 Maxwell 松弛与摩擦曲线：至少包含（1）粘性松弛测试输出应力-时间与能量曲线，（2）摩擦 stick-slip 测试输出切向力-位移曲线与接触状态，支持配置导入。

#### Scenario: Maxwell 松弛验证
- **WHEN** 运行粘性松弛验证脚本
- **THEN** 输出应力随时间衰减曲线和能量分解（含 E_viscous/E_proj），可用于检查松弛速率与耗散。

#### Scenario: 摩擦曲线验证
- **WHEN** 运行摩擦验证脚本
- **THEN** 输出切向力-位移曲线展示 stick–slip，记录接触掩码/切向位移以便分析。

### Requirement: CLI 支持架构与场景选择
系统必须（SHALL）在 CLI 中提供 CPU/GPU 选择和场景枚举（至少包含 drop、Maxwell 验证、摩擦验证），并在运行后导出能量与曲线数据（CSV/npz）。

#### Scenario: CLI 可选架构
- **WHEN** 用户通过 CLI 指定 arch=cpu 或 arch=gpu 及场景
- **THEN** 仿真在指定后端运行，并在输出目录生成所选场景的能量/曲线数据文件。
