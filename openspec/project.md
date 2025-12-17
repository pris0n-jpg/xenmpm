# Project Context

## Purpose
Xense Simulator：为软体触觉/机器人场景提供可视化与传感器仿真（左/右触觉传感器、URDF 机械臂、可加载外部对象），后续将落地 Taichi 版可微 VHE-MLS-MPM 求解器以支持高保真软体/接触模拟与验证。

## Tech Stack
- 语言：Python 3.9+
- 数值/基础：numpy<=1.26.4
- 引擎/图形：xensesdk（tb UI、Matrix4x4、Qt 工具）
- 未来扩展：Taichi（计划用于 VHE-MLS-MPM），cypack[build] 用于打包
- CLI：entry point `xengym-demo`

## Project Conventions

### Code Style
- 遵循 PEP 8，四空格缩进，snake_case 函数/变量，PascalCase 类，常量全大写。
- Docstring 简洁英文；必要内联注释用于解释非显式数学/校准逻辑，新增注释用中文写意图与约束。
- 避免通配 import，显式相对导入。
- 文件编码 UTF-8，无 BOM；默认 ASCII 内容，除非已有非 ASCII。

### Architecture Patterns
- 包结构：`xengym/` 核心；`render/` 场景与 UI，`fem/` 材料模型，`ezgym/` 机器人与资产包装；`main.py` 提供 `xengym-demo` 入口。
- 资产：`xengym/assets/` 存储 STL/传感器资源，需小体积并记录来源。
- 校准/工具：`calibration/` 后处理与数据助手，与 `CALIBRATION_DATA_FORMAT.md` 对齐。
- 示例与数据流：`example/` 提供可运行 demo 与数据管线参考；`quick_test.py`、`example/test*.py` 用作轻量冒烟/回归。
- 计划中的数值层：mpm_solver/constitutive/contact/decomp/fields/autodiff（Taichi），上层场景/验证解耦于 SDF/配置。

### Testing Strategy
- 冒烟：`python quick_test.py`
- 回归/示例：`python example/demo_main.py`, `python example/data_collection.py`, `python example/test_*.py`
- 新特性：在 `quick_test.py` 或相邻 example 下新增 `test_<feature>.py`；仿真/能量/摩擦验证输出 CSV/图像并检查曲线。

### Git Workflow
- 提交消息格式：`<type>: <message>`（如 `feat: add diff image api`）
- 每次提交前确保可运行冒烟/主要脚本；PR 包含行为总结、测试证据、资产变更位置、关联 issue、UI/传感器输出截图或深度场对比。

## Domain Context
- 机器人与软体触觉传感器（如 GelSlim）仿真；需要可视化、接触与能量分解。
- 目标引擎：显式 MLS-MPM/APIC，Ogden+Maxwell 粘弹性，正则化弹塑性摩擦；支持自动微分与能量一致性验证。
- 资产管理：仅提交必要的处理后资源，避免大型原始数据。

## Important Constraints
- 安全第一：避免未验证的外部执行/路径注入；遵循 DRY/SOLID/KISS/YAGNI。
- 不提交大体积资产；记录新资源来源。
- 时间步与材料参数需做稳定性检查（CFL、粘性时间尺度、接触刚度、Drucker 检查）；J 裁剪与 SPD 投影避免 NaN（针对 MPM 计划）。
- 输出/接口需保持向后兼容并可配置。

## External Dependencies
- `xensesdk`（tb UI、Qt 工具、Matrix4x4）
- `numpy<=1.26.4`
- `cypack[build]`（构建/打包）
- 操作系统依赖 Qt/图形后端（随 xensesdk 提供）
