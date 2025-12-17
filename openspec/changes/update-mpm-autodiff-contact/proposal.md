# Change: 升级 MPM 自动微分与接触参数化

## Why
- 现有 autodiff 封装未生成可微计算图，材料/初始状态梯度不可用。
- Ogden 本构硬编码 2 项，与配置上限不一致，需支持 1–4 项并做显式校验。
- 接触摩擦法向/切向刚度耦合，缺乏独立参数，难以稳定调参。
- 验证/工具需覆盖粘弹性与摩擦，以确保修复可验证。

## What Changes
- 重构 autodiff：使用 Taichi 可微 field/needs_grad，提供可配置的 kernel 版 loss 钩子，输出有效梯度。
- Ogden 支持 1–4 项：新增通用应力计算与配置校验，超限显式报错/截断。
- 接触参数拆分：独立法向/切向刚度（可选阻尼占位），保持旧配置自动兼容。
- 验证与工具：补充 Maxwell/摩擦/能量验证用例，CLI 支持 CPU/GPU 选择与场景选择。

## Impact
- 影响规格：`specs/vhe-mpm-solver/spec.md`、`specs/vhe-validation/spec.md`
- 影响代码：`autodiff_wrapper.py`、`constitutive.py`、`config.py`、`contact.py`、`mpm_solver.py`、`validation.py`、`cli.py`
