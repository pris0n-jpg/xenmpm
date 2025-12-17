# Change: 为 MPM 实现手工伴随以支持可微仿真

## Why
- Taichi AD 无法对 P2G/G2P 原子操作反向，autodiff 接口被阻断，无法对材料参数或初始状态求梯度。
- 需要基于手工伴随实现可微 MPM（前向 + 自定义反向），解除当前不可用状态。

## What Changes
- 设计并实现 P2G/G2P 的手工伴随（前向/反向 kernel），避免依赖 Taichi 对原子反向的限制。
- 明确 loss 入口与梯度输出（材料参数、初始状态），提供可配置的可微 loss 集。
- 调整 CLI/API 与文档：开放可用的可微仿真路径，保留不支持路径的防护和提示。
- 增加验证：梯度数值对比（finite-diff 或对照案例）、能量/摩擦场景的可微验证。

## Impact
- 规格：`specs/vhe-mpm-solver/spec.md`（可微仿真能力）、`specs/vhe-validation/spec.md`（可微验证场景）
- 代码（实施阶段）：P2G/G2P 前后向 kernel，autodiff wrapper，CLI/文档/示例同步
