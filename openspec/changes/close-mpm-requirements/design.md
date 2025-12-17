# Design Notes: close-mpm-requirements

## Scope boundaries
- Focus on aligning implementation with V2.2 FR/NFR: validation coverage, stability guardrails, energy outputs, and practical autodiff behavior.
- No large refactor of core solver; prefer additive scenes/scripts/guardrails.
- Autodiff: keep KISS/YAGNI—offer small-scale, optional P_total numerical gradients; keep pure Ogden fast path; block unsafe configurations by default.

## Verification suite
- Scenes (small particle counts by default):
  - 单轴拉伸：应力-应变对 Ogden 理论。
  - 纯剪切+客观性：对比叠加刚体旋转后的应力一致性。
  - 能量守恒/投影收敛：跟踪 E_kin/E_elastic/E_viscous*/ΔE_proj_step/E_proj_cum，收敛随 Δt/网格。
  - GelSlim/incipient slip：切向力-位移曲线、stick-slip 与 incipient slip 观察。
  - Hertz/弹性球撞击：误差-步长/网格收敛。
- Each scene exposes CLI flag and config template; outputs CSV with required metrics; companion plotting scripts generate mandated curves.

## Stability guardrails
- Default-on checks:
  - Drucker-type Ogden stability (代数 + 路径扫描) with strict/warn modes.
  - Time step/contact stiffness/viscous time-scale guidance; warn or block based on severity.
- CLI/config toggles: allow override for research, but default strict to avoid invalid runs.

## Energy/output completeness
- Standardize CSV columns: ΔE_proj_step, E_proj_cum, E_viscous_step/cum, E_kin, E_elastic, contact tangential force/displacement, convergence metrics.
- Plotting helpers: stress-strain, objective stress overlay, energy vs time, ΔE_proj_step vs viscous, convergence curves.

## Autodiff / manual adjoint
- Modes:
  1) Pure Ogden (fast analytical+numerical existing path).
  2) Optional “full P_total numerical gradient” for Maxwell/体粘性 on small particle counts; clearly marked experimental, with scale guard (particle/step caps) and warnings.
  3) Strict block remains default for incompatible configs.
- Gradient validation scripts: compare analytic/hand-adjoint vs finite difference for selected params and initial states on toy scenes.

## Compat & risks
- Maintain backward compatibility: new checks default on but configurable; outputs additive.
- Performance: numerical P_total gradients gated by size; examples keep low particle counts; document costs.
- Tooling: keep SVD/Eigen abstractions untouched; new work isolated in scenes, guardrails, and wrapper scripts.
