## Context

The project already:
- Implements a full VHE-MLS-MPM solver with Ogden + Maxwell + Kelvin-Voigt and energy accounting.
- Provides a manual adjoint implementation (`ManualAdjointMPMSolver`, `manual_adjoint.py`) to bypass Taichi’s autodiff limitations on atomic P2G/G2P.
- Includes gradient-mode configuration in `constitutive_gradients.py` with experimental P_total numerical mode and guardrails.
- Ships a rich validation suite (uniaxial, shear, objectivity, energy, stick–slip, Hertz) and targeted gradient tests.

Specs from previous changes establish:
- Manual adjoint requirements (grid normalisation backward, APIC affine term gradients, SPD STE, configurable Maxwell gradients).
- Guardrails for unsupported Taichi AD paths and experimental numerical modes.
- Verification scenes and convergence checks.

The gap is that “gradient quality” is only loosely defined, and some analytic gradients (especially g_F) are known to be incomplete, leading to relatively loose tolerances and weaker guarantees for calibration use-cases.

## Goals / Non-Goals

Goals:
- Treat manual adjoint as the primary, supported path for MPM gradients.
- Improve analytic coverage in constitutive gradients (g_F, g_μ, g_α) so that tests can target tighter tolerances.
- Introduce tiered gradient-accuracy requirements and ensure tests are aligned with those tiers.
- Extend verification outputs to surface gradient-quality metrics in a reusable, script-friendly way.

Non-Goals:
- No attempt to make Taichi Tape support P2G/G2P atomics.
- No change to the high-level solver algorithm, contact model, or energy accounting beyond correctness fixes.
- No mandate for a fully analytic SPD-projection gradient; STE remains acceptable when documented and constrained.

## Decisions

### Decision 1: Manual adjoint as the canonical MPM gradient path

- The “complete autodiff” story for MPM will be defined in terms of the manual adjoint solver, not Ti.Tape over the forward kernels.
- Requirements and tests will be written assuming:
  - Tape-based autodiff is blocked on unsupported paths and surfaces clear error messages.
  - Manual adjoint is the only path expected to deliver gradients for production calibration scenarios.

Alternatives:
- Try to re-architect P2G/G2P to avoid atomics and use Tape.
- Use external autodiff frameworks (JAX, PyTorch) for MPM.

These are left as future work due to higher complexity and migration cost.

### Decision 2: Tiered gradient-accuracy targets

We introduce three tiers:
- **Tier A (constitutive/stress level)**: small systems where analytic gradients should be very accurate.
- **Tier B (small MPM toy scenes)**: full P2G/G2P/manual adjoint pipeline with modest particle counts and steps.
- **Tier C (high-deformation end-to-end scenes)**: numerically challenging but still expected to produce reasonable gradients.

Each tier will have explicit thresholds for:
- Relative error between analytic/adjoint and finite-difference gradients.
- Cosine similarity between gradient vectors (where applicable).

These tiers will be tied to specific tests and documented in the verification spec.

### Decision 3: Constitutive gradients as the primary precision lever

- The main place to significantly improve gradient precision is the constitutive layer:
  - Ensure `compute_ogden_stress_with_gradients` covers the full dP/dF (including terms currently approximated or omitted).
  - Use higher precision or numerically stable formulations where needed.
- Manual adjoint kernels will be updated only as required to:
  - Use the improved constitutive gradients.
  - Propagate gradients consistently (no silent approximations beyond those documented in specs, e.g. SPD STE).

Alternatives:
- Tuning only finite difference epsilons and test tolerances without improving math.
- Expecting large improvements purely from numerical tweaks.

These alternatives are insufficient for users who want robust calibration gradients.

### Decision 4: Gradient-quality reporting in the verification suite

- Extend existing gradient tests or add new ones so that:
  - They output per-parameter gradient error and cosine similarity metrics.
  - These metrics can be inspected offline (e.g. CSVs used by plotting scripts).
- Keep this lightweight and focused on key parameters and initial states rather than exhaustive coverage.

## Risks / Trade-offs

- **Risk:** Tightening tolerances may make some tests flaky on different hardware or Taichi versions.
  - **Mitigation:** Use tiered tolerances; keep high-deformation tests slightly more relaxed; rely on small, controlled scenes for strict checks.
- **Risk:** Implementing more complete analytic g_F may introduce subtle bugs.
  - **Mitigation:** Maintain numerical fallback paths behind feature flags; add direct constitutive-level regression tests; rely on finite-difference comparisons as safety net.
- **Risk:** Manual adjoint code is already complex; adding more logic may hurt maintainability.
  - **Mitigation:** Limit changes to clearly-scoped areas (constitutive gradients, a few adjoint kernels), and back them with targeted tests.

### Decision 5: Tier thresholds (calibrated)

Based on current numerical experiments and expected Taichi float32 behaviour, the following thresholds are adopted:
- Tier A (constitutive/stress level):
  - Relative error ≤ 0.01
  - Cosine similarity ≥ 0.99
- Tier B (small MPM toy scenes):
  - Relative error ≤ 0.05
  - Cosine similarity ≥ 0.95
- Tier C (high-deformation end-to-end scenes):
  - Relative error ≤ 0.50
  - Cosine similarity ≥ 0.80

These thresholds are targets rather than hard physical limits; if specific scenes cannot meet them due to well-understood numerical reasons, this MUST be documented in the corresponding tests and/or verification scripts.

## Migration Plan

1. Introduce gradient-accuracy tiers and centralise tolerances in test utilities (already partially done).
2. Update specs for autodiff behaviour and verification suite to:
   - Declare manual adjoint as the canonical gradient path.
   - Encode tiered precision requirements and scenarios.
3. Implement constitutive gradient improvements and adjust tests to target the new tiers.
4. Extend manual adjoint tests and verification scenes to emit gradient-quality metrics.
5. Iterate on tolerances if CI or numerical experiments show instability, within the bounds set by the spec.

## Open Questions

- How many parameters and initial states should be covered by Tier B/C tests without bloating runtime?
