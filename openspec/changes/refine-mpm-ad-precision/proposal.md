## Summary

Improve the MPM autodiff story by:
- Treating the manual-adjoint solver as the primary, “complete” gradient provider for MPM.
- Tightening and formalising gradient-accuracy requirements at both constitutive/stress level and small MPM scenes.
- Extending the verification suite to report gradient quality with explicit tolerances instead of ad-hoc per-test thresholds.

This proposal does **not** attempt to make Taichi’s Tape support P2G/G2P atomics; instead it focuses on making the existing manual adjoint path mathematically more complete and numerically better behaved, while keeping the existing guardrails for unsupported AD paths.

## Motivation

- Current OpenSpec changes (`add-mpm-manual-adjoint`, `update-mpm-ad-guardrails`, `close-mpm-requirements`) define:
  - A manual adjoint design for full MPM gradients.
  - Strong guardrails that safely block unsupported Taichi AD paths.
  - High-level requirements for gradient verification and experimental P_total numerical modes.
- The implementation now includes:
  - `ManualAdjointMPMSolver` and a full adjoint pipeline (`manual_adjoint.py`, `manual_adjoint_solver.py`).
  - Gradient-mode configuration (`constitutive_gradients.configure_gradient_mode`, `validate_gradient_mode`) plus tests.
  - A rich set of gradient tests (`tests/test_manual_adjoint.py`, `tests/test_constitutive_gradients.py`, `tests/test_stress_gradients.py`, `tests/test_high_deformation.py`, `tests/test_gradient_mode.py`).
- However:
  - Constitutive/stress gradients still rely partly on numerical approximations or incomplete analytic paths (missing terms in g_F).
  - Gradient tolerances are relatively loose (10–50%) and encoded per-test instead of being specified as tiers.
  - The specs talk about “pass numerical checks” without stating explicit error/cosine thresholds per level of the pipeline.

Users now want a more “complete autodiff” experience (within Taichi’s constraints) and higher-confidence gradients for optimisation/calibration tasks.

## Goals

- Define manual adjoint as the first-class autodiff path for MPM, consistent with existing guardrails.
- Introduce explicit, tiered gradient-accuracy requirements (strict/normal/relaxed) and map them to:
  - Constitutive and stress-gradient tests.
  - Manual-adjoint MPM tests on small toy scenes.
  - High-deformation end-to-end gradient checks.
- Require improved analytic coverage in constitutive gradients (especially g_F) where this materially impacts accuracy.
- Require the verification suite to emit gradient-quality metrics (relative error, cosine similarity) for key parameters/initial states, not just “pass/fail”.

## Non-Goals

- Do **not** remove or bypass existing autodiff guardrails for Taichi Tape on unsupported atomic paths.
- Do **not** mandate a fully analytic SPD-projection gradient; STE remains acceptable, though its assumptions must be documented and tested.
- Do **not** change the overall MPM algorithm, contact model, or energy accounting beyond what is needed for gradient correctness.

## Scope

In scope:
- Spec changes for:
  - Autodiff / manual adjoint behaviour (capability: autodiff-behavior).
  - Gradient verification and tests (capability: verification-suite).
- Implementation work in:
  - `xengym/mpm/constitutive_gradients.py` and related tests.
  - `xengym/mpm/manual_adjoint*.py` and MPM gradient tests.
  - Test utilities (`xengym/mpm/tests/conftest.py`) and verification scripts as needed.

Out of scope (future work):
- Re-architecting P2G/G2P to be Tape-friendly without manual adjoint.
- Porting the MPM solver to a different autodiff framework (e.g., JAX, PyTorch).

## Open Questions

- What exact numeric thresholds (relative error, cosine similarity) are acceptable at each level (constitutive vs full MPM) given Taichi’s float32 numerics?
- How aggressive should we be in tightening current relaxed thresholds in high-deformation tests without making CI flaky?
- Should gradient-quality metrics be emitted as separate CSVs under the verification suite, or embedded into existing outputs?

