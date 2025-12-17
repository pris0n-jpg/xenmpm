## MODIFIED Requirements

### Requirement: Gradient Verification Tiers
The verification suite SHALL define explicit gradient-accuracy tiers and map them to specific tests:
- **Tier A (constitutive/stress level):**
  - Small systems (single-point constitutive and stress tests).
  - Target relative error ≤ 0.01 and cosine similarity ≥ 0.99 for gradients w.r.t. Ogden parameters.
- **Tier B (small MPM toy scenes):**
  - Manual-adjoint MPM tests with modest particle counts and steps.
  - Target relative error ≤ 0.05 and cosine similarity ≥ 0.95 for key parameters/initial states.
- **Tier C (high-deformation end-to-end scenes):**
  - More numerically challenging scenes.
  - Allow relaxed but bounded tolerances; for key metrics target relative error ≤ 0.50 and cosine similarity ≥ 0.80 where applicable,
  - Tests SHALL still report measured errors and cosine similarity.

Each gradient test SHALL declare its tier and SHALL be implemented to meet the corresponding thresholds where numerically feasible.

#### Scenario: Constitutive/stress gradient tests (Tier A)
- **WHEN** running constitutive and stress gradient tests (e.g., dP/dμ) on single-point configurations,
- **THEN** the reported analytic/manual-adjoint gradients SHALL match finite-difference gradients within Tier A thresholds,
- AND failures SHALL be treated as defects in the constitutive gradient implementation.

#### Scenario: Manual adjoint small-scene tests (Tier B)
- **WHEN** running manual adjoint MPM tests on small toy scenes (limited particles/steps),
- **THEN** gradients for selected parameters and initial states SHALL meet Tier B thresholds,
- AND tests SHALL clearly report measured relative error and cosine similarity.

#### Scenario: High-deformation validation (Tier C)
- **WHEN** running high-deformation end-to-end gradient validation scenes,
- **THEN** the suite SHALL:
  - Enforce Tier C thresholds,
  - Emit gradient-quality metrics,
  - And document any known numerical limitations impacting those metrics.

### Requirement: Gradient-Quality Reporting
The verification suite SHALL emit gradient-quality metrics for key gradient tests, including at least:
- Relative error between analytic/manual-adjoint and finite-difference gradients.
- Cosine similarity between gradient vectors where applicable.

These metrics SHALL be written to CSV or structured logs that can be consumed by plotting or analysis scripts.

#### Scenario: Gradient verification script output
- **WHEN** running the gradient verification scripts on toy scenes,
- **THEN** they SHALL output per-parameter gradient error and cosine similarity,
- AND these outputs SHALL be usable to diagnose regressions or improvements in gradient quality over time.

#### Implementation Note: Mapping to Test Constants
These tiers map to the following constants in `xengym/mpm/tests/conftest.py`:
- Tier A → `GRADIENT_RTOL_STRICT` (target value: 0.01).
- Tier B → `GRADIENT_RTOL_NORMAL` (target value: 0.05).
- Tier C → `GRADIENT_RTOL_LOOSE` (target value: 0.50).

