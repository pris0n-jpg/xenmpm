## MODIFIED Requirements

### Requirement: Autodiff / Manual Adjoint Behavior
The solver SHALL expose differentiated simulation for material parameters and initial states using a clearly defined combination of:
- Manual adjoint MPM gradients (primary path for all supported configs).
- Guarded Taichi AD paths (only where fully supported and explicitly enabled).

It SHALL:
- Treat the manual adjoint solver as the canonical way to obtain MPM gradients in production.
- Keep Taichi Tape-based autodiff behind existing guardrails for unsupported atomic paths.
- Provide gradient verification scripts comparing manual adjoint vs finite difference on toy scenes.

#### Scenario: Pure Ogden mode (manual adjoint)
- **WHEN** a pure Ogden configuration is used and gradients are requested via the manual adjoint interface,
- **THEN** gradients with respect to initial states and Ogden parameters are computed without blocking for small scenes,
- **AND** they satisfy Tier A/B gradient-accuracy thresholds defined in the verification suite (constitutive and small MPM tests).

#### Scenario: Maxwell/bulk enabled with experimental P_total mode
- **WHEN** Maxwell branches and/or bulk viscosity are enabled and experimental P_total numerical mode is on,
- **THEN** the solver SHALL:
  - Use manual adjoint for the Ogden component,
  - Use numerical P_total gradients within configured scale guards,
  - Emit warnings about experimental status,
  - And provide per-parameter gradient-quality metrics for small test scenes.

#### Scenario: Incompatible configurations (beyond supported modes)
- **WHEN** Maxwell/bulk or other features are configured outside the supported modes or scale guards,
- **THEN** the solver SHALL block gradient computation by default with actionable messaging,
- AND SHALL point users to manual adjoint + small-scene verification scripts as the supported workflow.

