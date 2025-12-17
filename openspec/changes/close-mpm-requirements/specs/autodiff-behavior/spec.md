## MODIFIED Requirements

### Requirement: Autodiff / Manual Adjoint Behavior
- The solver SHALL expose differentiated simulation for material parameters and initial states with clearly defined modes:
  - Pure Ogden mode: supported and validated (fast path).
  - Extended P_total numerical mode: optional, small-scale only, enabling Maxwell/bulk viscosity gradients with size safeguards and warnings.
  - Incompatible configurations (beyond supported modes) SHALL be blocked by default with actionable messaging.
- The solver SHALL provide gradient verification scripts comparing analytic/adjoint vs finite difference on toy scenes.

#### Scenario: Pure Ogden mode
- Given a pure Ogden configuration and autodiff enabled,
- When running gradient computation,
- Then gradients are computed without blocking and pass numerical checks on small scenes.

#### Scenario: Maxwell enabled (unsupported)
- Given Maxwell branches with autodiff requested in strict mode,
- When starting,
- Then the solver blocks with a clear incompatibility message.

#### Scenario: Experimental P_total numerical mode
- Given Maxwell/bulk viscosity enabled and experimental numerical mode on,
- When running small-scale simulation within size limits,
- Then gradients are produced via P_total numerical differentiation and reported with a warning about experimental status.

#### Scenario: Gradient verification script
- Given the provided verification script,
- When executed on a toy scene,
- Then it reports analytic vs numerical gradients and relative error for selected parameters/initial states.
