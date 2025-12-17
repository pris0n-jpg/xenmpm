## MODIFIED Requirements

### Requirement: Stability Guardrails Default-On
- The solver SHALL perform stability checks at startup by default:
  - Drucker-type Ogden stability (algebraic + optional path scan).
  - Time-step/CFL, viscous time-scale, and contact stiffness guidance.
- The checks SHALL support strict (block) and warn modes, configurable via CLI/config.
- If a configuration is unstable in strict mode, the solver SHALL refuse to run and report the violating condition.

#### Scenario: Drucker failure (strict)
- Given Ogden parameters that fail Drucker constraints,
- When running with strict mode,
- Then the solver aborts before simulation and reports the violation.

#### Scenario: Time-step/contact warning
- Given a configuration with aggressive Δt or contact stiffness,
- When running in warn mode,
- Then the solver emits warnings describing the risk and suggested bounds but continues.

#### Scenario: Path scan (debug)
- Given debug/path-scan enabled,
- When initializing,
- Then the solver samples predefined deformation paths and reports any negative tangent modulus, blocking in strict mode.
