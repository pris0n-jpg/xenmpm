## MODIFIED Requirements

### Requirement: Verification Scenes Coverage
- The simulator SHALL provide runnable scenes covering:
  - Uniaxial tension: stress–strain vs Ogden theory.
  - Pure shear + objectivity: stress invariant under superposed rigid rotation.
  - Energy conservation/dissipation with projection: tracking E_kin, E_elastic, E_viscous_step/cum, ΔE_proj_step, E_proj_cum and convergence vs Δt/grid.
  - Stick–slip / incipient slip (GelSlim-style): tangential force–displacement curve and stick/slip identification.
  - Hertzian contact / elastic impact: error vs time step/grid size convergence.
- Each scene SHALL have a CLI entry and config template with small default particle counts.
- Each scene SHALL emit CSV with mandated metrics and a companion plotting script to render required curves.

#### Scenario: Uniaxial tension
- Given a uniaxial-tension scene is selected via CLI,
- When the simulation runs to completion,
- Then it writes stress–strain CSV and a plot comparing to Ogden analytical curve.

#### Scenario: Pure shear objectivity
- Given a pure shear scene with optional rigid rotation toggle,
- When rotation is applied/not applied,
- Then stress response remains objective and is shown in an overlay plot.

#### Scenario: Energy convergence with projection
- Given an energy-validation scene,
- When varying time step or grid size,
- Then CSV includes E_kin, E_elastic, E_viscous_step/cum, ΔE_proj_step, E_proj_cum and plots show ΔE_proj_step vs viscous, and convergence vs Δt/grid.

#### Scenario: Stick–slip / incipient slip
- Given a GelSlim-style scene,
- When sliding occurs,
- Then CSV records tangential displacement/force and plot shows stick→slip transition and incipient slip behavior.

#### Scenario: Hertz / impact convergence
- Given an elastic impact scene,
- When running at multiple Δt or grid resolutions,
- Then CSV and plots report error vs Δt/grid with expected convergence trend.
