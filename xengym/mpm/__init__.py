"""
MPM (Material Point Method) Module
Provides VHE-MLS-MPM solver with automatic differentiation support

Autodiff support:
- DifferentiableMPMSolver: Blocked due to Taichi AD limitation on atomic ops
- ManualAdjointMPMSolver: Manual adjoint implementation that bypasses Taichi AD limitation
"""
from .config import (
    MPMConfig,
    GridConfig,
    TimeConfig,
    OgdenConfig,
    MaxwellBranchConfig,
    MaterialConfig,
    SDFConfig,
    ContactConfig,
    OutputConfig
)
from .fields import MPMFields
from .mpm_solver import MPMSolver
from .autodiff_wrapper import DifferentiableMPMSolver
from .manual_adjoint_solver import ManualAdjointMPMSolver
from .manual_adjoint import (
    ManualAdjointFields,
    bspline_weight,
    bspline_weight_gradient,
    grid_ops_backward_kernel,
    p2g_backward_kernel,
    g2p_backward_kernel,
    update_F_backward_kernel,
    maxwell_backward_kernel,
    maxwell_G_gradient_kernel,
    bulk_viscosity_gradient_kernel,
    position_loss_backward_kernel,
    velocity_loss_backward_kernel,
    kinetic_energy_loss_backward_kernel,
    total_energy_loss_backward_kernel,
)
from .constitutive import (
    compute_ogden_stress_2terms,
    compute_ogden_stress_general,
    compute_maxwell_stress,
    compute_bulk_viscosity_stress,
    compute_maxwell_stress_no_update,
    compute_bulk_viscosity_stress_no_energy,
)
from .constitutive_gradients import (
    configure_gradient_mode,
    validate_gradient_mode,
    is_experimental_mode_enabled,
    get_scale_guards,
    compute_p_total_with_gradients,
    compute_p_total_for_diff,
    compute_g_F_numerical_p_total,
    compute_bulk_viscosity_gradient,
)
from .contact import (
    compute_contact_force,
    sdf_sphere,
    sdf_plane,
    sdf_box
)
from .decomp import (
    polar_decompose,
    safe_svd,
    eig_sym_3x3,
    make_spd,
    make_spd_ste,
    clamp_J
)
from .stability import (
    check_ogden_drucker_stability,
    check_timestep_constraints,
    validate_config
)
from .exceptions import (
    MPMError,
    ConfigurationError,
    MaterialError,
    StabilityError,
    AutodiffError,
    GradientError,
    ScaleGuardError,
    TargetNotSetError,
)

__all__ = [
    # Config
    'MPMConfig',
    'GridConfig',
    'TimeConfig',
    'OgdenConfig',
    'MaxwellBranchConfig',
    'MaterialConfig',
    'SDFConfig',
    'ContactConfig',
    'OutputConfig',
    # Core
    'MPMFields',
    'MPMSolver',
    'DifferentiableMPMSolver',
    # Manual Adjoint (new)
    'ManualAdjointMPMSolver',
    'ManualAdjointFields',
    'bspline_weight',
    'bspline_weight_gradient',
    'grid_ops_backward_kernel',
    'p2g_backward_kernel',
    'g2p_backward_kernel',
    'update_F_backward_kernel',
    'maxwell_backward_kernel',
    'maxwell_G_gradient_kernel',
    'bulk_viscosity_gradient_kernel',
    'position_loss_backward_kernel',
    'velocity_loss_backward_kernel',
    'kinetic_energy_loss_backward_kernel',
    'total_energy_loss_backward_kernel',
    # Constitutive
    'compute_ogden_stress_2terms',
    'compute_ogden_stress_general',
    'compute_maxwell_stress',
    'compute_bulk_viscosity_stress',
    'compute_maxwell_stress_no_update',
    'compute_bulk_viscosity_stress_no_energy',
    # Gradient Configuration
    'configure_gradient_mode',
    'validate_gradient_mode',
    'is_experimental_mode_enabled',
    'get_scale_guards',
    'compute_p_total_with_gradients',
    'compute_p_total_for_diff',
    'compute_g_F_numerical_p_total',
    'compute_bulk_viscosity_gradient',
    # Contact
    'compute_contact_force',
    'sdf_sphere',
    'sdf_plane',
    'sdf_box',
    # Decomp
    'polar_decompose',
    'safe_svd',
    'eig_sym_3x3',
    'make_spd',
    'make_spd_ste',
    'clamp_J',
    # Stability
    'check_ogden_drucker_stability',
    'check_timestep_constraints',
    'validate_config',
    # Exceptions
    'MPMError',
    'ConfigurationError',
    'MaterialError',
    'StabilityError',
    'AutodiffError',
    'GradientError',
    'ScaleGuardError',
    'TargetNotSetError',
]
