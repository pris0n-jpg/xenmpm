"""
Constitutive Model Gradients for Manual Adjoint

Implements analytical gradients of Ogden hyperelastic stress:
- ∂P/∂F: Stress w.r.t. deformation gradient (4th-order tensor)
- ∂P/∂μ_k: Stress w.r.t. shear moduli
- ∂P/∂α_k: Stress w.r.t. exponents
- ∂P/∂κ: Stress w.r.t. bulk modulus (added in refine-mpm-ad-precision)

These gradients are needed for manual adjoint backpropagation in P2G kernel.

ACCURACY TIERS (from refine-mpm-ad-precision spec):
- Tier A (constitutive/stress level): rel_error ≤ 1%, cosine_sim ≥ 0.99
- Tier B (small MPM toy scenes): rel_error ≤ 5%, cosine_sim ≥ 0.95
- Tier C (high-deformation end-to-end): rel_error ≤ 50%, cosine_sim ≥ 0.80

MODES:
1. Pure Ogden mode (default): Fast numerical differentiation for pure hyperelastic configs.
   Maxwell and bulk viscosity are blocked in strict mode.

2. Experimental P_total numerical mode: Full numerical differentiation of total stress
   P_total = P_ogden + P_maxwell + P_visc. Includes:
   - compute_p_total_for_diff: Computes full stress without updating internal variables
   - compute_g_F_numerical_p_total: Numerical g_F through full P_total
   - compute_p_total_with_gradients: Full gradient computation with P_total support

   Limited to small-scale simulations (particle/step caps) with explicit warnings.

LIMITATIONS:
1. In standard mode, g_F only differentiates Ogden elastic stress.
   Maxwell and bulk viscosity contributions are NOT included.
2. In experimental P_total mode, g_F differentiates full P_total, but g_mu/g_alpha
   still only compute Ogden parameter gradients (Maxwell/bulk viscosity parameter
   gradients are NOT implemented).
3. Jacobian clamping uses FIXED values (0.5, 2.0) matching the forward pass in
   compute_ogden_stress_general(). This cannot be configured.
4. The analytical g_F path is INCOMPLETE (missing F @ ∂S/∂F term) and should only
   be used for debugging/performance comparison, not production calibration.
"""
import taichi as ti
from .decomp import eig_sym_3x3, clamp_J
from .constitutive import (
    compute_ogden_stress_general,
    compute_maxwell_stress_no_update,
    compute_bulk_viscosity_stress_no_energy,
)
from .exceptions import ScaleGuardError, MaterialError

# Module-level configuration (set before Taichi compilation)
_USE_NUMERICAL_G_F = True  # Default: use numerical for correctness
_FINITE_DIFF_EPS = 1e-4    # Finite difference step size
# Note: Jacobian clamp values are FIXED at (0.5, 2.0) in compute_ogden_stress_general
# and cannot be configured here. They are kept as constants for documentation.
_J_CLAMP_MIN = 0.5  # Fixed, matches forward pass
_J_CLAMP_MAX = 2.0  # Fixed, matches forward pass

# Experimental P_total numerical mode
_EXPERIMENTAL_P_TOTAL_MODE = False
_P_TOTAL_MAX_PARTICLES = 5000  # Scale guard
_P_TOTAL_MAX_STEPS = 500       # Scale guard


def configure_gradient_mode(use_numerical: bool = True, eps: float = 1e-4,
                            experimental_p_total: bool = False,
                            max_particles: int = 5000, max_steps: int = 500):
    """
    Configure gradient computation mode. Call BEFORE Taichi compilation.

    Args:
        use_numerical: If True, use numerical differentiation for g_F (more accurate
                       but slower, 18 stress evaluations per particle per step).
                       If False, use analytical approximation (fast but INCOMPLETE -
                       missing F @ ∂S/∂F term, only for debugging).
        eps: Finite difference step size for numerical mode.
        experimental_p_total: Enable experimental P_total numerical gradient mode.
                              When True, Maxwell/bulk viscosity gradients are computed
                              via full numerical differentiation. Subject to scale limits.
        max_particles: Maximum particles for experimental mode (default: 5000)
        max_steps: Maximum steps for experimental mode (default: 500)

    Note:
        Jacobian clamping uses FIXED values (0.5, 2.0) matching the forward pass.
        This cannot be configured to avoid forward/backward inconsistency.

    Warning:
        When use_numerical=False, the gradient is INCOMPLETE and should NOT be used
        for production calibration. A warning will be issued at import time.

        When experimental_p_total=True, P_total numerical gradients are available
        but computationally expensive. Scale guards will block large simulations.
    """
    import warnings
    global _USE_NUMERICAL_G_F, _FINITE_DIFF_EPS
    global _EXPERIMENTAL_P_TOTAL_MODE, _P_TOTAL_MAX_PARTICLES, _P_TOTAL_MAX_STEPS

    _USE_NUMERICAL_G_F = use_numerical
    _FINITE_DIFF_EPS = eps
    _EXPERIMENTAL_P_TOTAL_MODE = experimental_p_total
    _P_TOTAL_MAX_PARTICLES = max_particles
    _P_TOTAL_MAX_STEPS = max_steps

    if not use_numerical:
        warnings.warn(
            "Analytical g_F mode enabled. This is INCOMPLETE (missing F @ ∂S/∂F term) "
            "and should only be used for debugging/performance comparison. "
            "For production calibration, use numerical mode (use_numerical=True).",
            UserWarning
        )

    if experimental_p_total:
        warnings.warn(
            f"EXPERIMENTAL: P_total numerical gradient mode enabled. "
            f"Maxwell/bulk viscosity gradients will be computed via full numerical differentiation. "
            f"Scale guards: max_particles={max_particles}, max_steps={max_steps}. "
            f"This mode is computationally expensive and NOT validated for large-scale simulations.",
            UserWarning
        )


def validate_gradient_mode(config, strict: bool = True, n_particles: int = None,
                           n_steps: int = None) -> bool:
    """
    Validate gradient configuration against MPM config.

    Args:
        config: MPMConfig instance
        strict: If True (default), raise ValueError when Maxwell/bulk viscosity enabled
                and experimental mode is off.
                If False, only issue warning and return False.
        n_particles: Number of particles (for scale guard check in experimental mode)
        n_steps: Number of steps (for scale guard check in experimental mode)

    Returns:
        True if configuration is valid for gradient computation.

    Raises:
        ValueError: If strict=True and configuration is incompatible.
        RuntimeWarning: If strict=False and configuration has issues.
    """
    import warnings

    has_maxwell = len(config.material.maxwell_branches) > 0
    has_bulk_visc = config.material.enable_bulk_viscosity
    needs_p_total = has_maxwell or has_bulk_visc

    if needs_p_total:
        if _EXPERIMENTAL_P_TOTAL_MODE:
            # Check scale guards
            if n_particles is not None and n_particles > _P_TOTAL_MAX_PARTICLES:
                msg = (
                    f"EXPERIMENTAL P_total mode: particle count ({n_particles}) exceeds "
                    f"scale guard ({_P_TOTAL_MAX_PARTICLES}). Reduce particles or increase "
                    f"max_particles in configure_gradient_mode()."
                )
                if strict:
                    raise ScaleGuardError(msg)
                else:
                    warnings.warn(msg, RuntimeWarning)
                    return False

            if n_steps is not None and n_steps > _P_TOTAL_MAX_STEPS:
                msg = (
                    f"EXPERIMENTAL P_total mode: step count ({n_steps}) exceeds "
                    f"scale guard ({_P_TOTAL_MAX_STEPS}). Reduce steps or increase "
                    f"max_steps in configure_gradient_mode()."
                )
                if strict:
                    raise ScaleGuardError(msg)
                else:
                    warnings.warn(msg, RuntimeWarning)
                    return False

            # Experimental mode is enabled and within limits
            warnings.warn(
                f"Using EXPERIMENTAL P_total numerical mode for Maxwell/bulk viscosity gradients. "
                f"Maxwell branches={len(config.material.maxwell_branches)}, bulk_viscosity={has_bulk_visc}. "
                f"Results may be slow and should be verified numerically.",
                RuntimeWarning
            )
            return True
        else:
            # Standard mode: block Maxwell/bulk viscosity
            msg = (
                f"Manual adjoint gradient computation is INCOMPATIBLE with current config. "
                f"g_F only differentiates Ogden stress, but P_total includes: "
                f"Maxwell branches={len(config.material.maxwell_branches)}, "
                f"bulk_viscosity={has_bulk_visc}. "
                f"Gradients will be INCORRECT. "
                f"Options: (1) Disable Maxwell branches and bulk viscosity, or "
                f"(2) Enable experimental_p_total mode in configure_gradient_mode() "
                f"for small-scale simulations."
            )
            if strict:
                raise MaterialError(msg)
            else:
                warnings.warn(msg, RuntimeWarning)
                return False

    return True


def is_experimental_mode_enabled() -> bool:
    """Check if experimental P_total mode is enabled."""
    return _EXPERIMENTAL_P_TOTAL_MODE


def get_scale_guards() -> tuple:
    """Get current scale guard limits (max_particles, max_steps)."""
    return (_P_TOTAL_MAX_PARTICLES, _P_TOTAL_MAX_STEPS)


@ti.func
def compute_ogden_stress_for_diff(
    F: ti.template(),
    mu: ti.template(),
    alpha: ti.template(),
    n_terms: ti.i32,
    kappa: ti.f32
) -> ti.template():
    """
    Compute Ogden stress for finite difference. Reuses compute_ogden_stress_general.
    Returns only P (stress), discarding energy.
    """
    P, _ = compute_ogden_stress_general(F, mu, alpha, n_terms, kappa)
    return P


@ti.func
def compute_g_F_numerical(
    F: ti.template(),
    mu: ti.template(),
    alpha: ti.template(),
    n_terms: ti.i32,
    kappa: ti.f32,
    g_P: ti.template()
) -> ti.template():
    """
    Compute g_F using numerical differentiation (finite differences).
    g_F[m,n] = sum_{i,j} g_P[i,j] * dP[i,j]/dF[m,n]

    This correctly captures the full dP/dF including the F @ dS/dF term,
    which is missing from the analytical approximation.

    Note: Only differentiates Ogden elastic stress. Maxwell and bulk viscosity
    contributions are NOT included - use validate_gradient_mode() to check.

    Performance: 18 stress evaluations per particle per step (9 components × 2 for central diff).
    """
    # Use module-level eps (configured via configure_gradient_mode)
    eps = ti.static(_FINITE_DIFF_EPS)
    g_F = ti.Matrix.zero(ti.f32, 3, 3)

    # For each F component, compute numerical derivative
    for m in ti.static(range(3)):
        for n in ti.static(range(3)):
            # F + eps * e_m e_n^T
            F_plus = F
            F_plus[m, n] += eps

            # F - eps * e_m e_n^T
            F_minus = F
            F_minus[m, n] -= eps

            # Compute P at perturbed F values (reuse general stress function)
            P_plus = compute_ogden_stress_for_diff(F_plus, mu, alpha, n_terms, kappa)
            P_minus = compute_ogden_stress_for_diff(F_minus, mu, alpha, n_terms, kappa)

            # dP/dF[m,n] via central difference
            dP_dF_mn = (P_plus - P_minus) / (2.0 * eps)

            # Contract with g_P
            inner = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    inner += g_P[i, j] * dP_dF_mn[i, j]

            g_F[m, n] = inner

    return g_F


@ti.func
def compute_g_F_analytical(
    F: ti.template(),
    S: ti.template(),
    g_P: ti.template()
) -> ti.template():
    """
    Compute g_F using analytical approximation (fast but incomplete).

    Only computes g_F = g_P @ S^T, which is the ∂(F@S)/∂F part.
    Missing the F @ ∂S/∂F term (4th-order elasticity tensor contribution).

    Use this for performance when accuracy is less critical.
    """
    return g_P @ S.transpose()


@ti.func
def compute_ogden_stress_with_gradients(
    F: ti.template(),
    mu: ti.template(),
    alpha: ti.template(),
    n_terms: ti.i32,
    kappa: ti.f32,
    g_P: ti.template()  # Gradient w.r.t. P (3x3)
) -> ti.template():
    """
    Compute Ogden stress gradients and return them

    Given g_P (gradient w.r.t. stress P), compute:
    - g_F = ∂L/∂F = <g_P, ∂P/∂F>
    - g_mu[k] = ∂L/∂μ_k = <g_P, ∂P/∂μ_k>
    - g_alpha[k] = ∂L/∂α_k = <g_P, ∂P/∂α_k>
    - g_kappa = ∂L/∂κ = <g_P, ∂P/∂κ>

    This implements the chain rule for backpropagation through Ogden stress.

    Configuration (via configure_gradient_mode() - call BEFORE ti.init()):
    - _USE_NUMERICAL_G_F: If True, use numerical differentiation for g_F (accurate
      but 18x slower). If False, use analytical approximation (INCOMPLETE - missing
      F @ ∂S/∂F term, only for debugging).

    Note:
        Jacobian clamping uses FIXED values (0.5, 2.0) matching the forward pass.
        This ensures gradient consistency with forward stress computation.

    Args:
        F: 3x3 deformation gradient
        mu: Taichi field of shear moduli
        alpha: Taichi field of exponents
        n_terms: Number of Ogden terms
        kappa: Bulk modulus
        g_P: 3x3 gradient w.r.t. stress P (input)

    Returns:
        Tuple of (g_F, g_mu, g_alpha, g_kappa) where:
        - g_F: 3x3 matrix gradient w.r.t. F
        - g_mu: Vector[4] gradient w.r.t. mu
        - g_alpha: Vector[4] gradient w.r.t. alpha
        - g_kappa: Scalar gradient w.r.t. kappa (bulk modulus)
    """
    # Use module-level clamp configuration (compile-time constant)
    j_clamp_min = ti.static(_J_CLAMP_MIN)
    j_clamp_max = ti.static(_J_CLAMP_MAX)

    # Clamp Jacobian
    F_clamped = clamp_J(F, j_clamp_min, j_clamp_max)
    J = F_clamped.determinant()

    # Right Cauchy-Green tensor
    C = F_clamped.transpose() @ F_clamped
    C = 0.5 * (C + C.transpose())  # Explicit symmetrization

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eig_sym_3x3(C)

    # Principal stretches
    lambda_vec = ti.Vector([ti.sqrt(ti.max(eigenvalues[0], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[1], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[2], 1e-8))])

    # Deviatoric part
    J_pow = ti.pow(J, -1.0/3.0)
    lambda_bar = lambda_vec * J_pow

    # ============================================
    # Forward pass: Compute stress (needed for analytical g_F)
    # ============================================
    # Compute deviatoric stress
    S_dev_principal = ti.Vector([0.0, 0.0, 0.0])

    for k in ti.static(range(4)):  # Max 4 terms
        if k < n_terms:
            mu_k = mu[k]
            alpha_k = alpha[k]

            for i in ti.static(range(3)):
                S_dev_principal[i] += mu_k * ti.pow(lambda_bar[i], alpha_k - 1.0)

    # Apply deviatoric projection
    trace_S = S_dev_principal[0] + S_dev_principal[1] + S_dev_principal[2]
    for i in ti.static(range(3)):
        S_dev_principal[i] = J_pow * (S_dev_principal[i] - trace_S / 3.0)

    # Reconstruct deviatoric 2nd PK stress
    S_dev = ti.Matrix.zero(ti.f32, 3, 3)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            for k in ti.static(range(3)):
                S_dev[i, j] += S_dev_principal[k] * eigenvectors[i, k] * eigenvectors[j, k]

    # Volumetric part
    F_inv = F_clamped.inverse()
    S_vol = kappa * (J - 1.0) * J * F_inv.transpose() @ F_inv

    # Total stress
    S = S_dev + S_vol

    # ============================================
    # Backward pass: Compute gradients
    # ============================================

    # Initialize output gradients
    g_F_out = ti.Matrix.zero(ti.f32, 3, 3)
    g_mu_out = ti.Vector([0.0, 0.0, 0.0, 0.0])
    g_alpha_out = ti.Vector([0.0, 0.0, 0.0, 0.0])

    # 1. Gradient w.r.t. F - mode selected at compile time
    # P = F @ S, so ∂P/∂F involves both ∂(F@S)/∂F and F @ ∂S/∂F
    if ti.static(_USE_NUMERICAL_G_F):
        # Numerical differentiation: accurate but slower (18 stress evals)
        g_F_out = compute_g_F_numerical(F, mu, alpha, n_terms, kappa, g_P)
    else:
        # Analytical approximation: fast but missing F @ ∂S/∂F term
        g_F_out = compute_g_F_analytical(F, S, g_P)

    # 2. Gradient w.r.t. μ_k
    # For each Ogden term k: ∂P/∂μ_k = F @ ∂S_dev/∂μ_k
    for k in ti.static(range(4)):
        if k < n_terms:
            alpha_k = alpha[k]

            # Compute ∂S_dev_principal/∂μ_k
            dS_principal_dmu = ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(3)):
                dS_principal_dmu[i] = ti.pow(lambda_bar[i], alpha_k - 1.0)

            # Apply deviatoric projection
            trace_dS = dS_principal_dmu[0] + dS_principal_dmu[1] + dS_principal_dmu[2]
            for i in ti.static(range(3)):
                dS_principal_dmu[i] = J_pow * (dS_principal_dmu[i] - trace_dS / 3.0)

            # Reconstruct ∂S_dev/∂μ_k in full 3x3
            dS_dev_dmu = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for m in ti.static(range(3)):
                        dS_dev_dmu[i, j] += dS_principal_dmu[m] * eigenvectors[i, m] * eigenvectors[j, m]

            # ∂P/∂μ_k = F @ ∂S_dev/∂μ_k
            dP_dmu = F_clamped @ dS_dev_dmu

            # g_mu[k] = <g_P, ∂P/∂μ_k>
            inner_prod = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    inner_prod += g_P[i, j] * dP_dmu[i, j]

            g_mu_out[k] = inner_prod

    # 3. Gradient w.r.t. α_k
    # ∂P/∂α_k = F @ ∂S_dev/∂α_k
    for k in ti.static(range(4)):
        if k < n_terms:
            mu_k = mu[k]
            alpha_k = alpha[k]

            # Compute ∂S_dev_principal/∂α_k
            dS_principal_dalpha = ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(3)):
                # Avoid log(0) by clamping
                lambda_bar_i = ti.max(lambda_bar[i], 1e-8)
                dS_principal_dalpha[i] = mu_k * ti.pow(lambda_bar_i, alpha_k - 1.0) * ti.log(lambda_bar_i)

            # Apply deviatoric projection
            trace_dS = dS_principal_dalpha[0] + dS_principal_dalpha[1] + dS_principal_dalpha[2]
            for i in ti.static(range(3)):
                dS_principal_dalpha[i] = J_pow * (dS_principal_dalpha[i] - trace_dS / 3.0)

            # Reconstruct ∂S_dev/∂α_k in full 3x3
            dS_dev_dalpha = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for m in ti.static(range(3)):
                        dS_dev_dalpha[i, j] += dS_principal_dalpha[m] * eigenvectors[i, m] * eigenvectors[j, m]

            # ∂P/∂α_k = F @ ∂S_dev/∂α_k
            dP_dalpha = F_clamped @ dS_dev_dalpha

            # g_alpha[k] = <g_P, ∂P/∂α_k>
            inner_prod = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    inner_prod += g_P[i, j] * dP_dalpha[i, j]

            g_alpha_out[k] = inner_prod

    # 4. Gradient w.r.t. κ (bulk modulus)
    # S_vol = κ * (J - 1) * J * F^{-T} @ F^{-1}
    # ∂S_vol/∂κ = (J - 1) * J * F^{-T} @ F^{-1}
    # ∂P_vol/∂κ = F @ ∂S_vol/∂κ
    dS_vol_dkappa = (J - 1.0) * J * F_inv.transpose() @ F_inv
    dP_dkappa = F_clamped @ dS_vol_dkappa

    # g_kappa = <g_P, ∂P/∂κ>
    g_kappa_out = 0.0
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            g_kappa_out += g_P[i, j] * dP_dkappa[i, j]

    return g_F_out, g_mu_out, g_alpha_out, g_kappa_out


# ============================================
# P_total Numerical Gradient (Experimental Mode)
# ============================================
# These functions compute gradients through the FULL stress tensor including
# Maxwell viscoelasticity and bulk viscosity contributions.

@ti.func
def compute_p_total_for_diff(
    F: ti.template(),
    mu: ti.template(),
    alpha: ti.template(),
    n_terms: ti.i32,
    kappa: ti.f32,
    # Maxwell parameters (up to 4 branches)
    maxwell_G: ti.template(),
    maxwell_tau: ti.template(),
    maxwell_b_bar_e: ti.template(),  # (4, 3, 3) internal variables
    n_maxwell: ti.i32,
    # Bulk viscosity parameters
    F_old: ti.template(),
    dt: ti.f32,
    eta_bulk: ti.f32,
    enable_bulk_visc: ti.i32
) -> ti.template():
    """
    Compute total stress P_total = P_ogden + P_maxwell + P_visc for numerical differentiation.
    Does NOT update internal variables.

    Args:
        F: 3x3 deformation gradient
        mu, alpha, n_terms, kappa: Ogden parameters
        maxwell_G, maxwell_tau: Maxwell branch parameters (fields)
        maxwell_b_bar_e: (4, 3, 3) internal variables for Maxwell branches
        n_maxwell: Number of Maxwell branches
        F_old: Previous deformation gradient (for bulk viscosity)
        dt: Time step
        eta_bulk: Bulk viscosity coefficient
        enable_bulk_visc: Whether bulk viscosity is enabled (1 or 0)

    Returns:
        P_total: 3x3 total first Piola-Kirchhoff stress
    """
    # 1. Ogden elastic stress
    P_ogden, _ = compute_ogden_stress_general(F, mu, alpha, n_terms, kappa)

    P_total = P_ogden

    # 2. Maxwell viscoelastic stress (without updating internal variables)
    for k in ti.static(range(4)):  # Max 4 Maxwell branches
        if k < n_maxwell:
            P_maxwell_k = compute_maxwell_stress_no_update(
                F, maxwell_b_bar_e[k], maxwell_G[k], maxwell_tau[k]
            )
            P_total += P_maxwell_k

    # 3. Bulk viscosity stress
    if enable_bulk_visc == 1:
        P_visc = compute_bulk_viscosity_stress_no_energy(F, F_old, dt, eta_bulk)
        P_total += P_visc

    return P_total


@ti.func
def compute_g_F_numerical_p_total(
    F: ti.template(),
    mu: ti.template(),
    alpha: ti.template(),
    n_terms: ti.i32,
    kappa: ti.f32,
    maxwell_G: ti.template(),
    maxwell_tau: ti.template(),
    maxwell_b_bar_e: ti.template(),
    n_maxwell: ti.i32,
    F_old: ti.template(),
    dt: ti.f32,
    eta_bulk: ti.f32,
    enable_bulk_visc: ti.i32,
    g_P: ti.template()
) -> ti.template():
    """
    Compute g_F using numerical differentiation of FULL P_total.
    g_F[m,n] = sum_{i,j} g_P[i,j] * dP_total[i,j]/dF[m,n]

    This correctly captures gradients through:
    - Ogden elastic stress
    - Maxwell viscoelastic stress (all branches)
    - Bulk viscosity stress

    Performance: 18 stress evaluations per particle per step (9 components × 2 for central diff).
    Each evaluation computes full P_total including Maxwell branches.

    Note: This is computationally expensive. Scale guards should be enforced.
    """
    eps = ti.static(_FINITE_DIFF_EPS)
    g_F = ti.Matrix.zero(ti.f32, 3, 3)

    # For each F component, compute numerical derivative
    for m in ti.static(range(3)):
        for n in ti.static(range(3)):
            # F + eps * e_m e_n^T
            F_plus = F
            F_plus[m, n] += eps

            # F - eps * e_m e_n^T
            F_minus = F
            F_minus[m, n] -= eps

            # Compute P_total at perturbed F values
            P_plus = compute_p_total_for_diff(
                F_plus, mu, alpha, n_terms, kappa,
                maxwell_G, maxwell_tau, maxwell_b_bar_e, n_maxwell,
                F_old, dt, eta_bulk, enable_bulk_visc
            )
            P_minus = compute_p_total_for_diff(
                F_minus, mu, alpha, n_terms, kappa,
                maxwell_G, maxwell_tau, maxwell_b_bar_e, n_maxwell,
                F_old, dt, eta_bulk, enable_bulk_visc
            )

            # dP_total/dF[m,n] via central difference
            dP_dF_mn = (P_plus - P_minus) / (2.0 * eps)

            # Contract with g_P
            inner = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    inner += g_P[i, j] * dP_dF_mn[i, j]

            g_F[m, n] = inner

    return g_F


@ti.func
def compute_p_total_with_gradients(
    F: ti.template(),
    mu: ti.template(),
    alpha: ti.template(),
    n_terms: ti.i32,
    kappa: ti.f32,
    maxwell_G: ti.template(),
    maxwell_tau: ti.template(),
    maxwell_b_bar_e: ti.template(),
    n_maxwell: ti.i32,
    F_old: ti.template(),
    dt: ti.f32,
    eta_bulk: ti.f32,
    enable_bulk_visc: ti.i32,
    g_P: ti.template()
) -> ti.template():
    """
    Compute P_total gradients with full Maxwell/bulk viscosity support.
    EXPERIMENTAL: Used when experimental_p_total mode is enabled.

    Given g_P (gradient w.r.t. stress P), compute:
    - g_F = ∂L/∂F = <g_P, ∂P_total/∂F>
    - g_mu[k] = ∂L/∂μ_k (Ogden only, Maxwell μ not differentiated)
    - g_alpha[k] = ∂L/∂α_k (Ogden only)
    - g_kappa = ∂L/∂κ (bulk modulus)

    Note: This only computes g_F through full P_total. Ogden parameter gradients
    (g_mu, g_alpha, g_kappa) are still computed using the Ogden-only path for simplicity.
    Maxwell/bulk viscosity parameter gradients are NOT computed.

    Args:
        F: 3x3 deformation gradient
        mu, alpha, n_terms, kappa: Ogden parameters
        maxwell_G, maxwell_tau, maxwell_b_bar_e, n_maxwell: Maxwell parameters
        F_old, dt, eta_bulk, enable_bulk_visc: Bulk viscosity parameters
        g_P: 3x3 gradient w.r.t. stress P (input)

    Returns:
        Tuple of (g_F, g_mu, g_alpha, g_kappa) where:
        - g_F: 3x3 matrix gradient w.r.t. F (through full P_total)
        - g_mu: Vector[4] gradient w.r.t. mu (Ogden only)
        - g_alpha: Vector[4] gradient w.r.t. alpha (Ogden only)
        - g_kappa: Scalar gradient w.r.t. kappa (Ogden only)
    """
    # 1. g_F through full P_total (numerical differentiation)
    g_F_out = compute_g_F_numerical_p_total(
        F, mu, alpha, n_terms, kappa,
        maxwell_G, maxwell_tau, maxwell_b_bar_e, n_maxwell,
        F_old, dt, eta_bulk, enable_bulk_visc,
        g_P
    )

    # 2. g_mu and g_alpha still use Ogden-only path
    # (Maxwell/bulk viscosity parameter gradients not implemented)
    j_clamp_min = ti.static(_J_CLAMP_MIN)
    j_clamp_max = ti.static(_J_CLAMP_MAX)

    F_clamped = clamp_J(F, j_clamp_min, j_clamp_max)
    J = F_clamped.determinant()

    C = F_clamped.transpose() @ F_clamped
    C = 0.5 * (C + C.transpose())

    eigenvalues, eigenvectors = eig_sym_3x3(C)

    lambda_vec = ti.Vector([ti.sqrt(ti.max(eigenvalues[0], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[1], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[2], 1e-8))])

    J_pow = ti.pow(J, -1.0/3.0)
    lambda_bar = lambda_vec * J_pow

    g_mu_out = ti.Vector([0.0, 0.0, 0.0, 0.0])
    g_alpha_out = ti.Vector([0.0, 0.0, 0.0, 0.0])

    # Gradient w.r.t. μ_k (Ogden only)
    for k in ti.static(range(4)):
        if k < n_terms:
            alpha_k = alpha[k]

            dS_principal_dmu = ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(3)):
                dS_principal_dmu[i] = ti.pow(lambda_bar[i], alpha_k - 1.0)

            trace_dS = dS_principal_dmu[0] + dS_principal_dmu[1] + dS_principal_dmu[2]
            for i in ti.static(range(3)):
                dS_principal_dmu[i] = J_pow * (dS_principal_dmu[i] - trace_dS / 3.0)

            dS_dev_dmu = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for m in ti.static(range(3)):
                        dS_dev_dmu[i, j] += dS_principal_dmu[m] * eigenvectors[i, m] * eigenvectors[j, m]

            dP_dmu = F_clamped @ dS_dev_dmu

            inner_prod = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    inner_prod += g_P[i, j] * dP_dmu[i, j]

            g_mu_out[k] = inner_prod

    # Gradient w.r.t. α_k (Ogden only)
    for k in ti.static(range(4)):
        if k < n_terms:
            mu_k = mu[k]
            alpha_k = alpha[k]

            dS_principal_dalpha = ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(3)):
                lambda_bar_i = ti.max(lambda_bar[i], 1e-8)
                dS_principal_dalpha[i] = mu_k * ti.pow(lambda_bar_i, alpha_k - 1.0) * ti.log(lambda_bar_i)

            trace_dS = dS_principal_dalpha[0] + dS_principal_dalpha[1] + dS_principal_dalpha[2]
            for i in ti.static(range(3)):
                dS_principal_dalpha[i] = J_pow * (dS_principal_dalpha[i] - trace_dS / 3.0)

            dS_dev_dalpha = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for m in ti.static(range(3)):
                        dS_dev_dalpha[i, j] += dS_principal_dalpha[m] * eigenvectors[i, m] * eigenvectors[j, m]

            dP_dalpha = F_clamped @ dS_dev_dalpha

            inner_prod = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    inner_prod += g_P[i, j] * dP_dalpha[i, j]

            g_alpha_out[k] = inner_prod

    # 4. Gradient w.r.t. κ (bulk modulus) - Ogden only
    F_inv = F_clamped.inverse()
    dS_vol_dkappa = (J - 1.0) * J * F_inv.transpose() @ F_inv
    dP_dkappa = F_clamped @ dS_vol_dkappa

    g_kappa_out = 0.0
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            g_kappa_out += g_P[i, j] * dP_dkappa[i, j]

    return g_F_out, g_mu_out, g_alpha_out, g_kappa_out


# ============================================
# Maxwell Parameter Gradients (G, τ)
# ============================================
# These functions compute gradients w.r.t. Maxwell branch parameters.
# g_tau is computed in maxwell_backward_kernel (from internal variable update).
# g_G is computed here from the stress backward pass.

@ti.func
def compute_maxwell_G_gradient(
    F: ti.template(),
    b_bar_e: ti.template(),  # Internal variable for this branch
    g_P: ti.template(),      # Gradient w.r.t. total stress P
) -> ti.f32:
    """
    Compute gradient of loss w.r.t. Maxwell shear modulus G for a single branch.

    Forward:
        τ_k = G_k * dev(b_bar_e_k) where dev(B) = B - tr(B)/3 * I
        P_maxwell_k = J * τ_k @ F^(-T)

    Backward:
        ∂P_maxwell/∂G_k = J * dev(b_bar_e_k) @ F^(-T)
        g_G_k = <g_P, ∂P_maxwell/∂G_k>

    Args:
        F: 3x3 deformation gradient
        b_bar_e: 3x3 internal variable (isochoric left Cauchy-Green)
        g_P: 3x3 gradient w.r.t. stress P

    Returns:
        g_G: Scalar gradient w.r.t. G for this branch
    """
    J = F.determinant()

    # Compute deviatoric part: dev(b_bar_e) = b_bar_e - tr(b_bar_e)/3 * I
    trace_b = b_bar_e[0, 0] + b_bar_e[1, 1] + b_bar_e[2, 2]
    dev_b = b_bar_e - (trace_b / 3.0) * ti.Matrix.identity(ti.f32, 3)

    # F^(-T)
    F_inv_T = F.inverse().transpose()

    # ∂P_maxwell/∂G = J * dev(b_bar_e) @ F^(-T)
    dP_dG = J * (dev_b @ F_inv_T)

    # g_G = <g_P, dP/dG> = Frobenius inner product
    g_G = 0.0
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            g_G += g_P[i, j] * dP_dG[i, j]

    return g_G


# ============================================
# Bulk Viscosity Parameter Gradient (η_bulk)
# ============================================

@ti.func
def compute_bulk_viscosity_gradient(
    F: ti.template(),
    C: ti.template(),  # APIC velocity gradient matrix (approximates L)
    g_P: ti.template(),
) -> ti.f32:
    """
    Compute gradient of loss w.r.t. bulk viscosity coefficient eta_bulk.

    Forward:
        L = C  (APIC matrix approximates velocity gradient)
        tr(D) ≈ tr(L) = L[0,0] + L[1,1] + L[2,2]
        sigma_visc = eta_bulk * tr(D) * I
        P_visc = J * sigma_visc @ F^{-T}

    Backward:
        ∂P_visc/∂eta_bulk = J * tr(L) * I @ F^{-T}
        g_eta_bulk = <g_P, ∂P_visc/∂eta_bulk>

    Args:
        F: 3x3 deformation gradient
        C: 3x3 APIC velocity gradient matrix
        g_P: 3x3 gradient w.r.t. stress P

    Returns:
        g_eta_bulk: Scalar gradient w.r.t. bulk viscosity coefficient
    """
    J = F.determinant()

    # Trace of velocity gradient (rate of volumetric deformation)
    trace_L = C[0, 0] + C[1, 1] + C[2, 2]

    # F^(-T)
    F_inv_T = F.inverse().transpose()

    # ∂P_visc/∂eta_bulk = J * tr(L) * I @ F^(-T)
    dP_deta = J * trace_L * F_inv_T

    # g_eta_bulk = <g_P, dP/deta> = Frobenius inner product
    g_eta_bulk = 0.0
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            g_eta_bulk += g_P[i, j] * dP_deta[i, j]

    return g_eta_bulk
