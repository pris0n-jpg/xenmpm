"""
Constitutive Models for VHE-MPM
Implements Ogden hyperelasticity + generalized Maxwell viscoelasticity + optional Kelvin-Voigt bulk viscosity
"""
import taichi as ti
from .decomp import eig_sym_3x3, make_spd, clamp_J


@ti.func
def compute_ogden_stress_general(F: ti.template(), mu: ti.template(), alpha: ti.template(), n_terms: ti.i32, kappa: ti.f32) -> ti.template():
    """
    Compute Ogden hyperelastic stress with arbitrary number of terms

    Args:
        F: 3x3 deformation gradient
        mu: Taichi field of shear moduli
        alpha: Taichi field of exponents
        n_terms: Number of Ogden terms
        kappa: Bulk modulus

    Returns:
        P: 3x3 first Piola-Kirchhoff stress
        psi: Elastic energy density
    """
    # Clamp Jacobian
    F_clamped = clamp_J(F, 0.5, 2.0)
    J = F_clamped.determinant()

    # Right Cauchy-Green tensor (ensure symmetric due to numerical precision)
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

    # Compute deviatoric stress and energy
    psi_dev = 0.0
    S_dev_principal = ti.Vector([0.0, 0.0, 0.0])

    for k in ti.static(range(4)):  # Max 4 terms
        if k < n_terms:
            mu_k = mu[k]
            alpha_k = alpha[k]

            for i in ti.static(range(3)):
                psi_dev += mu_k / alpha_k * (ti.pow(lambda_bar[i], alpha_k) - 1.0)
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
    psi_vol = 0.5 * kappa * (J - 1.0) ** 2
    S_vol = kappa * (J - 1.0) * J * F_clamped.inverse().transpose() @ F_clamped.inverse()

    # Total stress
    S = S_dev + S_vol
    P = F_clamped @ S
    psi = psi_dev + psi_vol

    return P, psi


@ti.func
def compute_ogden_stress_2terms(F: ti.template(), mu0: ti.f32, alpha0: ti.f32, mu1: ti.f32, alpha1: ti.f32, kappa: ti.f32) -> ti.template():
    """
    Compute Ogden hyperelastic stress with 2 terms (deviatoric + volumetric)

    Ogden model: W = sum_k mu_k/alpha_k * (lambda_1^alpha_k + lambda_2^alpha_k + lambda_3^alpha_k - 3) + kappa/2 * (J-1)^2

    Args:
        F: 3x3 deformation gradient
        mu0, alpha0: First Ogden term
        mu1, alpha1: Second Ogden term
        kappa: Bulk modulus

    Returns:
        P: 3x3 first Piola-Kirchhoff stress
        psi: Elastic energy density
    """
    # Clamp Jacobian to avoid extreme deformation
    F_clamped = clamp_J(F, 0.5, 2.0)
    J = F_clamped.determinant()

    # Right Cauchy-Green tensor (ensure symmetric due to numerical precision)
    C = F_clamped.transpose() @ F_clamped
    C = 0.5 * (C + C.transpose())  # Explicit symmetrization

    # Eigenvalue decomposition of C
    eigenvalues, eigenvectors = eig_sym_3x3(C)

    # Principal stretches (lambda_i = sqrt(eigenvalue_i))
    lambda_vec = ti.Vector([ti.sqrt(ti.max(eigenvalues[0], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[1], 1e-8)),
                             ti.sqrt(ti.max(eigenvalues[2], 1e-8))])

    # Deviatoric part: J^(-1/3) * lambda_i
    J_pow = ti.pow(J, -1.0/3.0)
    lambda_bar = lambda_vec * J_pow

    # Compute deviatoric stress and energy
    psi_dev = 0.0
    S_dev_principal = ti.Vector([0.0, 0.0, 0.0])

    # Term 0
    for i in ti.static(range(3)):
        psi_dev += mu0 / alpha0 * (ti.pow(lambda_bar[i], alpha0) - 1.0)
        S_dev_principal[i] += mu0 * ti.pow(lambda_bar[i], alpha0 - 1.0)

    # Term 1
    for i in ti.static(range(3)):
        psi_dev += mu1 / alpha1 * (ti.pow(lambda_bar[i], alpha1) - 1.0)
        S_dev_principal[i] += mu1 * ti.pow(lambda_bar[i], alpha1 - 1.0)

    # Apply deviatoric projection: S_dev = J^(-1/3) * (S_principal - 1/3 * tr(S_principal) * I)
    trace_S = S_dev_principal[0] + S_dev_principal[1] + S_dev_principal[2]
    for i in ti.static(range(3)):
        S_dev_principal[i] = J_pow * (S_dev_principal[i] - trace_S / 3.0)

    # Reconstruct deviatoric 2nd PK stress in original basis
    S_dev = ti.Matrix.zero(ti.f32, 3, 3)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            for k in ti.static(range(3)):
                S_dev[i, j] += S_dev_principal[k] * eigenvectors[i, k] * eigenvectors[j, k]

    # Volumetric part: kappa * (J - 1) * J
    psi_vol = 0.5 * kappa * (J - 1.0) ** 2
    S_vol = kappa * (J - 1.0) * J * F_clamped.inverse().transpose() @ F_clamped.inverse()

    # Total 2nd PK stress
    S = S_dev + S_vol

    # Convert to 1st PK stress: P = F @ S
    P = F_clamped @ S

    # Total energy density
    psi = psi_dev + psi_vol

    return P, psi


@ti.func
def update_maxwell_branch(
    F: ti.template(),
    b_bar_e_old: ti.template(),
    dt: ti.f32,
    G: ti.f32,
    tau: ti.f32
) -> ti.template():
    """
    Update single Maxwell branch with upper-convected derivative + relaxation + SPD projection

    Args:
        F: 3x3 deformation gradient
        b_bar_e_old: 3x3 elastic left Cauchy-Green tensor (isochoric) from previous step
        dt: Time step
        G: Shear modulus of this branch
        tau: Relaxation time

    Returns:
        b_bar_e_new: Updated elastic left Cauchy-Green tensor
        tau_maxwell: Cauchy stress contribution from this branch
        delta_E_proj: Energy correction due to SPD projection
    """
    J = F.determinant()
    F_bar = ti.pow(J, -1.0/3.0) * F  # Isochoric deformation gradient

    # Velocity gradient (approximated from F)
    # For explicit MPM, we use: L ≈ (F - F_old) / dt / F_old ≈ (I - F_old^-1 @ F) / dt
    # Simplified: use F_bar directly for upper-convected update
    F_bar_inv = F_bar.inverse()

    # Upper-convected derivative: b_bar_e_dot = L @ b_bar_e + b_bar_e @ L^T
    # Approximation: b_bar_e_trial = F_bar @ b_bar_e_old @ F_bar^T (push-forward)
    b_bar_e_trial = F_bar @ b_bar_e_old @ F_bar.transpose()

    # Relaxation: b_bar_e_new = b_bar_e_trial * exp(-dt/tau)
    relax_factor = ti.exp(-dt / tau)
    b_bar_e_relaxed = relax_factor * b_bar_e_trial + (1.0 - relax_factor) * ti.Matrix.identity(ti.f32, 3)

    # SPD projection (ensure positive definiteness and isochoric constraint)
    b_bar_e_new = make_spd(b_bar_e_relaxed, 1e-8)

    # Enforce isochoric constraint: det(b_bar_e) = 1
    det_b = b_bar_e_new.determinant()
    if det_b > 1e-10:
        scale = ti.pow(det_b, -1.0/3.0)
        b_bar_e_new = scale * b_bar_e_new

    # Compute energy correction due to projection
    # ΔE_proj ≈ G/2 * ||b_bar_e_new - b_bar_e_relaxed||^2 (Frobenius norm)
    diff = b_bar_e_new - b_bar_e_relaxed
    delta_E_proj = 0.5 * G * (diff[0,0]**2 + diff[0,1]**2 + diff[0,2]**2 +
                               diff[1,0]**2 + diff[1,1]**2 + diff[1,2]**2 +
                               diff[2,0]**2 + diff[2,1]**2 + diff[2,2]**2)

    # Compute Cauchy stress: tau = G * dev(b_bar_e)
    trace_b = b_bar_e_new[0,0] + b_bar_e_new[1,1] + b_bar_e_new[2,2]
    tau_maxwell = G * (b_bar_e_new - trace_b / 3.0 * ti.Matrix.identity(ti.f32, 3))

    return b_bar_e_new, tau_maxwell, delta_E_proj


@ti.func
def compute_maxwell_stress(
    F: ti.template(),
    b_bar_e: ti.template(),
    dt: ti.f32,
    maxwell_G: ti.template(),
    maxwell_tau: ti.template(),
    n_maxwell: ti.i32
) -> ti.template():
    """
    Compute total Maxwell viscoelastic stress and update internal variables

    Args:
        F: 3x3 deformation gradient
        b_bar_e: (n_maxwell, 3, 3) elastic left Cauchy-Green tensors
        dt: Time step
        maxwell_G: List of shear moduli
        maxwell_tau: List of relaxation times
        n_maxwell: Number of Maxwell branches

    Returns:
        tau_total: Total Cauchy stress from all Maxwell branches
        b_bar_e_new: Updated internal variables
        delta_E_proj_total: Total energy correction
    """
    J = F.determinant()
    tau_total = ti.Matrix.zero(ti.f32, 3, 3)
    delta_E_proj_total = 0.0

    b_bar_e_new = ti.Matrix.zero(ti.f32, 3, 3)  # Placeholder, will be updated in loop

    for k in ti.static(range(n_maxwell)):
        b_bar_e_k_new, tau_k, delta_E_k = update_maxwell_branch(
            F, b_bar_e[k], dt, maxwell_G[k], maxwell_tau[k]
        )
        b_bar_e[k] = b_bar_e_k_new
        tau_total += tau_k
        delta_E_proj_total += delta_E_k

    # Convert Cauchy stress to 1st PK stress: P = J * tau * F^-T
    P_maxwell = J * tau_total @ F.inverse().transpose()

    return P_maxwell, delta_E_proj_total


@ti.func
def compute_maxwell_stress_no_update(
    F: ti.template(),
    b_bar_e: ti.template(),
    G: ti.f32,
    tau: ti.f32
) -> ti.template():
    """
    Compute Maxwell branch stress WITHOUT updating internal variables.
    Used for numerical differentiation in gradient computation.

    Args:
        F: 3x3 deformation gradient
        b_bar_e: 3x3 elastic left Cauchy-Green tensor (current state, not modified)
        G: Shear modulus of this branch
        tau: Relaxation time (not used here, kept for API consistency)

    Returns:
        P_maxwell: 1st PK stress contribution from this branch
    """
    J = F.determinant()

    # Compute Cauchy stress: tau = G * dev(b_bar_e)
    trace_b = b_bar_e[0, 0] + b_bar_e[1, 1] + b_bar_e[2, 2]
    tau_maxwell = G * (b_bar_e - trace_b / 3.0 * ti.Matrix.identity(ti.f32, 3))

    # Convert Cauchy stress to 1st PK stress: P = J * tau * F^-T
    P_maxwell = J * tau_maxwell @ F.inverse().transpose()

    return P_maxwell


@ti.func
def compute_bulk_viscosity_stress_no_energy(
    F: ti.template(),
    F_old: ti.template(),
    dt: ti.f32,
    eta_bulk: ti.f32
) -> ti.template():
    """
    Compute Kelvin-Voigt bulk viscosity stress WITHOUT energy calculation.
    Used for numerical differentiation in gradient computation.

    Args:
        F: Current deformation gradient
        F_old: Previous deformation gradient
        dt: Time step
        eta_bulk: Bulk viscosity coefficient

    Returns:
        P_visc: 1st PK stress from bulk viscosity
    """
    J = F.determinant()
    J_old = F_old.determinant()

    # Volumetric strain rate: J_dot / J ≈ (J - J_old) / (dt * J)
    J_dot = (J - J_old) / dt
    vol_strain_rate = J_dot / J

    # Bulk viscosity stress (Cauchy): sigma_visc = eta_bulk * (J_dot / J) * I
    sigma_visc = eta_bulk * vol_strain_rate * ti.Matrix.identity(ti.f32, 3)

    # Convert to 1st PK stress: P = J * sigma * F^-T
    P_visc = J * sigma_visc @ F.inverse().transpose()

    return P_visc


@ti.func
def compute_bulk_viscosity_stress(F: ti.template(), F_old: ti.template(), dt: ti.f32, eta_bulk: ti.f32) -> ti.template():
    """
    Compute Kelvin-Voigt bulk viscosity stress

    Args:
        F: Current deformation gradient
        F_old: Previous deformation gradient
        dt: Time step
        eta_bulk: Bulk viscosity coefficient

    Returns:
        P_visc: 1st PK stress from bulk viscosity
        delta_E_visc: Viscous dissipation
    """
    J = F.determinant()
    J_old = F_old.determinant()

    # Volumetric strain rate: J_dot / J ≈ (J - J_old) / (dt * J)
    J_dot = (J - J_old) / dt
    vol_strain_rate = J_dot / J

    # Bulk viscosity stress (Cauchy): sigma_visc = eta_bulk * (J_dot / J) * I
    sigma_visc = eta_bulk * vol_strain_rate * ti.Matrix.identity(ti.f32, 3)

    # Convert to 1st PK stress: P = J * sigma * F^-T
    P_visc = J * sigma_visc @ F.inverse().transpose()

    # Viscous dissipation: delta_E = eta_bulk * (J_dot / J)^2 * V * dt
    delta_E_visc = eta_bulk * vol_strain_rate ** 2

    return P_visc, delta_E_visc
