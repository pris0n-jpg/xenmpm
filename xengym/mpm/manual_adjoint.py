"""
Manual Adjoint Implementation for MPM Solver

Implements hand-written backward kernels for P2G/G2P/GridOps/F-update/Maxwell
to bypass Taichi AD's limitation on atomic operations.

Key components:
1. grid_ops_backward: v=P/M normalization backward + BC gradient mapping
2. p2g_backward: Gradient propagation from (g_P_I, g_M_I) to particles
3. g2p_backward: Gradient propagation from particle state to grid velocities
4. update_F_backward: F update backward with SPD STE
5. maxwell_backward: Maxwell internal variable backward (optional)
6. loss_backward: Various loss functions backward

Reference: design.md in openspec/changes/add-mpm-manual-adjoint/
"""
import taichi as ti
import numpy as np
from typing import Optional, Dict, List, Tuple
from .constitutive import compute_ogden_stress_general
from .constitutive_gradients import (
    compute_ogden_stress_with_gradients,
    compute_maxwell_G_gradient,
    compute_bulk_viscosity_gradient,
)


@ti.data_oriented
class ManualAdjointFields:
    """Fields for storing intermediate states and gradients in manual adjoint"""

    def __init__(self, n_particles: int, grid_size: Tuple[int, int, int],
                 n_maxwell: int = 0, max_steps: int = 100):
        """
        Initialize adjoint fields

        Args:
            n_particles: Number of particles
            grid_size: Grid dimensions (nx, ny, nz)
            n_maxwell: Number of Maxwell branches
            max_steps: Maximum simulation steps for state storage
        """
        self.n_particles = n_particles
        self.grid_size = grid_size
        self.n_maxwell = n_maxwell
        self.max_steps = max_steps

        # ============================================
        # Particle gradient fields (accumulated)
        # ============================================
        self.g_x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)  # grad w.r.t position
        self.g_v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)  # grad w.r.t velocity
        self.g_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)  # grad w.r.t F
        self.g_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)  # grad w.r.t C
        self.g_C_saved = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)  # saved g_C for G2P
        self.g_m = ti.field(dtype=ti.f32, shape=n_particles)  # grad w.r.t mass

        # Maxwell internal variable gradients
        if n_maxwell > 0:
            self.g_b_bar_e = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(n_particles, n_maxwell))
        else:
            self.g_b_bar_e = None

        # ============================================
        # Grid gradient fields
        # ============================================
        self.g_grid_v = ti.Vector.field(3, dtype=ti.f32, shape=grid_size)  # grad w.r.t grid velocity
        self.g_grid_P = ti.Vector.field(3, dtype=ti.f32, shape=grid_size)  # grad w.r.t grid momentum
        self.g_grid_M = ti.field(dtype=ti.f32, shape=grid_size)  # grad w.r.t grid mass

        # ============================================
        # Material parameter gradients (scalar accumulators)
        # ============================================
        self.g_ogden_mu = ti.field(dtype=ti.f32, shape=4)  # grad w.r.t Ogden mu
        self.g_ogden_alpha = ti.field(dtype=ti.f32, shape=4)  # grad w.r.t Ogden alpha
        self.g_ogden_kappa = ti.field(dtype=ti.f32, shape=())  # grad w.r.t bulk modulus (scalar)
        self.g_maxwell_G = ti.field(dtype=ti.f32, shape=max(n_maxwell, 1))  # grad w.r.t Maxwell G
        self.g_maxwell_tau = ti.field(dtype=ti.f32, shape=max(n_maxwell, 1))  # grad w.r.t Maxwell tau
        self.g_eta_bulk = ti.field(dtype=ti.f32, shape=())  # grad w.r.t bulk viscosity coefficient

        # ============================================
        # State trajectory storage (for BPTT)
        # ============================================
        # Particle states at each step
        self.x_history = ti.Vector.field(3, dtype=ti.f32, shape=(max_steps, n_particles))
        self.v_history = ti.Vector.field(3, dtype=ti.f32, shape=(max_steps, n_particles))
        self.F_history = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(max_steps, n_particles))
        self.C_history = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(max_steps, n_particles))

        # Maxwell internal variables history (if enabled)
        if n_maxwell > 0:
            self.b_bar_e_history = ti.Matrix.field(3, 3, dtype=ti.f32,
                                                    shape=(max_steps, n_particles, n_maxwell))
            # b_bar_e_trial history (needed for Maxwell backward)
            self.b_bar_e_trial_history = ti.Matrix.field(3, 3, dtype=ti.f32,
                                                          shape=(max_steps, n_particles, n_maxwell))
        else:
            self.b_bar_e_history = None
            self.b_bar_e_trial_history = None

        # Grid mass history (needed for grid ops backward)
        self.grid_M_history = ti.field(dtype=ti.f32, shape=(max_steps,) + grid_size)
        # Grid velocity after normalization (before BC)
        self.grid_v_normalized_history = ti.Vector.field(3, dtype=ti.f32,
                                                          shape=(max_steps,) + grid_size)
        # Grid velocity after BC (for G2P backward)
        self.grid_v_after_BC_history = ti.Vector.field(3, dtype=ti.f32,
                                                        shape=(max_steps,) + grid_size)

        # Stress history (needed for stress backward)
        self.P_history = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(max_steps, n_particles))

        # C_next history: C AFTER G2P at each step (needed for F update backward)
        # C_next_history[n] = C^{n+1} (the C used in F^{n+1} = (I + dt*C^{n+1}) @ F^n)
        self.C_next_history = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(max_steps, n_particles))

        # SPD projection statistics
        self.spd_trigger_count = ti.field(dtype=ti.i32, shape=())
        self.spd_total_count = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def clear_particle_grads(self):
        """Clear particle gradient fields"""
        for p in range(self.n_particles):
            self.g_x[p] = ti.Vector.zero(ti.f32, 3)
            self.g_v[p] = ti.Vector.zero(ti.f32, 3)
            self.g_F[p] = ti.Matrix.zero(ti.f32, 3, 3)
            self.g_C[p] = ti.Matrix.zero(ti.f32, 3, 3)
            self.g_m[p] = 0.0

    @ti.kernel
    def clear_maxwell_grads(self):
        """Clear Maxwell internal variable gradients"""
        if ti.static(self.n_maxwell > 0):
            for p in range(self.n_particles):
                for k in ti.static(range(self.n_maxwell)):
                    self.g_b_bar_e[p, k] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def clear_grid_grads(self):
        """Clear grid gradient fields"""
        for I in ti.grouped(self.g_grid_v):
            self.g_grid_v[I] = ti.Vector.zero(ti.f32, 3)
            self.g_grid_P[I] = ti.Vector.zero(ti.f32, 3)
            self.g_grid_M[I] = 0.0

    @ti.kernel
    def clear_material_grads(self):
        """Clear material parameter gradients"""
        for i in range(4):
            self.g_ogden_mu[i] = 0.0
            self.g_ogden_alpha[i] = 0.0
        self.g_ogden_kappa[None] = 0.0  # bulk modulus gradient
        for k in range(ti.static(max(self.n_maxwell, 1))):
            self.g_maxwell_G[k] = 0.0
            self.g_maxwell_tau[k] = 0.0
        self.g_eta_bulk[None] = 0.0

    def clear_all_grads(self):
        """Clear all gradient fields"""
        self.clear_particle_grads()
        if self.n_maxwell > 0:
            self.clear_maxwell_grads()
        self.clear_grid_grads()
        self.clear_material_grads()

    @ti.kernel
    def clear_spd_stats(self):
        """Clear SPD projection statistics"""
        self.spd_trigger_count[None] = 0
        self.spd_total_count[None] = 0


# ============================================
# Quadratic B-spline weight and its derivative
# ============================================
@ti.func
def bspline_weight(fx: ti.template()) -> ti.template():
    """
    Compute quadratic B-spline weights for 3x3x3 stencil

    Args:
        fx: Fractional position in cell (0 to 1 for each axis)

    Returns:
        w: List of 3 weight vectors for each axis
    """
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    return [w0, w1, w2]


@ti.func
def bspline_weight_gradient(fx: ti.template(), inv_dx: ti.f32) -> ti.template():
    """
    Compute gradient of quadratic B-spline weights w.r.t. particle position

    For quadratic B-spline:
    w0(u) = 0.5 * (1.5 - u)^2   -> dw0/du = -(1.5 - u)
    w1(u) = 0.75 - (u - 1)^2   -> dw1/du = -2(u - 1)
    w2(u) = 0.5 * (u - 0.5)^2  -> dw2/du = (u - 0.5)

    Since fx = x * inv_dx - base, and base is integer, dfx/dx = inv_dx
    So dw/dx = dw/dfx * inv_dx

    Args:
        fx: Fractional position in cell (vector)
        inv_dx: Inverse grid spacing

    Returns:
        dw: List of 3 gradient vectors for each axis
    """
    dw0 = -(1.5 - fx) * inv_dx
    dw1 = -2.0 * (fx - 1.0) * inv_dx
    dw2 = (fx - 0.5) * inv_dx
    return [dw0, dw1, dw2]


# ============================================
# Grid Operations Backward
# ============================================
@ti.kernel
def grid_ops_backward_kernel(
    g_grid_v_after: ti.template(),  # Input: grad w.r.t. grid_v after all ops (3D)
    grid_v_normalized_history: ti.template(),  # Input: v = P/M history (4D: step, i, j, k)
    grid_M_history: ti.template(),  # Input: grid mass history (4D: step, i, j, k)
    g_grid_P: ti.template(),  # Output: grad w.r.t. momentum (3D)
    g_grid_M: ti.template(),  # Output: grad w.r.t. mass (3D)
    step: ti.i32,  # Time step index for history access
    grid_size: ti.template(),
    eps: ti.f32
):
    """
    Backward pass for grid operations: BC -> gravity -> v=P/M

    Forward:
        v_raw = P / M (if M > eps)
        v_after_gravity = v_raw + dt * gravity
        v_after_bc = apply_BC(v_after_gravity)

    Backward:
        1. BC backward: g_v_before_bc = BC_grad(g_v_after_bc)
           - Sticky: g_v_n = 0 (truncate normal gradient)
           - Slip: g_v_n = 0, g_v_t = g_v_t
           - Bounce: g_v_n_orig = -g_v_n
        2. Gravity backward: g_v_raw = g_v_after_gravity (gravity is additive constant)
        3. Normalization backward:
           - g_P = g_v / M
           - g_M = -(g_v · v) / M
    """
    for i, j, k in ti.ndrange(grid_size[0], grid_size[1], grid_size[2]):
        M_I = grid_M_history[step, i, j, k]

        if M_I > eps:
            # Get gradient from after BC (input)
            g_v_after = g_grid_v_after[i, j, k]

            # BC backward: For simplicity, assume sticky BC at boundaries
            # Normal gradient is truncated at walls
            g_v_before_bc = g_v_after

            # Check if at boundary (sticky BC)
            if i < 3 or i >= grid_size[0] - 3:
                g_v_before_bc[0] = 0.0
            if j < 3 or j >= grid_size[1] - 3:
                g_v_before_bc[1] = 0.0
            if k < 3 or k >= grid_size[2] - 3:
                g_v_before_bc[2] = 0.0

            # Gravity backward: g_v_raw = g_v_after_gravity (additive)
            g_v_raw = g_v_before_bc

            # Normalization backward: v = P / M
            # g_P = g_v / M
            g_grid_P[i, j, k] = g_v_raw / M_I

            # g_M = -(g_v · v) / M
            v_I = grid_v_normalized_history[step, i, j, k]
            g_grid_M[i, j, k] = -g_v_raw.dot(v_I) / M_I
        else:
            g_grid_P[i, j, k] = ti.Vector.zero(ti.f32, 3)
            g_grid_M[i, j, k] = 0.0


# ============================================
# P2G Backward
# ============================================
@ti.kernel
def p2g_backward_kernel(
    # Input gradients
    g_grid_P: ti.template(),  # grad w.r.t. grid momentum
    g_grid_M: ti.template(),  # grad w.r.t. grid mass
    # Particle state (from forward)
    x: ti.template(),
    v: ti.template(),
    F: ti.template(),
    C: ti.template(),
    mass: ti.template(),
    volume: ti.template(),
    # Output gradients
    g_x: ti.template(),
    g_v: ti.template(),
    g_F: ti.template(),
    g_C: ti.template(),
    g_m: ti.template(),
    # Material parameters (for stress gradient)
    ogden_mu: ti.template(),
    ogden_alpha: ti.template(),
    n_ogden: ti.i32,
    ogden_kappa: ti.f32,
    g_ogden_mu: ti.template(),
    g_ogden_alpha: ti.template(),
    g_ogden_kappa: ti.template(),  # Scalar field for bulk modulus gradient
    # Grid parameters
    inv_dx: ti.f32,
    dx: ti.f32,
    dt: ti.f32,
    n_particles: ti.i32,
    grid_size: ti.template(),
    # Stored P_total from forward pass
    P_history: ti.template(),
    step: ti.i32
):
    """
    Backward pass for P2G transfer

    Forward P2G:
        For each particle p:
            For each neighbor grid node I:
                w_ip = weight(x_p, x_I)
                v_apic = v_p + C_p @ (x_I - x_p)
                P_I += w_ip * m_p * v_apic + stress_contribution
                M_I += w_ip * m_p

    Backward P2G:
        g_v_p += Σ_I w_ip * m_p * g_P_I
        g_C_p += Σ_I w_ip * m_p * (g_P_I ⊗ d_ip^T)
        g_x_p += Σ_I (∂w_ip/∂x_p)(ΔP_ip · g_P_I + m_p * g_M_I)  [weight derivative]
               + Σ_I w_ip * m_p * (-C_p)^T * g_P_I  [APIC affine term]
        g_m_p += Σ_I w_ip * (g_P_I · v_apic) + Σ_I w_ip * g_M_I
        g_F_p, g_θ: from stress term backward
    """
    for p in range(n_particles):
        x_p = x[p]
        v_p = v[p]
        F_p = F[p]
        C_p = C[p]
        m_p = mass[p]
        V_p = volume[p]

        # Grid node base index
        base = ti.cast(x_p * inv_dx - 0.5, ti.i32)
        fx = x_p * inv_dx - ti.cast(base, ti.f32)

        # Weights and weight gradients
        w = bspline_weight(fx)
        dw = bspline_weight_gradient(fx, inv_dx)

        # Accumulate gradients
        g_v_p = ti.Vector.zero(ti.f32, 3)
        g_C_p = ti.Matrix.zero(ti.f32, 3, 3)
        g_x_p_weight = ti.Vector.zero(ti.f32, 3)  # From weight derivative
        g_x_p_affine = ti.Vector.zero(ti.f32, 3)  # From APIC affine term
        g_m_p = 0.0

        # Loop over 3x3x3 stencil
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            grid_idx = base + offset

            # Check bounds
            in_bounds = (0 <= grid_idx[0] < grid_size[0] and
                        0 <= grid_idx[1] < grid_size[1] and
                        0 <= grid_idx[2] < grid_size[2])

            if in_bounds:
                # Weight
                weight = w[i][0] * w[j][1] * w[k][2]

                # d_ip = x_I - x_p in local coords
                dpos = (ti.cast(offset, ti.f32) - fx) * dx

                # APIC velocity
                v_apic = v_p + C_p @ dpos

                # Get gradients from grid
                g_P_I = g_grid_P[grid_idx]
                g_M_I = g_grid_M[grid_idx]

                # ----------------------------------------
                # 1. g_v_p += w_ip * m_p * g_P_I
                # ----------------------------------------
                g_v_p += weight * m_p * g_P_I

                # ----------------------------------------
                # 2. g_C_p += w_ip * m_p * (g_P_I ⊗ dpos^T)
                # ----------------------------------------
                g_C_p += weight * m_p * g_P_I.outer_product(dpos)

                # ----------------------------------------
                # 3a. g_x_p (weight derivative term)
                # g_x_p += (∂w_ip/∂x_p) * (ΔP_ip · g_P_I + m_p * g_M_I)
                # ----------------------------------------
                # Weight gradient: ∂w/∂x_p = dw[i] * w[j] * w[k] for x-component, etc.
                dw_dx = ti.Vector([
                    dw[i][0] * w[j][1] * w[k][2],
                    w[i][0] * dw[j][1] * w[k][2],
                    w[i][0] * w[j][1] * dw[k][2]
                ])

                # ΔP_ip = m_p * v_apic (momentum contribution)
                delta_P = m_p * v_apic
                scalar_contrib = delta_P.dot(g_P_I) + m_p * g_M_I
                g_x_p_weight += dw_dx * scalar_contrib

                # ----------------------------------------
                # 3b. g_x_p (APIC affine term): -C_p^T @ g_P_I
                # From: v_apic = v_p + C_p @ (x_I - x_p)
                #       ∂v_apic/∂x_p = -C_p
                # So: ∂(m_p * v_apic)/∂x_p = -m_p * C_p
                # And: g_x_p += w_ip * (-m_p * C_p)^T @ g_P_I
                # ----------------------------------------
                g_x_p_affine += weight * m_p * (-C_p.transpose()) @ g_P_I

                # ----------------------------------------
                # 4. g_m_p += w_ip * (g_P_I · v_apic) + w_ip * g_M_I
                # ----------------------------------------
                g_m_p += weight * (g_P_I.dot(v_apic) + g_M_I)

        # ============================================
        # 5. Stress term backward: affine_stress = V_p * P @ F^T
        # Forward: grid_P += weight * affine_stress @ dpos
        # ============================================
        # Accumulate g_affine from all grid nodes
        g_affine = ti.Matrix.zero(ti.f32, 3, 3)
        for i2, j2, k2 in ti.static(ti.ndrange(3, 3, 3)):
            offset2 = ti.Vector([i2, j2, k2])
            grid_idx2 = base + offset2

            in_bounds2 = (0 <= grid_idx2[0] < grid_size[0] and
                         0 <= grid_idx2[1] < grid_size[1] and
                         0 <= grid_idx2[2] < grid_size[2])

            if in_bounds2:
                weight2 = w[i2][0] * w[j2][1] * w[k2][2]
                dpos2 = (ti.cast(offset2, ti.f32) - fx) * dx
                g_P_I2 = g_grid_P[grid_idx2]

                # g_affine += weight * g_P_I ⊗ dpos^T
                g_affine += weight2 * g_P_I2.outer_product(dpos2)

        # ============================================
        # 6-7. Exact stress gradients using analytical derivatives
        # Compute ∂P/∂F, ∂P/∂μ, ∂P/∂α using constitutive_gradients
        # ============================================

        # Forward: grid_v += weight * (V_p * P @ F^T) @ dpos
        # Let A = V_p * P @ F^T, then grid_v += weight * A @ dpos
        #
        # Backward chain rule for A = V_p * P @ F^T:
        #   g_A = g_affine (accumulated from grid nodes)
        #   For Y = P @ F^T: g_P = g_Y @ F, g_F = g_Y^T @ P
        #   Therefore:
        #     g_P = V_p * g_affine @ F
        #     g_F = V_p * g_affine^T @ P

        # Use stored P_total from forward pass (includes elastic + Maxwell + bulk viscosity)
        # This ensures correct gradient computation for all stress components
        P_total = P_history[step, p]

        # Gradient w.r.t. P: g_P = V_p * g_affine @ F
        g_P_stress = V_p * (g_affine @ F_p)

        # Call analytical gradient function (returns tuple with g_kappa)
        g_F_local, g_mu_local, g_alpha_local, g_kappa_local = compute_ogden_stress_with_gradients(
            F_p, ogden_mu, ogden_alpha, n_ogden, ogden_kappa, g_P_stress
        )

        # Chain term for F: g_F = V_p * g_affine^T @ P
        g_F_chain = V_p * (g_affine.transpose() @ P_total)

        # Accumulate to global gradients
        g_F[p] += g_F_local + g_F_chain

        # Accumulate material parameter gradients (atomic for thread safety)
        for k in ti.static(range(4)):
            if k < n_ogden:
                g_ogden_mu[k] += g_mu_local[k]
                g_ogden_alpha[k] += g_alpha_local[k]

        # Accumulate bulk modulus gradient (scalar, atomic add)
        g_ogden_kappa[None] += g_kappa_local

        # Accumulate to output (atomic add for thread safety)
        g_x[p] += g_x_p_weight + g_x_p_affine
        # IMPORTANT: g_v is REPLACED, not accumulated!
        # In MPM, v^n is only used in P2G at step n. It does NOT directly
        # connect to v^{n+1} (velocity goes through grid). Therefore, the
        # gradient g_v^n should only come from P2G backward at step n.
        # Using += causes exponential gradient growth across steps!
        g_v[p] = g_v_p
        # IMPORTANT: g_C is REPLACED, not accumulated!
        # g_C^n = APIC contribution only (C^n is only used in P2G at step n)
        # The g_C^{n+1} value was already used by G2P backward and contains
        # contributions from both APIC^{n+1} and F_update at step n.
        g_C[p] = g_C_p
        g_m[p] += g_m_p


# ============================================
# G2P Backward
# ============================================
@ti.kernel
def g2p_backward_kernel(
    # Input gradients (from loss or next step)
    g_x_next: ti.template(),  # grad w.r.t. x^{n+1}
    g_v_next: ti.template(),  # grad w.r.t. v^{n+1}
    g_C_next: ti.template(),  # grad w.r.t. C^{n+1}
    # Current state
    x: ti.template(),  # x^n (position at this step)
    # Grid state
    grid_v: ti.template(),  # v_I after grid ops
    # Output gradients
    g_grid_v: ti.template(),  # grad w.r.t. grid velocity
    g_x_curr: ti.template(),  # grad w.r.t. x^n (accumulated)
    # Parameters
    inv_dx: ti.f32,
    dx: ti.f32,
    dt: ti.f32,
    n_particles: ti.i32,
    grid_size: ti.template()
):
    """
    Backward pass for G2P transfer

    Forward G2P:
        v_p^{n+1} = Σ_I w_ip * v_I
        C_p^{n+1} = κ * Σ_I w_ip * v_I ⊗ d_ip^T  (κ = 4/dx²)
        x_p^{n+1} = x_p^n + dt * v_p^{n+1}

    Backward G2P:
        From x update: x_p^{n+1} = x_p^n + dt * v_p^{n+1}
            g_x_p^n += g_x_p^{n+1}
            g_v_p^{n+1} += dt * g_x_p^{n+1}

        From v interpolation:
            g_v_I += w_ip * g_v_p^{n+1}
            g_x_p^n += Σ_I (∂w_ip/∂x_p) * (v_I · g_v_p^{n+1})

        From C interpolation:
            g_v_I += κ * w_ip * (g_C_p^{n+1} : d_ip^T)
            g_x_p^n += κ * Σ_I (∂w_ip/∂x_p) * (g_C_p : v_I ⊗ d_ip^T)
                     - κ * Σ_I w_ip * (g_C_p^T @ v_I)  [from d_ip = x_I - x_p]
    """
    kappa = 4.0 * inv_dx * inv_dx  # = 4/dx²

    for p in range(n_particles):
        x_p = x[p]

        # Grid node base index
        base = ti.cast(x_p * inv_dx - 0.5, ti.i32)
        fx = x_p * inv_dx - ti.cast(base, ti.f32)

        # Weights and gradients
        w = bspline_weight(fx)
        dw = bspline_weight_gradient(fx, inv_dx)

        # Get input gradients
        g_x_np1 = g_x_next[p]
        g_v_np1 = g_v_next[p]
        g_C_np1 = g_C_next[p]

        # ----------------------------------------
        # From x update: x^{n+1} = x^n + dt * v^{n+1}
        # ----------------------------------------
        # g_x^n += g_x^{n+1}
        g_x_n = g_x_np1
        # g_v^{n+1} += dt * g_x^{n+1}
        g_v_np1_from_x = dt * g_x_np1
        g_v_np1_total = g_v_np1 + g_v_np1_from_x

        # ----------------------------------------
        # Loop over stencil for v and C interpolation backward
        # ----------------------------------------
        g_x_from_v = ti.Vector.zero(ti.f32, 3)
        g_x_from_C_weight = ti.Vector.zero(ti.f32, 3)
        g_x_from_C_dpos = ti.Vector.zero(ti.f32, 3)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            grid_idx = base + offset

            in_bounds = (0 <= grid_idx[0] < grid_size[0] and
                        0 <= grid_idx[1] < grid_size[1] and
                        0 <= grid_idx[2] < grid_size[2])

            if in_bounds:
                weight = w[i][0] * w[j][1] * w[k][2]
                dpos = (ti.cast(offset, ti.f32) - fx) * dx
                v_I = grid_v[grid_idx]

                # Weight gradient
                dw_dx = ti.Vector([
                    dw[i][0] * w[j][1] * w[k][2],
                    w[i][0] * dw[j][1] * w[k][2],
                    w[i][0] * w[j][1] * dw[k][2]
                ])

                # ----------------------------------------
                # From v interpolation: v^{n+1} = Σ_I w_ip * v_I
                # g_v_I += w_ip * g_v^{n+1}
                # ----------------------------------------
                g_grid_v[grid_idx] += weight * g_v_np1_total

                # g_x^n += Σ_I (∂w/∂x_p) * (v_I · g_v^{n+1})
                g_x_from_v += dw_dx * v_I.dot(g_v_np1_total)

                # ----------------------------------------
                # From C interpolation: C^{n+1} = κ * Σ_I w_ip * v_I ⊗ d_ip^T
                # g_v_I += κ * w_ip * (g_C : d_ip^T) = κ * w_ip * g_C @ d_ip
                # ----------------------------------------
                g_grid_v[grid_idx] += kappa * weight * (g_C_np1 @ dpos)

                # g_x^n from weight derivative in C
                # C_contribution = κ * w_ip * (v_I ⊗ d_ip^T)
                # ∂C/∂x_p from weight: κ * (∂w/∂x_p) * (v_I ⊗ d_ip^T)
                # g_x^n += κ * (∂w/∂x_p) * (g_C : (v_I ⊗ d_ip^T))
                #       = κ * (∂w/∂x_p) * (v_I · (g_C @ d_ip))
                g_x_from_C_weight += kappa * dw_dx * v_I.dot(g_C_np1 @ dpos)

                # g_x^n from d_ip = x_I - x_p in C
                # ∂(v_I ⊗ d_ip^T)/∂x_p = -v_I ⊗ I^T (for each row)
                # More precisely: g_x += -κ * w_ip * g_C^T @ v_I
                g_x_from_C_dpos -= kappa * weight * (g_C_np1.transpose() @ v_I)

        # Accumulate x gradient
        # NOTE: g_x_n is NOT added here because g_x_next and g_x_curr are the same field
        # (aliased). The pass-through gradient (∂L/∂x^{n+1}) is already in g_x_curr.
        # We only add the additional contributions from how x^n affects v^{n+1} and C^{n+1}
        # through the G2P interpolation weights and dpos.
        g_x_curr[p] += g_x_from_v + g_x_from_C_weight + g_x_from_C_dpos


# ============================================
# F Update Backward (with SPD STE)
# ============================================
@ti.kernel
def update_F_backward_kernel(
    # Input gradients
    g_F_new: ti.template(),  # grad w.r.t. F^{n+1}
    # State
    F_old: ti.template(),  # F^n
    C_new: ti.template(),  # C^{n+1} (used in F update)
    # Output gradients
    g_F_old: ti.template(),  # grad w.r.t. F^n
    g_C: ti.template(),  # grad w.r.t. C^{n+1}
    # Parameters
    dt: ti.f32,
    n_particles: ti.i32
):
    """
    Backward pass for F update with SPD STE

    Forward:
        F_raw = (I + dt * C) @ F_old
        F_new = SPD(F_raw)  [with STE: backward treats SPD as identity]

    Backward (STE):
        g_F_raw = g_F_new  [straight-through]
        g_C += dt * g_F_raw @ F_old^T
        g_F_old = (I + dt * C)^T @ g_F_raw  [REPLACE, not accumulate]

    Note: This kernel should be called BEFORE P2G backward, so that g_F
    is properly propagated and then P2G backward can ADD its contribution.
    """
    for p in range(n_particles):
        g_F_np1 = g_F_new[p]
        F_n = F_old[p]
        C_np1 = C_new[p]

        # STE: g_F_raw = g_F_new
        g_F_raw = g_F_np1

        # g_C += dt * g_F_raw @ F_old^T
        g_C[p] += dt * g_F_raw @ F_n.transpose()

        # g_F_old = (I + dt * C)^T @ g_F_raw  [REPLACE]
        I_plus_dtC = ti.Matrix.identity(ti.f32, 3) + dt * C_np1
        g_F_old[p] = I_plus_dtC.transpose() @ g_F_raw


# ============================================
# Maxwell Internal Variable Backward
# ============================================
@ti.kernel
def maxwell_backward_kernel(
    # Input gradients
    g_b_bar_e_new: ti.template(),  # grad w.r.t. b_bar_e^{n+1}
    # State
    b_bar_e_old: ti.template(),  # b_bar_e^n
    b_bar_e_trial: ti.template(),  # b_bar_e_trial (before relaxation)
    # Output gradients
    g_b_bar_e_old: ti.template(),  # grad w.r.t. b_bar_e^n
    g_maxwell_tau: ti.template(),  # grad w.r.t. tau (accumulated)
    # Parameters
    maxwell_tau: ti.template(),
    dt: ti.f32,
    n_particles: ti.i32,
    n_maxwell: ti.i32
):
    """
    Backward pass for Maxwell internal variable update

    Forward:
        a = exp(-dt / tau)
        b_bar_e^{n+1} = a * b_bar_e^n + (1-a) * b_bar_e_trial
        [Note: b_bar_e_trial is computed from F and b_bar_e_old via upper-convected update]

    Backward:
        g_b_bar_e^n += a * g_b_bar_e^{n+1}
        g_b_bar_e_trial += (1-a) * g_b_bar_e^{n+1}
        g_tau += <g_b_bar_e^{n+1}, ∂b/∂tau>

        where ∂a/∂tau = dt/tau² * exp(-dt/tau)
              ∂b/∂tau = ∂a/∂tau * (b_bar_e^n - b_bar_e_trial)
    """
    for p in range(n_particles):
        for k in ti.static(range(n_maxwell)):
            tau_k = maxwell_tau[k]
            a = ti.exp(-dt / tau_k)

            g_b_new = g_b_bar_e_new[p, k]
            b_old = b_bar_e_old[p, k]
            b_trial = b_bar_e_trial[p, k]

            # g_b_bar_e^n += a * g_b_bar_e^{n+1}
            g_b_bar_e_old[p, k] += a * g_b_new

            # g_tau: ∂a/∂tau = dt/tau² * a
            da_dtau = dt / (tau_k * tau_k) * a
            # ∂b^{n+1}/∂tau = da/dtau * (b_old - b_trial)
            db_dtau = da_dtau * (b_old - b_trial)
            # g_tau += Frobenius inner product <g_b_new, db_dtau>
            g_tau_contrib = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    g_tau_contrib += g_b_new[i, j] * db_dtau[i, j]
            g_maxwell_tau[k] += g_tau_contrib


# ============================================
# Maxwell G Gradient from Stress Path
# ============================================
@ti.kernel
def maxwell_G_gradient_kernel(
    # Particle state
    F: ti.template(),
    volume: ti.template(),
    # Maxwell internal variables
    b_bar_e: ti.template(),  # (n_particles, n_maxwell)
    # Grid momentum gradient (from backward pass)
    g_grid_P: ti.template(),
    # Grid parameters
    x: ti.template(),
    inv_dx: ti.f32,
    dx: ti.f32,
    n_particles: ti.i32,
    n_maxwell: ti.i32,
    grid_size: ti.template(),
    # Output gradients
    g_maxwell_G: ti.template()
):
    """
    Compute gradient of loss w.r.t. Maxwell shear modulus G from stress path.

    This kernel computes g_G[k] by:
    1. Computing g_P (gradient w.r.t. stress) from grid momentum gradients
    2. Using the chain rule: g_G[k] = <g_P, ∂P_maxwell/∂G_k>

    Forward path:
        τ_k = G_k * dev(b_bar_e_k)
        P_maxwell_k = J * τ_k @ F^(-T)

    Backward:
        ∂P_maxwell/∂G_k = J * dev(b_bar_e_k) @ F^(-T)
        g_G_k = <g_P, ∂P_maxwell/∂G_k>

    Note: This kernel should be called AFTER p2g_backward to ensure grid
    gradients are properly set up.
    """
    for p in range(n_particles):
        x_p = x[p]
        F_p = F[p]
        V_p = volume[p]

        # Grid node base index
        base = ti.cast(x_p * inv_dx - 0.5, ti.i32)
        fx = x_p * inv_dx - ti.cast(base, ti.f32)

        # Compute weights
        w0 = 0.5 * (1.5 - fx) ** 2
        w1 = 0.75 - (fx - 1.0) ** 2
        w2 = 0.5 * (fx - 0.5) ** 2
        w = [w0, w1, w2]

        # Accumulate g_affine from all grid nodes (same as p2g_backward)
        g_affine = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            grid_idx = base + offset

            in_bounds = (0 <= grid_idx[0] < grid_size[0] and
                        0 <= grid_idx[1] < grid_size[1] and
                        0 <= grid_idx[2] < grid_size[2])

            if in_bounds:
                weight = w[i][0] * w[j][1] * w[k][2]
                dpos = (ti.cast(offset, ti.f32) - fx) * dx
                g_P_I = g_grid_P[grid_idx]

                # g_affine += weight * g_P_I ⊗ dpos^T
                g_affine += weight * g_P_I.outer_product(dpos)

        # Compute g_P from g_affine: g_P = V_p * g_affine @ F
        g_P_stress = V_p * (g_affine @ F_p)

        # Compute Maxwell G gradient for each branch
        J = F_p.determinant()
        F_inv_T = F_p.inverse().transpose()

        for k in ti.static(range(4)):  # Max 4 branches
            if k < n_maxwell:
                # Get internal variable for this branch
                b_bar_e_k = b_bar_e[p, k]

                # Compute deviatoric part: dev(b_bar_e) = b_bar_e - tr(b_bar_e)/3 * I
                trace_b = b_bar_e_k[0, 0] + b_bar_e_k[1, 1] + b_bar_e_k[2, 2]
                dev_b = b_bar_e_k - (trace_b / 3.0) * ti.Matrix.identity(ti.f32, 3)

                # ∂P_maxwell/∂G = J * dev(b_bar_e) @ F^(-T)
                dP_dG = J * (dev_b @ F_inv_T)

                # g_G = <g_P, dP/dG> = Frobenius inner product
                g_G_contrib = 0.0
                for ii in ti.static(range(3)):
                    for jj in ti.static(range(3)):
                        g_G_contrib += g_P_stress[ii, jj] * dP_dG[ii, jj]

                # Accumulate to global gradient (atomic add)
                g_maxwell_G[k] += g_G_contrib


# ============================================
# Bulk Viscosity Gradient from Stress Path
# ============================================
@ti.kernel
def bulk_viscosity_gradient_kernel(
    # Particle state
    F: ti.template(),
    C: ti.template(),  # APIC velocity gradient matrix
    volume: ti.template(),
    # Grid momentum gradient (from backward pass)
    g_grid_P: ti.template(),
    # Grid parameters
    x: ti.template(),
    inv_dx: ti.f32,
    dx: ti.f32,
    n_particles: ti.i32,
    grid_size: ti.template(),
    # Output gradients
    g_eta_bulk: ti.template()
):
    """
    Compute gradient of loss w.r.t. bulk viscosity coefficient eta_bulk.

    This kernel computes g_eta_bulk by:
    1. Computing g_P (gradient w.r.t. stress) from grid momentum gradients
    2. Using the chain rule: g_eta_bulk = <g_P, ∂P_visc/∂eta_bulk>

    Forward path:
        sigma_visc = eta_bulk * tr(L) * I
        P_visc = J * sigma_visc @ F^(-T)

    Backward:
        ∂P_visc/∂eta_bulk = J * tr(L) * I @ F^(-T)
        g_eta_bulk = Σ_p <g_P_p, ∂P_visc/∂eta_bulk>

    Note: This kernel should be called AFTER p2g_backward to ensure grid
    gradients are properly set up.
    """
    for p in range(n_particles):
        x_p = x[p]
        F_p = F[p]
        C_p = C[p]
        V_p = volume[p]

        # Grid node base index
        base = ti.cast(x_p * inv_dx - 0.5, ti.i32)
        fx = x_p * inv_dx - ti.cast(base, ti.f32)

        # Compute weights
        w0 = 0.5 * (1.5 - fx) ** 2
        w1 = 0.75 - (fx - 1.0) ** 2
        w2 = 0.5 * (fx - 0.5) ** 2
        w = [w0, w1, w2]

        # Accumulate g_affine from all grid nodes (same as p2g_backward)
        g_affine = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            grid_idx = base + offset

            in_bounds = (0 <= grid_idx[0] < grid_size[0] and
                        0 <= grid_idx[1] < grid_size[1] and
                        0 <= grid_idx[2] < grid_size[2])

            if in_bounds:
                weight = w[i][0] * w[j][1] * w[k][2]
                dpos = (ti.cast(offset, ti.f32) - fx) * dx
                g_P_I = g_grid_P[grid_idx]

                # g_affine += weight * g_P_I ⊗ dpos^T
                g_affine += weight * g_P_I.outer_product(dpos)

        # Compute g_P from g_affine: g_P = V_p * g_affine @ F
        g_P_stress = V_p * (g_affine @ F_p)

        # Compute bulk viscosity gradient
        J = F_p.determinant()
        trace_L = C_p[0, 0] + C_p[1, 1] + C_p[2, 2]
        F_inv_T = F_p.inverse().transpose()

        # ∂P_visc/∂eta_bulk = J * tr(L) * I @ F^(-T) = J * tr(L) * F^(-T)
        dP_deta = J * trace_L * F_inv_T

        # g_eta_bulk = <g_P, dP/deta> = Frobenius inner product
        g_eta_contrib = 0.0
        for ii in ti.static(range(3)):
            for jj in ti.static(range(3)):
                g_eta_contrib += g_P_stress[ii, jj] * dP_deta[ii, jj]

        # Accumulate to global gradient (atomic add)
        g_eta_bulk[None] += g_eta_contrib


# ============================================
# Loss Backward Kernels
# ============================================
@ti.kernel
def position_loss_backward_kernel(
    x: ti.template(),
    target_x: ti.template(),
    g_x: ti.template(),
    loss: ti.template(),
    n_particles: ti.i32
):
    """
    Backward for position loss: L = 0.5 * Σ_p ||x_p - target_x_p||²
    g_x_p = x_p - target_x_p
    """
    for p in range(n_particles):
        diff = x[p] - target_x[p]
        g_x[p] += diff
        loss[None] += 0.5 * diff.dot(diff)


@ti.kernel
def velocity_loss_backward_kernel(
    v: ti.template(),
    target_v: ti.template(),
    g_v: ti.template(),
    loss: ti.template(),
    n_particles: ti.i32
):
    """
    Backward for velocity loss: L = 0.5 * Σ_p ||v_p - target_v_p||²
    g_v_p = v_p - target_v_p
    """
    for p in range(n_particles):
        diff = v[p] - target_v[p]
        g_v[p] += diff
        loss[None] += 0.5 * diff.dot(diff)


@ti.kernel
def kinetic_energy_loss_backward_kernel(
    v: ti.template(),
    mass: ti.template(),
    g_v: ti.template(),
    loss: ti.template(),
    target_energy: ti.f32,
    n_particles: ti.i32
):
    """
    Backward for kinetic energy loss: L = 0.5 * (E_kin - E_target)²
    E_kin = 0.5 * Σ_p m_p * ||v_p||²

    dL/dE_kin = E_kin - E_target
    dE_kin/dv_p = m_p * v_p

    g_v_p = (E_kin - E_target) * m_p * v_p
    """
    # First compute E_kin
    E_kin = 0.0
    for p in range(n_particles):
        E_kin += 0.5 * mass[p] * v[p].dot(v[p])

    # Then compute gradient
    diff = E_kin - target_energy
    loss[None] = 0.5 * diff * diff

    for p in range(n_particles):
        g_v[p] += diff * mass[p] * v[p]


@ti.kernel
def total_energy_loss_backward_kernel(
    # Particle state
    x: ti.template(),
    v: ti.template(),
    F: ti.template(),
    mass: ti.template(),
    volume: ti.template(),
    # Material parameters
    ogden_mu: ti.template(),
    ogden_alpha: ti.template(),
    n_ogden: ti.i32,
    ogden_kappa: ti.f32,
    # Output gradients
    g_v: ti.template(),
    g_F: ti.template(),
    loss: ti.template(),
    # Target
    target_energy: ti.f32,
    n_particles: ti.i32
):
    """
    Backward for total energy loss: L = 0.5 * (E_total - E_target)²
    E_total = E_kin + E_elastic
    E_kin = 0.5 * Σ_p m_p * ||v_p||²
    E_elastic = Σ_p V_p * ψ(F_p)

    dL/dE_total = E_total - E_target

    For kinetic energy:
        dE_kin/dv_p = m_p * v_p
        g_v_p = (E_total - E_target) * m_p * v_p

    For elastic energy:
        dE_elastic/dF_p = V_p * P_p  (where P = ∂ψ/∂F is the 1st PK stress)
        g_F_p = (E_total - E_target) * V_p * P_p
    """
    # First compute total energy
    E_kin = 0.0
    E_elastic = 0.0

    for p in range(n_particles):
        # Kinetic energy
        E_kin += 0.5 * mass[p] * v[p].dot(v[p])

        # Elastic energy (computed via Ogden model)
        F_p = F[p]
        V_p = volume[p]
        _, psi = compute_ogden_stress_general(
            F_p, ogden_mu, ogden_alpha, n_ogden, ogden_kappa
        )
        E_elastic += V_p * psi

    E_total = E_kin + E_elastic

    # Compute loss
    diff = E_total - target_energy
    loss[None] = 0.5 * diff * diff

    # Compute gradients
    for p in range(n_particles):
        # Kinetic energy gradient: g_v = diff * m * v
        g_v[p] += diff * mass[p] * v[p]

        # Elastic energy gradient: g_F = diff * V * P
        F_p = F[p]
        V_p = volume[p]
        P_p, _ = compute_ogden_stress_general(
            F_p, ogden_mu, ogden_alpha, n_ogden, ogden_kappa
        )
        g_F[p] += diff * V_p * P_p
