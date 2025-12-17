"""
Manual Adjoint MPM Solver

High-level interface for differentiable MPM simulation using hand-written adjoints.
This bypasses Taichi AD's limitation on atomic operations in P2G/G2P kernels.

This is the **canonical gradient path** for MPM in xengym, as defined in
the refine-mpm-ad-precision specification. Taichi Tape-based autodiff is
blocked on P2G/G2P atomics and will raise errors.

Gradient Accuracy Tiers (see conftest.py for thresholds):
- Tier A (constitutive/stress level): rel_error ≤ 1%, cosine_sim ≥ 0.99
- Tier B (small MPM toy scenes): rel_error ≤ 5%, cosine_sim ≥ 0.95
- Tier C (high-deformation end-to-end): rel_error ≤ 50%, cosine_sim ≥ 0.80

Usage:
    solver = ManualAdjointMPMSolver(config, n_particles)
    solver.initialize_particles(positions, velocities)
    solver.set_target_positions(target)

    # Run forward + backward
    result = solver.solve_with_gradients(num_steps, loss_type='position')

    # Access gradients
    grad_mu = result['grad_ogden_mu']
    grad_kappa = result['grad_ogden_kappa']  # bulk modulus
    grad_x0 = result['grad_initial_x']
"""
from __future__ import annotations
from typing import Dict, Optional, Union

import taichi as ti
import numpy as np
from numpy.typing import NDArray

from .mpm_solver import MPMSolver
from .config import MPMConfig
from .manual_adjoint import (
    ManualAdjointFields,
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
from .constitutive import compute_ogden_stress_general
from .constitutive_gradients import (
    validate_gradient_mode,
    is_experimental_mode_enabled,
    get_scale_guards,
)
from .exceptions import GradientError, ScaleGuardError, TargetNotSetError


@ti.data_oriented
class ManualAdjointMPMSolver:
    """
    Differentiable MPM Solver using Manual Adjoint Method

    This solver implements BPTT (Backpropagation Through Time) with hand-written
    adjoint kernels to support gradient computation w.r.t.:
    - Material parameters (Ogden mu/alpha, Maxwell G/tau)
    - Initial particle state (x0, v0, F0)

    The manual adjoint approach bypasses Taichi AD's limitation on atomic
    scatter/gather operations in P2G/G2P kernels.
    """

    def __init__(self, config: MPMConfig, n_particles: int,
                 max_grad_steps: int = 100,
                 maxwell_needs_grad: bool = False):
        """
        Initialize manual adjoint MPM solver

        Args:
            config: MPM configuration
            n_particles: Number of particles
            max_grad_steps: Maximum steps for gradient computation (state storage limit)
            maxwell_needs_grad: If True, compute gradients for Maxwell parameters
        """
        self.config = config
        self.n_particles = n_particles
        self.max_grad_steps = max_grad_steps
        self.maxwell_needs_grad = maxwell_needs_grad

        # Validate gradient mode with scale guards (strict=True blocks on issues)
        # This ensures experimental P_total mode respects particle/step limits
        validate_gradient_mode(
            config,
            strict=True,
            n_particles=n_particles,
            n_steps=max_grad_steps
        )

        # Create base solver (without Taichi AD)
        self.solver = MPMSolver(config, n_particles, enable_grad=False, use_spd_ste=True)

        # Create adjoint fields
        self.n_maxwell = len(config.material.maxwell_branches)
        self.adj_fields = ManualAdjointFields(
            n_particles=n_particles,
            grid_size=tuple(config.grid.grid_size),
            n_maxwell=self.n_maxwell if maxwell_needs_grad else 0,
            max_steps=max_grad_steps
        )

        # Target fields for loss computation
        self._target_x = None
        self._target_v = None
        self._target_energy = ti.field(dtype=ti.f32, shape=())
        self._target_total_energy = ti.field(dtype=ti.f32, shape=())

        # Loss field
        self.loss_field = ti.field(dtype=ti.f32, shape=())

        # Grid parameters
        self.dx = config.grid.dx
        self.inv_dx = 1.0 / self.dx
        self.dt = config.time.dt
        self.grid_size = tuple(config.grid.grid_size)
        self.eps = 1e-10

        # Current step counter
        self._current_forward_step = 0

        # Step field for kernel access (Taichi 1.7+ requires type-annotated kernel params)
        self._step_field = ti.field(dtype=ti.i32, shape=())

    def initialize_particles(
        self,
        positions: NDArray[np.float32],
        velocities: Optional[NDArray[np.float32]] = None,
        volumes: Optional[NDArray[np.float32]] = None
    ) -> None:
        """Initialize particle data"""
        self.solver.initialize_particles(positions, velocities, volumes)

    def set_target_positions(self, target_positions: NDArray[np.float32]) -> None:
        """Set target positions for position loss"""
        if self._target_x is None:
            self._target_x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self._target_x.from_numpy(target_positions.astype(np.float32))

    def set_target_velocities(self, target_velocities: NDArray[np.float32]) -> None:
        """Set target velocities for velocity loss"""
        if self._target_v is None:
            self._target_v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self._target_v.from_numpy(target_velocities.astype(np.float32))

    def set_target_energy(self, target_energy: float) -> None:
        """Set target kinetic energy for energy loss"""
        self._target_energy[None] = target_energy

    def set_target_total_energy(self, target_total_energy: float) -> None:
        """Set target total energy (kinetic + elastic) for total energy loss"""
        self._target_total_energy[None] = target_total_energy

    # ============================================
    # State Storage Kernels
    # ============================================
    @ti.kernel
    def _store_particle_state(self):
        """Store particle state at given step for backward pass"""
        step = self._step_field[None]
        for p in range(self.n_particles):
            self.adj_fields.x_history[step, p] = self.solver.fields.x[p]
            self.adj_fields.v_history[step, p] = self.solver.fields.v[p]
            self.adj_fields.F_history[step, p] = self.solver.fields.F[p]
            self.adj_fields.C_history[step, p] = self.solver.fields.C[p]

    @ti.kernel
    def _store_maxwell_state(self):
        """Store Maxwell internal variable state"""
        step = self._step_field[None]
        if ti.static(self.maxwell_needs_grad and self.n_maxwell > 0):
            for p in range(self.n_particles):
                for k in ti.static(range(self.n_maxwell)):
                    self.adj_fields.b_bar_e_history[step, p, k] = self.solver.fields.b_bar_e[p, k]

    @ti.kernel
    def _store_maxwell_trial_state(self):
        """Store b_bar_e_trial for Maxwell backward

        Called after update_F_and_internal(). Uses:
        - Current F (which is F^{n+1} after the update)
        - Stored b_bar_e_old from b_bar_e_history[step] (which is b_bar_e^n before update)

        Computes: b_bar_e_trial = F_bar @ b_bar_e_old @ F_bar^T
        where F_bar = J^{-1/3} * F (isochoric part)
        """
        step = self._step_field[None]
        if ti.static(self.maxwell_needs_grad and self.n_maxwell > 0):
            for p in range(self.n_particles):
                # Get updated F (F^{n+1})
                F_new = self.solver.fields.F[p]
                J_new = F_new.determinant()

                # Compute isochoric deformation gradient
                F_bar = ti.pow(J_new, -1.0/3.0) * F_new

                for k in ti.static(range(self.n_maxwell)):
                    # Get stored b_bar_e_old (from before update)
                    b_bar_e_old = self.adj_fields.b_bar_e_history[step, p, k]

                    # Compute trial: upper-convected update
                    b_bar_e_trial = F_bar @ b_bar_e_old @ F_bar.transpose()

                    # Store for backward
                    self.adj_fields.b_bar_e_trial_history[step, p, k] = b_bar_e_trial

    @ti.kernel
    def _store_P_total(self):
        """Store complete P_total (elastic + Maxwell + bulk viscosity) for backward

        This is critical for correct gradient computation in p2g_backward.
        The backward pass needs the FULL P_total used in forward, not just P_elastic.
        """
        step = self._step_field[None]
        for p in range(self.n_particles):
            F_p = self.solver.fields.F[p]
            C_p = self.solver.fields.C[p]

            # Compute elastic stress (same as forward)
            P_elastic, _ = compute_ogden_stress_general(
                F_p,
                self.solver.ogden_mu,
                self.solver.ogden_alpha,
                self.solver.n_ogden,
                self.solver.ogden_kappa
            )

            P_total = P_elastic

            # Add Maxwell stress if enabled
            if ti.static(self.n_maxwell > 0):
                J = F_p.determinant()
                tau_maxwell_total = ti.Matrix.zero(ti.f32, 3, 3)

                for k in ti.static(range(self.n_maxwell)):
                    b_bar_e_k = self.solver.fields.b_bar_e[p, k]
                    G_k = self.solver.maxwell_G[k]

                    trace_b = b_bar_e_k[0, 0] + b_bar_e_k[1, 1] + b_bar_e_k[2, 2]
                    tau_k = G_k * (b_bar_e_k - trace_b / 3.0 * ti.Matrix.identity(ti.f32, 3))
                    tau_maxwell_total += tau_k

                P_maxwell = J * tau_maxwell_total @ F_p.inverse().transpose()
                P_total = P_elastic + P_maxwell

            # Add bulk viscosity stress if enabled
            if ti.static(self.config.material.enable_bulk_viscosity):
                L = C_p
                trace_L = L[0, 0] + L[1, 1] + L[2, 2]
                eta_bulk = self.config.material.bulk_viscosity
                sigma_visc = eta_bulk * trace_L * ti.Matrix.identity(ti.f32, 3)
                J = F_p.determinant()
                P_visc = J * sigma_visc @ F_p.inverse().transpose()
                P_total += P_visc

            # Store complete P_total
            self.adj_fields.P_history[step, p] = P_total

    @ti.kernel
    def _store_grid_state(self):
        """Store grid state (mass and normalized velocity v=P/M) for backward

        Called after P2G, grid_v contains momentum P.
        We compute and store v_raw = P/M (normalized velocity before BC).
        This is what grid_ops_backward expects.
        """
        step = self._step_field[None]
        for I in ti.grouped(self.solver.fields.grid_m):
            M_I = self.solver.fields.grid_m[I]
            self.adj_fields.grid_M_history[step, I[0], I[1], I[2]] = M_I

            # Compute v_raw = P/M (normalized velocity before BC and gravity)
            # grid_v at this point contains momentum P from P2G
            v_raw = ti.Vector.zero(ti.f32, 3)
            if M_I > 1e-10:
                v_raw = self.solver.fields.grid_v[I] / M_I

            self.adj_fields.grid_v_normalized_history[step, I[0], I[1], I[2]] = v_raw

    @ti.kernel
    def _restore_particle_state(self):
        """Restore particle state from history for backward computation"""
        step = self._step_field[None]
        for p in range(self.n_particles):
            self.solver.fields.x[p] = self.adj_fields.x_history[step, p]
            self.solver.fields.v[p] = self.adj_fields.v_history[step, p]
            self.solver.fields.F[p] = self.adj_fields.F_history[step, p]
            self.solver.fields.C[p] = self.adj_fields.C_history[step, p]

    @ti.kernel
    def _save_g_C_for_g2p(self):
        """Save current g_C before it gets modified by P2G and F_update backward"""
        for p in range(self.n_particles):
            self.adj_fields.g_C_saved[p] = self.adj_fields.g_C[p]

    @ti.kernel
    def _store_C_next(self):
        """Store C AFTER G2P (C^{n+1}) for F update backward

        F update: F^{n+1} = (I + dt * C^{n+1}) @ F^n
        So backward needs C^{n+1}, not C^n.
        """
        step = self._step_field[None]
        for p in range(self.n_particles):
            self.adj_fields.C_next_history[step, p] = self.solver.fields.C[p]

    @ti.kernel
    def _store_grid_v_after_BC(self):
        """Store grid_v AFTER grid_ops (for G2P backward)

        Called after grid_op(), grid_v contains velocity after normalization,
        gravity, and boundary conditions. This is what G2P uses in forward.
        """
        step = self._step_field[None]
        for I in ti.grouped(self.solver.fields.grid_v):
            self.adj_fields.grid_v_after_BC_history[step, I[0], I[1], I[2]] = self.solver.fields.grid_v[I]

    @ti.kernel
    def _load_grid_v_for_g2p_backward(self):
        """Load stored grid_v for G2P backward

        This loads grid_v from history into the solver's grid_v field,
        so that G2P backward uses the correct grid_v for step n.
        """
        step = self._step_field[None]
        for I in ti.grouped(self.solver.fields.grid_v):
            self.solver.fields.grid_v[I] = self.adj_fields.grid_v_after_BC_history[step, I[0], I[1], I[2]]

    # ============================================
    # Forward Pass with State Storage
    # ============================================
    def forward_step_with_storage(self, step: int):
        """Execute one forward step and store state for backward"""
        # Set step in field for kernel access
        self._step_field[None] = step

        # Store state BEFORE the step
        self._store_particle_state()
        if self.maxwell_needs_grad and self.n_maxwell > 0:
            self._store_maxwell_state()

        # Store P_total BEFORE p2g (uses same F^n as stored in particle state)
        # This is critical for correct gradient computation in backward pass
        self._store_P_total()

        # Run forward step
        self.solver.fields.clear_grid()
        self.solver.fields.clear_particle_energy_increments()
        self.solver.fields.clear_global_energy_step()

        self.solver.p2g()

        # Store grid state after P2G (for backward)
        self._store_grid_state()

        self.solver.grid_op()

        # Store grid_v AFTER grid_ops (for G2P backward)
        # This is the velocity after normalization, gravity, and BC
        self._store_grid_v_after_BC()

        self.solver.g2p()

        # Store C AFTER G2P (C^{n+1} for F update backward)
        self._store_C_next()

        self.solver.update_F_and_internal()

        # Store b_bar_e_trial AFTER update_F_and_internal (for Maxwell backward)
        if self.maxwell_needs_grad and self.n_maxwell > 0:
            self._store_maxwell_trial_state()

        self.solver.cleanup_ut()

    def run_forward_with_storage(self, num_steps: int):
        """Run forward simulation with state storage for BPTT"""
        actual_steps = min(num_steps, self.max_grad_steps)
        self._current_forward_step = 0

        for step in range(actual_steps):
            self.forward_step_with_storage(step)
            self._current_forward_step = step + 1

        return actual_steps

    # ============================================
    # Backward Pass
    # ============================================
    def backward_step(self, step: int):
        """Execute backward pass for one time step

        CORRECTED ORDER for proper gradient propagation:
        Forward: P2G -> GridOps -> G2P -> F_update
        Backward: F_update_bwd -> G2P_bwd -> GridOps_bwd -> P2G_bwd

        Critical fixes:
        1. F_update backward runs FIRST to propagate g_F from step+1
        2. Uses C_next_history (C^{n+1}) instead of restored C^n
        3. g_F is REPLACED (not accumulated) in F_update backward
        4. P2G backward ADDS stress contribution to the propagated g_F
        5. g_C for G2P backward must include F_update contribution
        """
        # Set step in field for kernel access
        self._step_field[None] = step

        # Restore state at this step (x^n, v^n, F^n, C^n)
        self._restore_particle_state()

        # 1. F update backward FIRST (propagate g_F from step+1)
        # Uses C_next_history[step] = C^{n+1} (the C used in F^{n+1} = (I + dt*C^{n+1}) @ F^n)
        # This REPLACES g_F with the propagated value
        # Also ADDS to g_C the contribution from F update
        self._update_F_backward_with_C_next()

        # Save g_C AFTER F update backward (now contains FULL g_C^{n+1}:
        # - APIC contribution from P2G at step n+1
        # - F update contribution from this step)
        self._save_g_C_for_g2p()

        # 2. G2P backward (from g_x, g_v, g_C_saved to g_grid_v)
        # Clear grid gradients for this step
        self.adj_fields.clear_grid_grads()

        # Load stored grid_v for this step (grid_v after BC from forward)
        self._load_grid_v_for_g2p_backward()

        g2p_backward_kernel(
            g_x_next=self.adj_fields.g_x,
            g_v_next=self.adj_fields.g_v,
            g_C_next=self.adj_fields.g_C_saved,  # Use FULL g_C^{n+1} (APIC + F_update)
            x=self.solver.fields.x,
            grid_v=self.solver.fields.grid_v,
            g_grid_v=self.adj_fields.g_grid_v,
            g_x_curr=self.adj_fields.g_x,
            inv_dx=self.inv_dx,
            dx=self.dx,
            dt=self.dt,
            n_particles=self.n_particles,
            grid_size=ti.Vector(self.grid_size)
        )

        # 3. Grid ops backward (from g_grid_v to g_grid_P, g_grid_M)
        self._load_grid_state_for_backward()

        grid_ops_backward_kernel(
            g_grid_v_after=self.adj_fields.g_grid_v,
            grid_v_normalized_history=self.adj_fields.grid_v_normalized_history,
            grid_M_history=self.adj_fields.grid_M_history,
            g_grid_P=self.adj_fields.g_grid_P,
            g_grid_M=self.adj_fields.g_grid_M,
            step=step,
            grid_size=ti.Vector(self.grid_size),
            eps=self.eps
        )

        # 4. P2G backward (from g_grid_P, g_grid_M to g_x, g_v, g_F, g_C)
        # This ADDS stress contribution to g_F (which already has propagated value)
        p2g_backward_kernel(
            g_grid_P=self.adj_fields.g_grid_P,
            g_grid_M=self.adj_fields.g_grid_M,
            x=self.solver.fields.x,
            v=self.solver.fields.v,
            F=self.solver.fields.F,
            C=self.solver.fields.C,
            mass=self.solver.fields.mass,
            volume=self.solver.fields.volume,
            g_x=self.adj_fields.g_x,
            g_v=self.adj_fields.g_v,
            g_F=self.adj_fields.g_F,
            g_C=self.adj_fields.g_C,
            g_m=self.adj_fields.g_m,
            ogden_mu=self.solver.ogden_mu,
            ogden_alpha=self.solver.ogden_alpha,
            n_ogden=self.solver.n_ogden,
            ogden_kappa=self.solver.ogden_kappa,
            g_ogden_mu=self.adj_fields.g_ogden_mu,
            g_ogden_alpha=self.adj_fields.g_ogden_alpha,
            g_ogden_kappa=self.adj_fields.g_ogden_kappa,
            inv_dx=self.inv_dx,
            dx=self.dx,
            dt=self.dt,
            n_particles=self.n_particles,
            grid_size=ti.Vector(self.grid_size),
            P_history=self.adj_fields.P_history,
            step=step
        )

        # 5. Maxwell backward (if enabled)
        if self.maxwell_needs_grad and self.n_maxwell > 0:
            # 5a. Maxwell G gradient (from stress path)
            # Computes g_G[k] = <g_P, ∂P_maxwell/∂G_k> using g_grid_P
            maxwell_G_gradient_kernel(
                F=self.solver.fields.F,
                volume=self.solver.fields.volume,
                b_bar_e=self.solver.fields.b_bar_e,
                g_grid_P=self.adj_fields.g_grid_P,
                x=self.solver.fields.x,
                inv_dx=self.inv_dx,
                dx=self.dx,
                n_particles=self.n_particles,
                n_maxwell=self.n_maxwell,
                grid_size=ti.Vector(self.grid_size),
                g_maxwell_G=self.adj_fields.g_maxwell_G
            )

            # 5b. Maxwell internal variable backward (tau gradients from exponential update)
            self._maxwell_backward_step()

        # 6. Bulk viscosity backward (if enabled)
        if self.config.material.enable_bulk_viscosity:
            bulk_viscosity_gradient_kernel(
                F=self.solver.fields.F,
                C=self.solver.fields.C,
                volume=self.solver.fields.volume,
                g_grid_P=self.adj_fields.g_grid_P,
                x=self.solver.fields.x,
                inv_dx=self.inv_dx,
                dx=self.dx,
                n_particles=self.n_particles,
                grid_size=ti.Vector(self.grid_size),
                g_eta_bulk=self.adj_fields.g_eta_bulk
            )

    @ti.kernel
    def _update_F_backward_with_C_next(self):
        """F update backward using stored C_next_history

        Uses C^{n+1} from C_next_history[step] instead of restored C^n.
        REPLACES g_F (not accumulates) to properly propagate gradient.
        """
        step = self._step_field[None]
        for p in range(self.n_particles):
            g_F_np1 = self.adj_fields.g_F[p]
            F_n = self.solver.fields.F[p]  # Restored F^n
            C_np1 = self.adj_fields.C_next_history[step, p]  # C^{n+1} from history

            # STE: g_F_raw = g_F_new
            g_F_raw = g_F_np1

            # g_C += dt * g_F_raw @ F_old^T
            self.adj_fields.g_C[p] += self.dt * g_F_raw @ F_n.transpose()

            # g_F_old = (I + dt * C)^T @ g_F_raw  [REPLACE]
            I_plus_dtC = ti.Matrix.identity(ti.f32, 3) + self.dt * C_np1
            self.adj_fields.g_F[p] = I_plus_dtC.transpose() @ g_F_raw

    @ti.kernel
    def _maxwell_backward_step(self):
        """Maxwell internal variable backward for a single time step

        Uses stored b_bar_e_history[step] (b_bar_e_old) and b_bar_e_trial_history[step].
        Accumulates gradients into g_b_bar_e and g_maxwell_tau.

        Forward:
            a = exp(-dt / tau)
            b_bar_e^{n+1} = a * b_bar_e^n + (1-a) * b_bar_e_trial

        Backward:
            g_b_bar_e^n += a * g_b_bar_e^{n+1}
            g_tau += <g_b_bar_e^{n+1}, ∂b/∂tau>
        """
        step = self._step_field[None]
        if ti.static(self.maxwell_needs_grad and self.n_maxwell > 0):
            for p in range(self.n_particles):
                for k in ti.static(range(self.n_maxwell)):
                    tau_k = self.solver.maxwell_tau[k]
                    a = ti.exp(-self.dt / tau_k)

                    # Current gradient (from next step or loss)
                    g_b_new = self.adj_fields.g_b_bar_e[p, k]

                    # Get stored states
                    b_old = self.adj_fields.b_bar_e_history[step, p, k]
                    b_trial = self.adj_fields.b_bar_e_trial_history[step, p, k]

                    # Propagate gradient to b_bar_e_old: g_b^n += a * g_b^{n+1}
                    self.adj_fields.g_b_bar_e[p, k] += a * g_b_new

                    # Gradient w.r.t. tau: ∂a/∂tau = dt/tau² * a
                    da_dtau = self.dt / (tau_k * tau_k) * a
                    # ∂b^{n+1}/∂tau = da/dtau * (b_old - b_trial)
                    db_dtau = da_dtau * (b_old - b_trial)

                    # g_tau += Frobenius inner product <g_b_new, db_dtau>
                    g_tau_contrib = 0.0
                    for i in ti.static(range(3)):
                        for j in ti.static(range(3)):
                            g_tau_contrib += g_b_new[i, j] * db_dtau[i, j]
                    self.adj_fields.g_maxwell_tau[k] += g_tau_contrib

    @ti.kernel
    def _load_grid_state_for_backward(self):
        """Load stored grid state for backward computation

        Note: This is intentionally a no-op because grid_ops_backward_kernel
        accesses grid_v_normalized_history and grid_M_history directly
        using the step parameter. No intermediate loading is needed.
        """
        pass  # Grid state is accessed directly from history fields

    def run_backward(self, num_steps: int):
        """Run backward pass through all stored steps"""
        for step in range(num_steps - 1, -1, -1):
            self.backward_step(step)

    # ============================================
    # Loss Computation
    # ============================================
    def compute_loss_and_grad(self, loss_type: str):
        """Compute loss and initialize particle gradients from loss"""
        self.loss_field[None] = 0.0

        if loss_type == 'position':
            if self._target_x is None:
                raise TargetNotSetError("Target positions not set. Call set_target_positions() first.")
            position_loss_backward_kernel(
                x=self.solver.fields.x,
                target_x=self._target_x,
                g_x=self.adj_fields.g_x,
                loss=self.loss_field,
                n_particles=self.n_particles
            )
        elif loss_type == 'velocity':
            if self._target_v is None:
                raise TargetNotSetError("Target velocities not set. Call set_target_velocities() first.")
            velocity_loss_backward_kernel(
                v=self.solver.fields.v,
                target_v=self._target_v,
                g_v=self.adj_fields.g_v,
                loss=self.loss_field,
                n_particles=self.n_particles
            )
        elif loss_type == 'kinetic_energy':
            kinetic_energy_loss_backward_kernel(
                v=self.solver.fields.v,
                mass=self.solver.fields.mass,
                g_v=self.adj_fields.g_v,
                loss=self.loss_field,
                target_energy=self._target_energy[None],
                n_particles=self.n_particles
            )
        elif loss_type == 'total_energy':
            total_energy_loss_backward_kernel(
                x=self.solver.fields.x,
                v=self.solver.fields.v,
                F=self.solver.fields.F,
                mass=self.solver.fields.mass,
                volume=self.solver.fields.volume,
                ogden_mu=self.solver.ogden_mu,
                ogden_alpha=self.solver.ogden_alpha,
                n_ogden=self.solver.n_ogden,
                ogden_kappa=self.solver.ogden_kappa,
                g_v=self.adj_fields.g_v,
                g_F=self.adj_fields.g_F,
                loss=self.loss_field,
                target_energy=self._target_total_energy[None],
                n_particles=self.n_particles
            )
        else:
            raise GradientError(f"Unknown loss type: {loss_type}. Valid: position, velocity, kinetic_energy, total_energy")

        return self.loss_field[None]

    # ============================================
    # Main API
    # ============================================
    def solve_with_gradients(
        self,
        num_steps: int,
        loss_type: str = 'position',
        requires_grad: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Union[float, NDArray[np.float32]]]:
        """
        Run forward simulation and compute gradients via manual adjoint

        Args:
            num_steps: Number of simulation steps
            loss_type: Type of loss function
                - 'position': L2 distance to target positions
                - 'velocity': L2 distance to target velocities
                - 'kinetic_energy': L2 distance to target kinetic energy (0.5 * m * v^2)
                - 'total_energy': L2 distance to target total energy (kinetic + elastic)
                  Note: Does not include viscous energy dissipation.
            requires_grad: Dict specifying which gradients to compute
                - 'ogden_mu': Gradient w.r.t. Ogden shear moduli
                - 'ogden_alpha': Gradient w.r.t. Ogden exponents
                - 'maxwell_G': Gradient w.r.t. Maxwell shear moduli
                - 'maxwell_tau': Gradient w.r.t. Maxwell relaxation times
                - 'eta_bulk': Gradient w.r.t. bulk viscosity coefficient
                - 'initial_x': Gradient w.r.t. initial positions
                - 'initial_v': Gradient w.r.t. initial velocities

        Returns:
            Dict containing:
                - 'loss': Scalar loss value
                - 'grad_*': Requested gradients as numpy arrays
        """
        if requires_grad is None:
            requires_grad = {}

        # Check scale guards in experimental P_total mode
        if is_experimental_mode_enabled():
            max_particles, max_steps = get_scale_guards()
            if self.n_particles > max_particles:
                raise ScaleGuardError(
                    f"Experimental P_total mode: particle count ({self.n_particles}) exceeds "
                    f"scale guard ({max_particles}). Reduce particles or increase max_particles "
                    f"in configure_gradient_mode()."
                )
            if num_steps > max_steps:
                raise ScaleGuardError(
                    f"Experimental P_total mode: step count ({num_steps}) exceeds "
                    f"scale guard ({max_steps}). Reduce steps or increase max_steps "
                    f"in configure_gradient_mode()."
                )

        # Clear all gradients
        self.adj_fields.clear_all_grads()

        # Run forward with state storage
        actual_steps = self.run_forward_with_storage(num_steps)

        # Compute loss and initialize gradients
        loss = self.compute_loss_and_grad(loss_type)

        # Run backward pass
        self.run_backward(actual_steps)

        # Collect results
        results = {'loss': loss}

        if requires_grad.get('ogden_mu', False):
            results['grad_ogden_mu'] = self.adj_fields.g_ogden_mu.to_numpy()[:self.solver.n_ogden]

        if requires_grad.get('ogden_alpha', False):
            results['grad_ogden_alpha'] = self.adj_fields.g_ogden_alpha.to_numpy()[:self.solver.n_ogden]

        if requires_grad.get('ogden_kappa', False):
            results['grad_ogden_kappa'] = self.adj_fields.g_ogden_kappa[None]

        if requires_grad.get('maxwell_G', False) and self.n_maxwell > 0:
            results['grad_maxwell_G'] = self.adj_fields.g_maxwell_G.to_numpy()[:self.n_maxwell]

        if requires_grad.get('maxwell_tau', False) and self.n_maxwell > 0:
            results['grad_maxwell_tau'] = self.adj_fields.g_maxwell_tau.to_numpy()[:self.n_maxwell]

        if requires_grad.get('eta_bulk', False) and self.config.material.enable_bulk_viscosity:
            results['grad_eta_bulk'] = self.adj_fields.g_eta_bulk[None]

        if requires_grad.get('initial_x', False):
            results['grad_initial_x'] = self.adj_fields.g_x.to_numpy()

        if requires_grad.get('initial_v', False):
            results['grad_initial_v'] = self.adj_fields.g_v.to_numpy()

        if requires_grad.get('F', False):
            results['grad_F'] = self.adj_fields.g_F.to_numpy()

        return results

    def verify_gradient_numerical(self, param_name: str, param_idx: int = 0,
                                   num_steps: int = 1, loss_type: str = 'position',
                                   eps: float = 1e-4) -> Dict[str, float]:
        """
        Verify gradient using finite difference

        Args:
            param_name: Parameter to verify ('ogden_mu', 'ogden_alpha', 'initial_x', 'initial_v', etc.)
            param_idx: Index of parameter (for array parameters, or particle*3+dim for positions)
            num_steps: Number of simulation steps
            loss_type: Loss type
            eps: Finite difference epsilon

        Returns:
            Dict with 'analytic', 'numerical', 'rel_error', 'cos_sim'
        """
        # Get analytic gradient
        requires_grad = {param_name: True}
        result = self.solve_with_gradients(num_steps, loss_type, requires_grad)
        grad_key = f'grad_{param_name}'
        grad_array = result.get(grad_key, None)
        if grad_array is None:
            raise GradientError(f"No gradient found for {param_name}")

        # Get the initial positions/velocities for perturbation
        init_x = self.adj_fields.x_history.to_numpy()[0].copy()
        init_v = self.adj_fields.v_history.to_numpy()[0].copy()

        # Handle different parameter types
        if param_name in ['initial_x', 'initial_v']:
            # param_idx encodes particle*3 + dim
            particle_idx = param_idx // 3
            dim_idx = param_idx % 3
            grad_analytic = grad_array[particle_idx, dim_idx]

            # Perturb initial position or velocity
            if param_name == 'initial_x':
                init_perturbed = init_x.copy()
            else:
                init_perturbed = init_v.copy()

            # Forward difference: f(x+eps)
            perturbed_plus = init_perturbed.copy()
            perturbed_plus[particle_idx, dim_idx] += eps
            if param_name == 'initial_x':
                self.solver.initialize_particles(perturbed_plus, init_v)
            else:
                self.solver.initialize_particles(init_x, perturbed_plus)
            self.run_forward_with_storage(num_steps)
            loss_plus = self.compute_loss_and_grad(loss_type)

            # Backward difference: f(x-eps)
            perturbed_minus = init_perturbed.copy()
            perturbed_minus[particle_idx, dim_idx] -= eps
            if param_name == 'initial_x':
                self.solver.initialize_particles(perturbed_minus, init_v)
            else:
                self.solver.initialize_particles(init_x, perturbed_minus)
            self.run_forward_with_storage(num_steps)
            loss_minus = self.compute_loss_and_grad(loss_type)

            # Restore original
            self.solver.initialize_particles(init_x, init_v)

        else:
            # Material parameters
            grad_analytic = grad_array[param_idx]

            # Get parameter field
            if param_name == 'ogden_mu':
                param_field = self.solver.ogden_mu
            elif param_name == 'ogden_alpha':
                param_field = self.solver.ogden_alpha
            elif param_name == 'maxwell_G':
                param_field = self.solver.maxwell_G
            elif param_name == 'maxwell_tau':
                param_field = self.solver.maxwell_tau
            else:
                raise GradientError(f"Unknown parameter: {param_name}")

            # Save original value
            original_value = param_field[param_idx]

            # Forward difference: f(x+eps)
            param_field[param_idx] = original_value + eps
            self.solver.initialize_particles(init_x, init_v)
            self.run_forward_with_storage(num_steps)
            loss_plus = self.compute_loss_and_grad(loss_type)

            # Backward difference: f(x-eps)
            param_field[param_idx] = original_value - eps
            self.solver.initialize_particles(init_x, init_v)
            self.run_forward_with_storage(num_steps)
            loss_minus = self.compute_loss_and_grad(loss_type)

            # Restore original value
            param_field[param_idx] = original_value

        loss_center = result['loss']

        # Numerical gradient
        grad_numerical = (loss_plus - loss_minus) / (2 * eps)

        # Compute metrics
        rel_error = abs(grad_analytic - grad_numerical) / max(1e-10, abs(grad_numerical))
        cos_sim = 1.0 if (grad_analytic * grad_numerical > 0) else -1.0

        return {
            'analytic': grad_analytic,
            'numerical': grad_numerical,
            'rel_error': rel_error,
            'cos_sim': cos_sim,
            'loss_center': loss_center,
            'loss_plus': loss_plus,
            'loss_minus': loss_minus
        }

    # ============================================
    # Utility Methods
    # ============================================
    def get_particle_data(self) -> Dict[str, NDArray[np.float32]]:
        """Get current particle data"""
        return self.solver.get_particle_data()

    def get_energy_data(self) -> Dict[str, float]:
        """Get current energy data"""
        return self.solver.get_energy_data()

    def get_spd_statistics(self) -> Dict[str, int]:
        """Get SPD projection statistics"""
        return {
            'trigger_count': self.adj_fields.spd_trigger_count[None],
            'total_count': self.adj_fields.spd_total_count[None]
        }

    def reset(self) -> None:
        """Reset solver state"""
        self.adj_fields.clear_all_grads()
        self.adj_fields.clear_spd_stats()
        self._current_forward_step = 0
