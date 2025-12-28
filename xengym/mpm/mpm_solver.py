"""
MPM Solver Main Flow
Implements the complete MLS-MPM/APIC solver with VHE constitutive model
Supports automatic differentiation via enable_grad parameter
"""
from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING

import taichi as ti
import numpy as np
from numpy.typing import NDArray

from .config import MPMConfig
from .fields import MPMFields
from .constitutive import compute_ogden_stress_general, compute_ogden_stress_2terms, compute_maxwell_stress, compute_bulk_viscosity_stress
from .contact import compute_contact_force, update_contact_age, sdf_plane, evaluate_sdf, compute_sdf_normal
from .decomp import make_spd, make_spd_ste
from .exceptions import ConfigurationError, MaterialError


@ti.data_oriented
class MPMSolver:
    """
    3D Explicit MLS-MPM/APIC Solver with VHE constitutive model
    Supports automatic differentiation when enable_grad=True
    """

    def __init__(self, config: MPMConfig, n_particles: int, enable_grad: bool = False, use_spd_ste: bool = True):
        """
        Initialize MPM solver

        Args:
            config: MPM configuration
            n_particles: Number of particles
            enable_grad: If True, enable automatic differentiation support
            use_spd_ste: If True, use Straight-Through Estimator for SPD projection in AD mode
        """
        self.config = config
        self.n_particles = n_particles
        self.enable_grad = enable_grad
        self.use_spd_ste = use_spd_ste
        self.fields = MPMFields(config, n_particles, enable_grad=enable_grad)
        self.current_step = 0

        # Loss field for autodiff (always create, only needs_grad when enable_grad=True)
        self.loss_field = ti.field(dtype=ti.f32, shape=(), needs_grad=enable_grad)

        # Material parameters (convert to Taichi fields for kernel access)
        # Support up to 4 Ogden terms
        self.n_ogden = min(len(config.material.ogden.mu), 4)
        if self.n_ogden == 0:
            raise MaterialError("At least one Ogden term is required")
        if len(config.material.ogden.mu) != len(config.material.ogden.alpha):
            raise MaterialError("Ogden mu and alpha must have the same length")

        # Ogden parameters with optional gradient support
        self.ogden_mu = ti.field(dtype=ti.f32, shape=4, needs_grad=enable_grad)
        self.ogden_alpha = ti.field(dtype=ti.f32, shape=4, needs_grad=enable_grad)
        # Fill with actual values
        for i in range(self.n_ogden):
            self.ogden_mu[i] = config.material.ogden.mu[i]
            self.ogden_alpha[i] = config.material.ogden.alpha[i]
        # Fill remaining with zeros (won't be used)
        for i in range(self.n_ogden, 4):
            self.ogden_mu[i] = 0.0
            self.ogden_alpha[i] = 1.0

        self.ogden_kappa = config.material.ogden.kappa

        # Maxwell parameters with optional gradient support
        self.n_maxwell = len(config.material.maxwell_branches)
        if self.n_maxwell > 0:
            self.maxwell_G = ti.field(dtype=ti.f32, shape=self.n_maxwell, needs_grad=enable_grad)
            self.maxwell_tau = ti.field(dtype=ti.f32, shape=self.n_maxwell, needs_grad=enable_grad)
            G_list = [b.G for b in config.material.maxwell_branches]
            tau_list = [b.tau for b in config.material.maxwell_branches]
            self.maxwell_G.from_numpy(np.array(G_list, dtype=np.float32))
            self.maxwell_tau.from_numpy(np.array(tau_list, dtype=np.float32))

        # Contact parameters
        self.enable_contact = config.contact.enable_contact
        self.contact_stiffness_normal = config.contact.contact_stiffness_normal
        self.contact_stiffness_tangent = config.contact.contact_stiffness_tangent
        self.mu_s = config.contact.mu_s
        self.mu_k = config.contact.mu_k
        self.v_transition = config.contact.friction_transition_vel
        self.K_clear = config.contact.K_clear

        # SDF obstacles (default: ground plane at z=0 for backward compatibility)
        obstacles = config.contact.obstacles
        if len(obstacles) == 0:
            # Default ground plane
            self.n_obstacles = 1
            self.obstacle_types = ti.field(dtype=ti.i32, shape=1)
            self.obstacle_centers = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.obstacle_normals = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.obstacle_half_extents = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.obstacle_types[0] = 0  # plane
            self.obstacle_centers[0] = ti.Vector([0.0, 0.0, 0.0])
            self.obstacle_normals[0] = ti.Vector([0.0, 0.0, 1.0])
            self.obstacle_half_extents[0] = ti.Vector([0.0, 0.0, 0.0])
        else:
            self.n_obstacles = len(obstacles)
            self.obstacle_types = ti.field(dtype=ti.i32, shape=self.n_obstacles)
            self.obstacle_centers = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
            self.obstacle_normals = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
            self.obstacle_half_extents = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
            type_map = {'plane': 0, 'sphere': 1, 'box': 2, 'cylinder': 3}
            for i, obs in enumerate(obstacles):
                self.obstacle_types[i] = type_map.get(obs.sdf_type, 0)
                self.obstacle_centers[i] = ti.Vector(list(obs.center))
                self.obstacle_normals[i] = ti.Vector(list(obs.normal))
                self.obstacle_half_extents[i] = ti.Vector(list(obs.half_extents))

        # Obstacle kinematics (for moving obstacles / friction in relative frame)
        self.obstacle_centers_prev = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
        self.obstacle_velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.n_obstacles)
        for i in range(self.n_obstacles):
            self.obstacle_centers_prev[i] = self.obstacle_centers[i]
            self.obstacle_velocities[i] = ti.Vector([0.0, 0.0, 0.0])

        # Time stepping
        self.dt = config.time.dt
        self.dx = config.grid.dx
        self.inv_dx = 1.0 / self.dx

        # Gravity
        self.gravity = ti.Vector([0.0, 0.0, -9.81])

    def initialize_particles(
        self,
        positions: NDArray[np.float32],
        velocities: Optional[NDArray[np.float32]] = None,
        volumes: Optional[NDArray[np.float32]] = None
    ) -> None:
        """Initialize particle data"""
        self.fields.initialize_particles(positions, velocities, volumes)

    @ti.kernel
    def p2g(self):
        """Particle to grid transfer (P2G)"""
        for p in range(self.n_particles):
            # Particle state
            x_p = self.fields.x[p]
            v_p = self.fields.v[p]
            F_p = self.fields.F[p]
            C_p = self.fields.C[p]
            m_p = self.fields.mass[p]
            V_p = self.fields.volume[p]

            # Compute stress (using general Ogden model)
            P_elastic, psi_elastic = compute_ogden_stress_general(
                F_p,
                self.ogden_mu,
                self.ogden_alpha,
                self.n_ogden,
                self.ogden_kappa
            )

            # Add Maxwell stress if enabled
            P_total = P_elastic
            if ti.static(self.n_maxwell > 0):
                # Compute Maxwell stress from internal variables
                J = F_p.determinant()
                tau_maxwell_total = ti.Matrix.zero(ti.f32, 3, 3)

                for k in ti.static(range(self.n_maxwell)):
                    b_bar_e_k = self.fields.b_bar_e[p, k]
                    G_k = self.maxwell_G[k]

                    # Cauchy stress: tau = G * dev(b_bar_e)
                    trace_b = b_bar_e_k[0,0] + b_bar_e_k[1,1] + b_bar_e_k[2,2]
                    tau_k = G_k * (b_bar_e_k - trace_b / 3.0 * ti.Matrix.identity(ti.f32, 3))
                    tau_maxwell_total += tau_k

                # Convert Cauchy stress to 1st PK stress: P = J * tau * F^-T
                P_maxwell = J * tau_maxwell_total @ F_p.inverse().transpose()
                P_total = P_elastic + P_maxwell

            # Add bulk viscosity stress if enabled
            if ti.static(self.config.material.enable_bulk_viscosity):
                # Approximate velocity gradient from C
                L = self.fields.C[p]
                trace_L = L[0,0] + L[1,1] + L[2,2]

                # Bulk viscosity stress (Cauchy): sigma_visc = eta_bulk * tr(D) * I
                # where D = (L + L^T) / 2 is the rate of deformation
                eta_bulk = self.config.material.bulk_viscosity
                sigma_visc = eta_bulk * trace_L * ti.Matrix.identity(ti.f32, 3)

                # Convert to 1st PK stress
                J = F_p.determinant()
                P_visc = J * sigma_visc @ F_p.inverse().transpose()
                P_total += P_visc

            # Affine momentum
            affine = P_total @ F_p.transpose() * V_p + m_p * C_p

            # Grid node base index
            base = ti.cast(x_p * self.inv_dx - 0.5, ti.i32)

            # Quadratic B-spline weights
            fx = x_p * self.inv_dx - ti.cast(base, ti.f32)

            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

            # Scatter to grid
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]

                grid_idx = base + offset

                # Check bounds
                if 0 <= grid_idx[0] < self.fields.grid_size[0] and \
                   0 <= grid_idx[1] < self.fields.grid_size[1] and \
                   0 <= grid_idx[2] < self.fields.grid_size[2]:

                    # Mass
                    self.fields.grid_m[grid_idx] += weight * m_p

                    # Momentum (with affine contribution)
                    self.fields.grid_v[grid_idx] += weight * (m_p * v_p + affine @ dpos)

    @ti.kernel
    def grid_op(self):
        """Grid operations: apply forces, boundary conditions, and contact"""
        for I in ti.grouped(self.fields.grid_m):
            if self.fields.grid_m[I] > 1e-10:
                # Normalize momentum to get velocity
                self.fields.grid_v[I] /= self.fields.grid_m[I]

                # Apply gravity
                self.fields.grid_v[I] += self.dt * self.gravity

                # Boundary conditions (sticky walls)
                for d in ti.static(range(3)):
                    if I[d] < 3 or I[d] >= self.fields.grid_size[d] - 3:
                        self.fields.grid_v[I][d] = 0.0

                # Contact with SDF obstacles (configurable: plane/sphere/box/cylinder)
                if ti.static(self.enable_contact):
                    grid_x = ti.cast(I, ti.f32) * self.dx
                    any_contact = 0

                    # Iterate over all obstacles
                    for obs_idx in range(self.n_obstacles):
                        obs_type = self.obstacle_types[obs_idx]
                        obs_center = self.obstacle_centers[obs_idx]       
                        obs_normal = self.obstacle_normals[obs_idx]       
                        obs_half_ext = self.obstacle_half_extents[obs_idx] 
                        obs_vel = self.obstacle_velocities[obs_idx]

                        # Evaluate SDF for this obstacle
                        phi = evaluate_sdf(grid_x, obs_type, obs_center, obs_normal, obs_half_ext)

                        if phi < 0.0:
                            # In contact with this obstacle
                            normal = compute_sdf_normal(grid_x, obs_type, obs_center, obs_normal, obs_half_ext)
                            v_rel = self.fields.grid_v[I] - obs_vel

                            # Compute contact force with friction
                            f_contact, u_t_new, is_contact = compute_contact_force(
                                phi, v_rel, normal,
                                self.fields.grid_ut[I],
                                self.dt,
                                self.contact_stiffness_normal,
                                self.contact_stiffness_tangent,
                                self.mu_s, self.mu_k,
                                self.v_transition
                            )

                            # Apply contact force (impulse)
                            if self.fields.grid_m[I] > 1e-10:
                                self.fields.grid_v[I] += self.dt * f_contact / self.fields.grid_m[I]

                            # Update tangential displacement
                            self.fields.grid_ut[I] = u_t_new

                            # Mark as in contact
                            any_contact = 1

                    self.fields.grid_contact_mask[I] = any_contact

    @ti.kernel
    def g2p(self):
        """Grid to particle transfer (G2P)"""
        for p in range(self.n_particles):
            x_p = self.fields.x[p]

            # Grid node base index
            base = ti.cast(x_p * self.inv_dx - 0.5, ti.i32)
            fx = x_p * self.inv_dx - ti.cast(base, ti.f32)

            # Quadratic B-spline weights
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

            # Gather from grid
            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]

                grid_idx = base + offset

                if 0 <= grid_idx[0] < self.fields.grid_size[0] and \
                   0 <= grid_idx[1] < self.fields.grid_size[1] and \
                   0 <= grid_idx[2] < self.fields.grid_size[2]:

                    grid_v = self.fields.grid_v[grid_idx]
                    new_v += weight * grid_v
                    new_C += 4.0 * self.inv_dx * weight * grid_v.outer_product(dpos)

            # Update particle velocity and position
            self.fields.v[p] = new_v
            self.fields.x[p] += self.dt * new_v

            # Update APIC matrix
            self.fields.C[p] = new_C

    @ti.kernel
    def update_F_and_internal(self):
        """Update deformation gradient and internal variables"""
        for p in range(self.n_particles):
            # Save old F for viscous dissipation calculation
            F_old = self.fields.F[p]

            # Update deformation gradient: F_new = (I + dt * C) @ F_old
            self.fields.F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt * self.fields.C[p]) @ F_old

            # Clamp F to avoid extreme deformation
            J = self.fields.F[p].determinant()
            if J < 0.5:
                self.fields.F[p] *= ti.pow(0.5 / J, 1.0/3.0)
            elif J > 2.0:
                self.fields.F[p] *= ti.pow(2.0 / J, 1.0/3.0)

            # Update Maxwell internal variables and compute energy corrections
            if ti.static(self.n_maxwell > 0):
                F_new = self.fields.F[p]
                J_new = F_new.determinant()
                F_bar = ti.pow(J_new, -1.0/3.0) * F_new

                delta_E_proj_p = 0.0
                delta_E_visc_maxwell = 0.0

                for k in ti.static(range(self.n_maxwell)):
                    # Get old internal variable
                    b_bar_e_old = self.fields.b_bar_e[p, k]

                    # Upper-convected update
                    b_bar_e_trial = F_bar @ b_bar_e_old @ F_bar.transpose()

                    # Relaxation
                    G_k = self.maxwell_G[k]
                    tau_k = self.maxwell_tau[k]
                    relax_factor = ti.exp(-self.dt / tau_k)
                    b_bar_e_relaxed = relax_factor * b_bar_e_trial + (1.0 - relax_factor) * ti.Matrix.identity(ti.f32, 3)

                    # Compute Maxwell viscous dissipation (before projection)
                    # Dissipation = G_k / tau_k * ||b_bar_e_trial - I||^2 * dt / 2
                    diff_relax = b_bar_e_trial - ti.Matrix.identity(ti.f32, 3)
                    delta_E_visc_k = 0.5 * G_k / tau_k * (
                        diff_relax[0,0]**2 + diff_relax[0,1]**2 + diff_relax[0,2]**2 +
                        diff_relax[1,0]**2 + diff_relax[1,1]**2 + diff_relax[1,2]**2 +
                        diff_relax[2,0]**2 + diff_relax[2,1]**2 + diff_relax[2,2]**2
                    ) * self.dt
                    delta_E_visc_maxwell += delta_E_visc_k

                    # SPD projection - use STE in autodiff mode if enabled
                    if ti.static(self.enable_grad and self.use_spd_ste):
                        b_bar_e_new = make_spd_ste(b_bar_e_relaxed, 1e-8)
                    else:
                        b_bar_e_new = make_spd(b_bar_e_relaxed, 1e-8)

                    # Enforce isochoric constraint
                    det_b = b_bar_e_new.determinant()
                    if det_b > 1e-10:
                        scale = ti.pow(det_b, -1.0/3.0)
                        b_bar_e_new = scale * b_bar_e_new

                    # Compute projection energy correction
                    diff_proj = b_bar_e_new - b_bar_e_relaxed
                    delta_E_proj_k = 0.5 * G_k * (
                        diff_proj[0,0]**2 + diff_proj[0,1]**2 + diff_proj[0,2]**2 +
                        diff_proj[1,0]**2 + diff_proj[1,1]**2 + diff_proj[1,2]**2 +
                        diff_proj[2,0]**2 + diff_proj[2,1]**2 + diff_proj[2,2]**2
                    )
                    delta_E_proj_p += delta_E_proj_k

                    # Update internal variable
                    self.fields.b_bar_e[p, k] = b_bar_e_new

                # Store energy corrections
                self.fields.delta_E_proj_step[p] = delta_E_proj_p
                self.fields.delta_E_viscous_step[p] += delta_E_visc_maxwell

            # Compute bulk viscosity dissipation if enabled
            if ti.static(self.config.material.enable_bulk_viscosity):
                J_old = F_old.determinant()
                J_new = self.fields.F[p].determinant()
                J_dot = (J_new - J_old) / self.dt
                vol_strain_rate = J_dot / J_new
                eta_bulk = self.config.material.bulk_viscosity
                delta_E_visc_bulk = eta_bulk * vol_strain_rate ** 2 * self.dt
                # Accumulate (not overwrite) bulk viscosity dissipation
                self.fields.delta_E_viscous_step[p] += delta_E_visc_bulk

    @ti.kernel
    def clear_energy_fields(self):
        """Clear global energy accumulators (separate kernel for autodiff compatibility)"""
        self.fields.E_kin[None] = 0.0
        self.fields.E_elastic[None] = 0.0
        self.fields.E_viscous_step[None] = 0.0
        self.fields.E_proj_step[None] = 0.0

    @ti.kernel
    def reduce_energies(self):
        """Reduce particle-level energies to global scalars"""
        # Accumulate from particles
        for p in range(self.n_particles):
            # Kinetic energy
            v_p = self.fields.v[p]
            m_p = self.fields.mass[p]
            self.fields.E_kin[None] += 0.5 * m_p * v_p.dot(v_p)

            # Elastic energy
            F_p = self.fields.F[p]
            V_p = self.fields.volume[p]
            _, psi = compute_ogden_stress_general(
                F_p,
                self.ogden_mu,
                self.ogden_alpha,
                self.n_ogden,
                self.ogden_kappa
            )
            self.fields.E_elastic[None] += psi * V_p

            # Viscous and projection energies
            self.fields.E_viscous_step[None] += self.fields.delta_E_viscous_step[p] * V_p
            self.fields.E_proj_step[None] += self.fields.delta_E_proj_step[p] * V_p

        # Update cumulative energies
        self.fields.E_viscous_cum[None] += self.fields.E_viscous_step[None]
        self.fields.E_proj_cum[None] += self.fields.E_proj_step[None]

    @ti.kernel
    def cleanup_ut(self):
        """Cleanup tangential displacement based on hysteresis counter"""
        for I in ti.grouped(self.fields.grid_ut):
            age_new, should_clear = update_contact_age(
                self.fields.grid_contact_mask[I],
                self.fields.grid_nocontact_age[I],
                self.K_clear
            )
            self.fields.grid_nocontact_age[I] = age_new

            if should_clear == 1:
                self.fields.grid_ut[I] = ti.Vector([0.0, 0.0, 0.0])       

    @ti.kernel
    def update_obstacle_velocities(self):
        """Update obstacle velocities from center delta (for moving obstacle friction)."""
        for i in range(self.n_obstacles):
            cur = self.obstacle_centers[i]
            prev = self.obstacle_centers_prev[i]
            self.obstacle_velocities[i] = (cur - prev) / self.dt
            self.obstacle_centers_prev[i] = cur

    def step(self) -> None:
        """Execute one simulation step"""
        self.fields.clear_grid()
        self.fields.clear_particle_energy_increments()
        self.fields.clear_global_energy_step()

        self.update_obstacle_velocities()
        self.p2g()
        self.grid_op()
        self.g2p()
        self.update_F_and_internal()  # Now handles both STE/non-STE via ti.static
        self.clear_energy_fields()  # Clear energy accumulators (autodiff-safe)
        self.reduce_energies()  # Accumulate energies from particles
        self.cleanup_ut()

        self.current_step += 1

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run simulation for specified number of steps"""
        if num_steps is None:
            num_steps = self.config.time.num_steps

        for _ in range(num_steps):
            self.step()

    def reset_loss(self) -> None:
        """Reset loss field to zero. Call before starting a new autodiff pass."""
        self.loss_field[None] = 0.0

    def reset_gradients(self) -> None:
        """Reset all gradient fields to zero. Call before starting a new autodiff pass."""
        if not self.enable_grad:
            return

        # Reset loss gradient
        self.loss_field.grad[None] = 0.0

        # Reset material parameter gradients
        for i in range(4):
            self.ogden_mu.grad[i] = 0.0
            self.ogden_alpha.grad[i] = 0.0

        if self.n_maxwell > 0:
            for k in range(self.n_maxwell):
                self.maxwell_G.grad[k] = 0.0
                self.maxwell_tau.grad[k] = 0.0

        # Reset particle field gradients
        self.fields.reset_gradients()

    def get_particle_data(self) -> Dict[str, NDArray[np.float32]]:
        """Get current particle data"""
        return self.fields.get_particle_data()

    def get_energy_data(self) -> Dict[str, float]:
        """Get current energy data"""
        return self.fields.get_energy_data()
