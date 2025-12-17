"""
Autodiff Wrapper for MPM Solver
Provides automatic differentiation interface for material parameters, initial states, and external controls.
Uses pure Taichi kernels for differentiable loss computation - no numpy in gradient chain.
"""
import taichi as ti
import numpy as np
from typing import Dict, Optional, List
from .mpm_solver import MPMSolver
from .config import MPMConfig


@ti.data_oriented
class DifferentiableMPMSolver:
    """
    Wrapper for MPM solver with automatic differentiation support.

    Key features:
    - Creates solver with enable_grad=True for gradient-enabled fields
    - Provides predefined differentiable loss kernels (position/force/energy matching)
    - Uses ti.ad.Tape(loss=loss_field) for correct gradient computation
    - Supports memory control via max_grad_steps parameter
    """

    def __init__(self, config: MPMConfig, n_particles: int,
                 use_spd_ste: bool = True, max_grad_steps: int = 50):
        """
        Initialize differentiable MPM solver

        Args:
            config: MPM configuration
            n_particles: Number of particles
            use_spd_ste: If True, use Straight-Through Estimator for SPD projection
            max_grad_steps: Maximum steps for gradient computation to control memory
        """
        self.config = config
        self.n_particles = n_particles
        self.max_grad_steps = max_grad_steps

        # Create solver with gradient support
        self.solver = MPMSolver(config, n_particles, enable_grad=True, use_spd_ste=use_spd_ste)

        # Target fields for loss computation (created on demand)
        self._target_x = None
        self._target_v = None
        self._target_energy = None

    def initialize_particles(self, positions, velocities=None, volumes=None):
        """Initialize particle data"""
        self.solver.initialize_particles(positions, velocities, volumes)

    def set_target_positions(self, target_positions: np.ndarray):
        """Set target positions for position matching loss"""
        if self._target_x is None:
            self._target_x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self._target_x.from_numpy(target_positions.astype(np.float32))

    def set_target_velocities(self, target_velocities: np.ndarray):
        """Set target velocities for velocity matching loss"""
        if self._target_v is None:
            self._target_v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self._target_v.from_numpy(target_velocities.astype(np.float32))

    def set_target_energy(self, target_energy: float):
        """Set target energy for energy matching loss"""
        if self._target_energy is None:
            self._target_energy = ti.field(dtype=ti.f32, shape=())
        self._target_energy[None] = target_energy

    @ti.kernel
    def _compute_position_loss(self):
        """Compute L2 loss between current and target positions (pure Taichi kernel)"""
        for p in range(self.n_particles):
            diff = self.solver.fields.x[p] - self._target_x[p]
            self.solver.loss_field[None] += diff.dot(diff)

    @ti.kernel
    def _compute_velocity_loss(self):
        """Compute L2 loss between current and target velocities (pure Taichi kernel)"""
        for p in range(self.n_particles):
            diff = self.solver.fields.v[p] - self._target_v[p]
            self.solver.loss_field[None] += diff.dot(diff)

    @ti.kernel
    def _compute_kinetic_energy_loss(self):
        """Compute L2 loss for kinetic energy matching (pure Taichi kernel)"""
        E_kin = 0.0
        for p in range(self.n_particles):
            v_p = self.solver.fields.v[p]
            m_p = self.solver.fields.mass[p]
            E_kin += 0.5 * m_p * v_p.dot(v_p)
        diff = E_kin - self._target_energy[None]
        self.solver.loss_field[None] += diff * diff

    @ti.kernel
    def _compute_com_position_loss(self):
        """Compute loss on center of mass position (pure Taichi kernel)"""
        com = ti.Vector([0.0, 0.0, 0.0])
        total_mass = 0.0
        for p in range(self.n_particles):
            com += self.solver.fields.mass[p] * self.solver.fields.x[p]
            total_mass += self.solver.fields.mass[p]
        com = com / total_mass

        # Target COM from target positions
        target_com = ti.Vector([0.0, 0.0, 0.0])
        for p in range(self.n_particles):
            target_com += self.solver.fields.mass[p] * self._target_x[p]
        target_com = target_com / total_mass

        diff = com - target_com
        self.solver.loss_field[None] += diff.dot(diff)

    def run_forward(self, num_steps: int) -> None:
        """Run forward simulation (without gradient computation)"""
        actual_steps = min(num_steps, self.max_grad_steps) if self.max_grad_steps > 0 else num_steps
        self.solver.run(actual_steps)

    def run_with_gradients(
        self,
        num_steps: int,
        loss_type: str = 'position',
        requires_grad: Optional[Dict[str, bool]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run simulation and compute loss with automatic differentiation.

        Args:
            num_steps: Number of simulation steps
            loss_type: Type of loss to compute ('position', 'velocity', 'energy', 'com')
            requires_grad: Dictionary specifying which parameters need gradients
                          e.g., {'ogden_mu': True, 'initial_x': True}

        Returns:
            Dictionary containing:
                - 'loss': Scalar loss value
                - 'grad_ogden_mu': Gradient w.r.t. Ogden mu parameters (if requested)
                - 'grad_ogden_alpha': Gradient w.r.t. Ogden alpha parameters (if requested)
                - 'grad_initial_x': Gradient w.r.t. initial particle positions (if requested)
                - ... (other gradients as requested)
        """
        if requires_grad is None:
            requires_grad = {}

        # Limit steps for memory control
        actual_steps = min(num_steps, self.max_grad_steps) if self.max_grad_steps > 0 else num_steps

        # Reset loss and gradients
        self.solver.reset_loss()
        self.solver.reset_gradients()

        # Select loss kernel
        loss_kernel = self._get_loss_kernel(loss_type)

        # Run with autodiff tape
        with ti.ad.Tape(loss=self.solver.loss_field):
            # Forward simulation
            for _ in range(actual_steps):
                self.solver.step()

            # Compute loss (inside tape for gradient flow)
            loss_kernel()

        # Extract results
        results = {'loss': self.solver.loss_field[None]}

        # Extract gradients
        if requires_grad.get('ogden_mu', False):
            results['grad_ogden_mu'] = self.solver.ogden_mu.grad.to_numpy()[:self.solver.n_ogden]

        if requires_grad.get('ogden_alpha', False):
            results['grad_ogden_alpha'] = self.solver.ogden_alpha.grad.to_numpy()[:self.solver.n_ogden]

        if requires_grad.get('maxwell_G', False) and self.solver.n_maxwell > 0:
            results['grad_maxwell_G'] = self.solver.maxwell_G.grad.to_numpy()

        if requires_grad.get('maxwell_tau', False) and self.solver.n_maxwell > 0:
            results['grad_maxwell_tau'] = self.solver.maxwell_tau.grad.to_numpy()

        if requires_grad.get('initial_x', False):
            results['grad_initial_x'] = self.solver.fields.x.grad.to_numpy()

        if requires_grad.get('initial_v', False):
            results['grad_initial_v'] = self.solver.fields.v.grad.to_numpy()

        if requires_grad.get('F', False):
            results['grad_F'] = self.solver.fields.F.grad.to_numpy()

        if requires_grad.get('b_bar_e', False) and self.solver.fields.b_bar_e is not None:
            results['grad_b_bar_e'] = self.solver.fields.b_bar_e.grad.to_numpy()

        return results

    def _get_loss_kernel(self, loss_type: str):
        """Get the appropriate loss kernel function"""
        if loss_type == 'position':
            if self._target_x is None:
                raise ValueError("Target positions not set. Call set_target_positions() first.")
            return self._compute_position_loss
        elif loss_type == 'velocity':
            if self._target_v is None:
                raise ValueError("Target velocities not set. Call set_target_velocities() first.")
            return self._compute_velocity_loss
        elif loss_type == 'kinetic_energy':
            if self._target_energy is None:
                raise ValueError("Target kinetic energy not set. Call set_target_energy() first.")
            return self._compute_kinetic_energy_loss
        elif loss_type == 'com':
            if self._target_x is None:
                raise ValueError("Target positions not set. Call set_target_positions() first.")
            return self._compute_com_position_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Use 'position', 'velocity', 'kinetic_energy', or 'com'.")

    def compute_gradient_wrt_material_params(
        self,
        num_steps: int,
        loss_type: str = 'position'
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients with respect to material parameters

        Args:
            num_steps: Number of simulation steps
            loss_type: Type of loss to compute

        Returns:
            Dictionary of gradients
        """
        return self.run_with_gradients(
            num_steps,
            loss_type=loss_type,
            requires_grad={
                'ogden_mu': True,
                'ogden_alpha': True,
                'maxwell_G': True,
                'maxwell_tau': True
            }
        )

    def compute_gradient_wrt_initial_state(
        self,
        num_steps: int,
        loss_type: str = 'position'
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients with respect to initial particle state

        Args:
            num_steps: Number of simulation steps
            loss_type: Type of loss to compute

        Returns:
            Dictionary of gradients
        """
        return self.run_with_gradients(
            num_steps,
            loss_type=loss_type,
            requires_grad={
                'initial_x': True,
                'initial_v': True
            }
        )

    def get_particle_data(self):
        """Get current particle data"""
        return self.solver.get_particle_data()

    def get_energy_data(self):
        """Get current energy data"""
        return self.solver.get_energy_data()

    @property
    def loss_field(self):
        """Access to the loss field for custom loss functions"""
        return self.solver.loss_field

    @property
    def fields(self):
        """Access to particle fields"""
        return self.solver.fields
