"""
Autodiff Wrapper for MPM Solver

Provides automatic differentiation interface with multiple loss types.

IMPORTANT LIMITATIONS (Taichi AD v1.7.4):
    The P2G/G2P kernels contain atomic scatter/gather operations that are NOT
    supported by Taichi's autodiff. This means:
    - ✅ Loss computation w.r.t. final state IS differentiable
    - ❌ Gradient propagation through simulation steps (p2g/grid_op/g2p) is BLOCKED
    - ❌ Cannot compute gradients w.r.t. material parameters or initial state
          through forward simulation in current implementation

    For now, this wrapper provides the infrastructure and API. Full autodiff support
    requires either:
    1. Manual adjoint implementation for P2G/G2P (complex, out of current scope)
    2. Taichi version with improved AD support for atomic operations
    3. External AD framework (JAX/PyTorch) wrapping

    See TAICHI_AUTODIFF_LIMITATIONS.md for details.

Fixed issues: loss_type branching, target validation, proper loss kernels, energy kernel separation.
"""
import taichi as ti
import numpy as np
from typing import Dict, Optional
import warnings
from .mpm_solver import MPMSolver
from .config import MPMConfig


@ti.data_oriented
class DifferentiableMPMSolver:
    """Wrapper for MPM solver with automatic differentiation support.

    WARNING: Autodiff is currently NOT functional due to Taichi AD limitations.
    The P2G/G2P kernels contain atomic scatter/gather operations that are not
    supported by Taichi's autodiff system (v1.7.4). Calling gradient computation
    methods will raise NotImplementedError.

    This class provides the infrastructure for future autodiff support when:
    1. Taichi improves AD support for atomic operations
    2. Manual adjoint method is implemented for P2G/G2P
    3. External AD framework (JAX/PyTorch) wrapper is used

    See xengym/mpm/TAICHI_AUTODIFF_LIMITATIONS.md for details.
    """

    def __init__(self, config: MPMConfig, n_particles: int,
                 use_spd_ste: bool = True, max_grad_steps: int = 50):
        self.config = config
        self.n_particles = n_particles
        self.max_grad_steps = max_grad_steps
        self.solver = MPMSolver(config, n_particles, enable_grad=True, use_spd_ste=use_spd_ste)
        self._target_x = None
        self._target_v = None
        self._target_energy = None
        self._warned = False  # Track if we've shown the limitation warning

        # Emit warning about autodiff limitations
        warnings.warn(
            "DifferentiableMPMSolver: Autodiff is currently NOT functional.\n"
            "Taichi AD does not support atomic operations in P2G/G2P kernels.\n"
            "Gradient computation methods will raise NotImplementedError.\n"
            "See xengym/mpm/TAICHI_AUTODIFF_LIMITATIONS.md for details and alternatives.",
            category=UserWarning,
            stacklevel=2
        )

    def initialize_particles(self, positions, velocities=None, volumes=None):
        self.solver.initialize_particles(positions, velocities, volumes)

    def set_target_positions(self, target_positions: np.ndarray):
        if self._target_x is None:
            self._target_x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self._target_x.from_numpy(target_positions.astype(np.float32))

    def set_target_velocities(self, target_velocities: np.ndarray):
        if self._target_v is None:
            self._target_v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self._target_v.from_numpy(target_velocities.astype(np.float32))

    def set_target_energy(self, target_energy: float):
        if self._target_energy is None:
            self._target_energy = ti.field(dtype=ti.f32, shape=())
        self._target_energy[None] = target_energy

    @ti.kernel
    def _compute_position_loss(self):
        for p in range(self.n_particles):
            diff = self.solver.fields.x[p] - self._target_x[p]
            self.solver.loss_field[None] += diff.dot(diff)

    @ti.kernel
    def _compute_velocity_loss(self):
        for p in range(self.n_particles):
            diff = self.solver.fields.v[p] - self._target_v[p]
            self.solver.loss_field[None] += diff.dot(diff)

    @ti.kernel
    def _compute_kinetic_energy_loss(self):
        """Compute loss based on kinetic energy only.

        Note: This computes kinetic energy (E_kin = 0.5 * m * v^2) only,
        NOT total energy. For total energy matching (kinetic + elastic +
        viscous + projection), use 'total_energy' loss type or implement
        a custom loss kernel.
        """
        E_kin = 0.0
        for p in range(self.n_particles):
            v_p = self.solver.fields.v[p]
            m_p = self.solver.fields.mass[p]
            E_kin += 0.5 * m_p * v_p.dot(v_p)
        diff = E_kin - self._target_energy[None]
        self.solver.loss_field[None] += diff * diff

    @ti.kernel
    def _compute_com_loss(self):
        for p in range(self.n_particles):
            diff = self.solver.fields.x[p] - self._target_x[p]
            weight = self.solver.fields.mass[p]
            self.solver.loss_field[None] += weight * diff.dot(diff)

    def _validate_targets(self, loss_type: str):
        if loss_type in ('position', 'com'):
            if self._target_x is None:
                raise ValueError(f'Target positions must be set for loss_type="{loss_type}". Call set_target_positions() first.')
        elif loss_type == 'velocity':
            if self._target_v is None:
                raise ValueError('Target velocities must be set. Call set_target_velocities() first.')
        elif loss_type == 'kinetic_energy':
            if self._target_energy is None:
                raise ValueError('Target kinetic energy must be set. Call set_target_energy() first.')

    def _get_loss_kernel(self, loss_type: str):
        """Get loss computation kernel for the given loss type.

        Args:
            loss_type: Type of loss to compute.
                - 'position': Position matching loss
                - 'velocity': Velocity matching loss
                - 'kinetic_energy': Kinetic energy matching loss (E_kin only)
                - 'com': Center of mass matching loss

        Note: 'kinetic_energy' loss type computes KINETIC ENERGY only by default.
        For total energy (kinetic + elastic + viscous + projection),
        implement a custom loss kernel or see documentation.
        """
        kernels = {
            'position': self._compute_position_loss,
            'velocity': self._compute_velocity_loss,
            'kinetic_energy': self._compute_kinetic_energy_loss,
            'com': self._compute_com_loss,
        }
        if loss_type not in kernels:
            raise ValueError(f'Unknown loss_type: "{loss_type}". Valid: {list(kernels.keys())}')
        return kernels[loss_type]

    def run_with_gradients(self, num_steps: int, loss_type: str = 'position',
                           requires_grad: Optional[Dict[str, bool]] = None) -> Dict[str, np.ndarray]:
        if requires_grad is None:
            requires_grad = {}

        # CRITICAL: Block Tape execution due to Taichi AD limitation
        # P2G/G2P contain atomic scatter/gather which causes "Not supported" error in Tape
        raise NotImplementedError(
            "run_with_gradients() is currently BLOCKED due to Taichi AD v1.7.4 limitation.\n"
            "P2G/G2P kernels contain atomic scatter/gather operations that are not supported "
            "by Taichi's autodiff system. Running solver.step() inside ti.ad.Tape will fail with:\n"
            "  [auto_diff.cpp:taichi::lang::ADTransform::visit@1099] Not supported.\n\n"
            "Current status:\n"
            "  - ✅ Infrastructure ready (loss_field, needs_grad, STE)\n"
            "  - ✅ Loss kernels implemented (position/velocity/energy/com)\n"
            "  - ❌ Gradient backpropagation through simulation BLOCKED\n\n"
            "Possible solutions:\n"
            "  1. Wait for Taichi version with improved AD support\n"
            "  2. Implement manual adjoint method for P2G/G2P (complex)\n"
            "  3. Use external AD framework (JAX/PyTorch)\n\n"
            "See xengym/mpm/TAICHI_AUTODIFF_LIMITATIONS.md for detailed explanation."
        )

        # Code below is infrastructure-ready but blocked by Taichi limitation
        # Keeping for future when Taichi improves AD support

        self._validate_targets(loss_type)
        actual_steps = min(num_steps, self.max_grad_steps) if self.max_grad_steps > 0 else num_steps
        self.solver.reset_loss()
        self.solver.reset_gradients()
        loss_kernel = self._get_loss_kernel(loss_type)
        with ti.ad.Tape(loss=self.solver.loss_field):
            for _ in range(actual_steps):
                self.solver.step()
            loss_kernel()
        results = {'loss': self.solver.loss_field[None]}
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

    def compute_gradient_wrt_material_params(self, num_steps: int, loss_type: str = 'position') -> Dict[str, np.ndarray]:
        """BLOCKED: See run_with_gradients() for explanation"""
        return self.run_with_gradients(num_steps, loss_type=loss_type, requires_grad={'ogden_mu': True, 'ogden_alpha': True, 'maxwell_G': True, 'maxwell_tau': True})

    def compute_gradient_wrt_initial_state(self, num_steps: int, loss_type: str = 'position') -> Dict[str, np.ndarray]:
        """BLOCKED: See run_with_gradients() for explanation"""
        return self.run_with_gradients(num_steps, loss_type=loss_type, requires_grad={'initial_x': True, 'initial_v': True})

    def get_particle_data(self):
        return self.solver.get_particle_data()

    def get_energy_data(self):
        return self.solver.get_energy_data()

    @property
    def loss_field(self):
        return self.solver.loss_field

    @property
    def fields(self):
        return self.solver.fields
