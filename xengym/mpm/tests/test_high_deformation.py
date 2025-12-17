"""
High-Deformation End-to-End Gradient Validation Tests

Tests material parameter gradients (ogden_mu, initial_x) under high deformation
conditions where the gradients are more significant and easier to verify.

Run with: pytest xengym/mpm/tests/test_high_deformation.py -v
"""
import pytest
import numpy as np

# Skip entire module if Taichi not available
taichi = pytest.importorskip("taichi")

# Import centralized gradient tolerances (Tier C for high-deformation tests)
from xengym.mpm.tests.conftest import GRADIENT_RTOL_LOOSE, GRADIENT_RTOL_RELAXED


def create_high_deformation_config():
    """Create MPM config for high-deformation testing"""
    from xengym.mpm import (
        MPMConfig, GridConfig, TimeConfig, OgdenConfig,
        MaterialConfig, ContactConfig, OutputConfig
    )

    return MPMConfig(
        grid=GridConfig(grid_size=[16, 16, 16], dx=0.02),
        time=TimeConfig(dt=5e-5, num_steps=100),
        material=MaterialConfig(
            density=1000.0,
            ogden=OgdenConfig(mu=[500.0], alpha=[2.0], kappa=2500.0),
            maxwell_branches=[],
            enable_bulk_viscosity=False
        ),
        contact=ContactConfig(enable_contact=False),
        output=OutputConfig()
    )


def create_cube_particles():
    """Create a small cube of particles"""
    positions = np.array([
        [0.14, 0.14, 0.14],
        [0.16, 0.14, 0.14],
        [0.14, 0.16, 0.14],
        [0.16, 0.16, 0.14],
        [0.14, 0.14, 0.16],
        [0.16, 0.14, 0.16],
        [0.14, 0.16, 0.16],
        [0.16, 0.16, 0.16],
    ], dtype=np.float32)

    velocities = np.zeros((8, 3), dtype=np.float32)
    velocities[:, 2] = -0.5  # Downward velocity

    return positions, velocities


def apply_prestrain(solver, n_particles):
    """Apply 30% compression pre-strain in z-direction"""
    F_initial = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.7]
    ], dtype=np.float32)

    for p in range(n_particles):
        solver.solver.fields.F[p] = F_initial


class TestHighDeformationOgdenMu:
    """Test ogden_mu gradient under high deformation"""

    @pytest.mark.slow
    @pytest.mark.gradient
    def test_ogden_mu_gradient_accuracy(self, init_taichi):
        """Verify ogden_mu gradient matches numerical finite difference"""
        from xengym.mpm import ManualAdjointMPMSolver

        config = create_high_deformation_config()
        n_particles = 8
        num_steps = 50

        solver = ManualAdjointMPMSolver(config, n_particles, max_grad_steps=num_steps)

        positions, velocities = create_cube_particles()

        # Target: particles return to original z-position (elastic rebound)
        target_positions = positions.copy()
        target_positions[:, 2] += 0.02

        solver.initialize_particles(positions, velocities)
        apply_prestrain(solver, n_particles)
        solver.set_target_positions(target_positions)

        # Forward + backward
        result_mu = solver.solve_with_gradients(
            num_steps=num_steps,
            loss_type='position',
            requires_grad={'ogden_mu': True, 'initial_x': True}
        )

        grad_mu_analytic = result_mu.get('grad_ogden_mu', np.zeros(1))[0]

        # Numerical verification
        init_x = solver.adj_fields.x_history.to_numpy()[0].copy()
        init_v = solver.adj_fields.v_history.to_numpy()[0].copy()
        init_F = solver.adj_fields.F_history.to_numpy()[0].copy()
        original_mu = solver.solver.ogden_mu[0]

        eps_mu = 10.0

        # f(mu + eps)
        solver.solver.ogden_mu[0] = original_mu + eps_mu
        solver.solver.initialize_particles(init_x, init_v)
        for p in range(n_particles):
            solver.solver.fields.F[p] = init_F[p]
        solver.run_forward_with_storage(num_steps)
        final_x_plus = solver.solver.fields.x.to_numpy()
        loss_plus = 0.5 * np.sum((final_x_plus - target_positions)**2)

        # f(mu - eps)
        solver.solver.ogden_mu[0] = original_mu - eps_mu
        solver.solver.initialize_particles(init_x, init_v)
        for p in range(n_particles):
            solver.solver.fields.F[p] = init_F[p]
        solver.run_forward_with_storage(num_steps)
        final_x_minus = solver.solver.fields.x.to_numpy()
        loss_minus = 0.5 * np.sum((final_x_minus - target_positions)**2)

        # Restore
        solver.solver.ogden_mu[0] = original_mu

        grad_mu_numerical = (loss_plus - loss_minus) / (2 * eps_mu)

        # Check gradient
        if abs(grad_mu_numerical) > 1e-12:
            rel_error = abs(grad_mu_analytic - grad_mu_numerical) / (abs(grad_mu_numerical) + 1e-15)
            sign_match = (grad_mu_analytic * grad_mu_numerical) > 0

            assert sign_match, f"Sign mismatch: analytic={grad_mu_analytic:.6e}, numerical={grad_mu_numerical:.6e}"
            assert rel_error < GRADIENT_RTOL_LOOSE, f"Relative error {rel_error:.4f} exceeds Tier C ({GRADIENT_RTOL_LOOSE*100:.0f}%)"
        else:
            assert abs(grad_mu_analytic) < 1e-8, "Expected small gradient when numerical is near zero"


class TestHighDeformationInitialX:
    """Test initial_x gradient under high deformation"""

    @pytest.mark.slow
    @pytest.mark.gradient
    def test_initial_x_gradient_accuracy(self, init_taichi):
        """Verify initial_x gradient matches numerical finite difference"""
        from xengym.mpm import ManualAdjointMPMSolver

        config = create_high_deformation_config()
        n_particles = 8
        num_steps = 50

        solver = ManualAdjointMPMSolver(config, n_particles, max_grad_steps=num_steps)

        positions, velocities = create_cube_particles()

        target_positions = positions.copy()
        target_positions[:, 2] += 0.02

        solver.initialize_particles(positions, velocities)
        apply_prestrain(solver, n_particles)
        solver.set_target_positions(target_positions)

        # Forward + backward
        result_x = solver.solve_with_gradients(
            num_steps=num_steps,
            loss_type='position',
            requires_grad={'initial_x': True}
        )

        grad_x_analytic = result_x.get('grad_initial_x', np.zeros((n_particles, 3)))

        # Pick first particle, z-component
        particle_idx = 0
        dim_idx = 2
        grad_x0_z_analytic = grad_x_analytic[particle_idx, dim_idx]

        # Numerical verification
        init_x = solver.adj_fields.x_history.to_numpy()[0].copy()
        init_v = solver.adj_fields.v_history.to_numpy()[0].copy()
        init_F = solver.adj_fields.F_history.to_numpy()[0].copy()

        eps_x = 1e-4

        # f(x + eps)
        perturbed_x_plus = init_x.copy()
        perturbed_x_plus[particle_idx, dim_idx] += eps_x
        solver.solver.initialize_particles(perturbed_x_plus, init_v)
        for p in range(n_particles):
            solver.solver.fields.F[p] = init_F[p]
        solver.run_forward_with_storage(num_steps)
        final_x_plus = solver.solver.fields.x.to_numpy()
        loss_plus = 0.5 * np.sum((final_x_plus - target_positions)**2)

        # f(x - eps)
        perturbed_x_minus = init_x.copy()
        perturbed_x_minus[particle_idx, dim_idx] -= eps_x
        solver.solver.initialize_particles(perturbed_x_minus, init_v)
        for p in range(n_particles):
            solver.solver.fields.F[p] = init_F[p]
        solver.run_forward_with_storage(num_steps)
        final_x_minus = solver.solver.fields.x.to_numpy()
        loss_minus = 0.5 * np.sum((final_x_minus - target_positions)**2)

        grad_x0_z_numerical = (loss_plus - loss_minus) / (2 * eps_x)

        # Check gradient
        if abs(grad_x0_z_numerical) > 1e-12:
            rel_error = abs(grad_x0_z_analytic - grad_x0_z_numerical) / (abs(grad_x0_z_numerical) + 1e-15)
            sign_match = (grad_x0_z_analytic * grad_x0_z_numerical) > 0

            assert sign_match, f"Sign mismatch: analytic={grad_x0_z_analytic:.6e}, numerical={grad_x0_z_numerical:.6e}"
            assert rel_error < GRADIENT_RTOL_RELAXED, f"Relative error {rel_error:.4f} exceeds transitional ({GRADIENT_RTOL_RELAXED*100:.0f}%)"
        else:
            assert abs(grad_x0_z_analytic) < 1e-8, "Expected small gradient when numerical is near zero"
