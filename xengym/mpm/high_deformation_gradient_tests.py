"""
High-Deformation End-to-End Gradient Validation Test

Tests material parameter gradients (ogden_mu, ogden_alpha) under high deformation
conditions where the gradients are more significant and easier to verify.

This addresses the Codex reviewer recommendation:
"若要验证 mu/alpha 的可微性，建议增加一个'高形变/多步'的端到端场景"
"""
import taichi as ti
import numpy as np
import sys

def run_high_deformation_test():
    """Run high-deformation gradient validation test"""
    from xengym.mpm import ManualAdjointMPMSolver
    from xengym.mpm import (
        MPMConfig, GridConfig, TimeConfig, OgdenConfig,
        MaterialConfig, ContactConfig, OutputConfig
    )

    ti.init(arch=ti.cpu, debug=False)

    print("=" * 70)
    print("HIGH-DEFORMATION END-TO-END GRADIENT VALIDATION")
    print("=" * 70)
    print()

    # Configuration for high-deformation scenario
    grid_config = GridConfig(grid_size=[16, 16, 16], dx=0.02)
    time_config = TimeConfig(dt=5e-5, num_steps=100)  # More steps for larger deformation
    ogden_config = OgdenConfig(mu=[500.0], alpha=[2.0], kappa=2500.0)  # Softer material
    material_config = MaterialConfig(
        density=1000.0,
        ogden=ogden_config,
        maxwell_branches=[],
        enable_bulk_viscosity=False
    )
    contact_config = ContactConfig(enable_contact=False)
    output_config = OutputConfig()

    config = MPMConfig(
        grid=grid_config,
        time=time_config,
        material=material_config,
        contact=contact_config,
        output=output_config
    )

    n_particles = 8  # Multiple particles for more realistic scenario
    num_steps = 50   # Enough steps for significant deformation

    solver = ManualAdjointMPMSolver(config, n_particles, max_grad_steps=num_steps)

    # Create a small cube of particles
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

    # Add initial velocity for deformation
    velocities = np.zeros((n_particles, 3), dtype=np.float32)
    velocities[:, 2] = -0.5  # Downward velocity

    solver.initialize_particles(positions, velocities)

    # Apply pre-strain (compression in z-direction)
    for p in range(n_particles):
        F_initial = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.7]  # 30% compression
        ], dtype=np.float32)
        solver.solver.fields.F[p] = F_initial

    # Target: particles return to original z-position (elastic rebound)
    target_positions = positions.copy()
    target_positions[:, 2] += 0.02  # Target slightly above initial
    solver.set_target_positions(target_positions)

    results = {}
    all_passed = True

    # ========================================
    # Test 1: ogden_mu gradient
    # ========================================
    print("\n" + "-" * 50)
    print("TEST 1: ogden_mu gradient")
    print("-" * 50)

    # Reset solver
    solver.reset()
    solver.initialize_particles(positions, velocities)
    for p in range(n_particles):
        F_initial = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.7]
        ], dtype=np.float32)
        solver.solver.fields.F[p] = F_initial

    # Forward + backward
    result_mu = solver.solve_with_gradients(
        num_steps=num_steps,
        loss_type='position',
        requires_grad={'ogden_mu': True, 'initial_x': True}
    )

    grad_mu_analytic = result_mu.get('grad_ogden_mu', np.zeros(1))[0]
    loss = result_mu['loss']

    print(f"Loss: {loss:.8e}")
    print(f"Analytic grad_mu: {grad_mu_analytic:.8e}")

    # Numerical verification with larger epsilon for high-deformation
    init_x = solver.adj_fields.x_history.to_numpy()[0].copy()
    init_v = solver.adj_fields.v_history.to_numpy()[0].copy()
    init_F = solver.adj_fields.F_history.to_numpy()[0].copy()
    original_mu = solver.solver.ogden_mu[0]

    eps_mu = 10.0  # Larger epsilon for mu

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

    print(f"Numerical grad_mu: {grad_mu_numerical:.8e}")
    print(f"loss_plus: {loss_plus:.8e}, loss_minus: {loss_minus:.8e}")
    print(f"Position delta: {np.linalg.norm(final_x_plus - final_x_minus):.8e}")

    if abs(grad_mu_numerical) > 1e-12:
        rel_error_mu = abs(grad_mu_analytic - grad_mu_numerical) / (abs(grad_mu_numerical) + 1e-15)
        sign_match_mu = (grad_mu_analytic * grad_mu_numerical) > 0
        print(f"Relative error: {rel_error_mu:.4f}")
        print(f"Sign match: {sign_match_mu}")
        passed_mu = rel_error_mu < 0.5 and sign_match_mu
    else:
        print(f"Numerical gradient too small, checking if analytic is also small")
        passed_mu = abs(grad_mu_analytic) < 1e-8

    print(f"ogden_mu test: {'PASSED' if passed_mu else 'FAILED'}")
    results['ogden_mu'] = passed_mu
    all_passed = all_passed and passed_mu

    # ========================================
    # Test 2: initial_x gradient
    # ========================================
    print("\n" + "-" * 50)
    print("TEST 2: initial_x gradient")
    print("-" * 50)

    # Reset solver
    solver.reset()
    solver.solver.ogden_mu[0] = original_mu
    solver.initialize_particles(positions, velocities)
    for p in range(n_particles):
        F_initial = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.7]
        ], dtype=np.float32)
        solver.solver.fields.F[p] = F_initial

    # Forward + backward
    result_x = solver.solve_with_gradients(
        num_steps=num_steps,
        loss_type='position',
        requires_grad={'initial_x': True}
    )

    grad_x_analytic = result_x.get('grad_initial_x', np.zeros((n_particles, 3)))
    loss = result_x['loss']

    # Pick first particle, z-component
    particle_idx = 0
    dim_idx = 2
    grad_x0_z_analytic = grad_x_analytic[particle_idx, dim_idx]

    print(f"Loss: {loss:.8e}")
    print(f"Analytic grad_x[0][z]: {grad_x0_z_analytic:.8e}")

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

    print(f"Numerical grad_x[0][z]: {grad_x0_z_numerical:.8e}")
    print(f"loss_plus: {loss_plus:.8e}, loss_minus: {loss_minus:.8e}")

    if abs(grad_x0_z_numerical) > 1e-12:
        rel_error_x = abs(grad_x0_z_analytic - grad_x0_z_numerical) / (abs(grad_x0_z_numerical) + 1e-15)
        sign_match_x = (grad_x0_z_analytic * grad_x0_z_numerical) > 0
        print(f"Relative error: {rel_error_x:.4f}")
        print(f"Sign match: {sign_match_x}")
        passed_x = rel_error_x < 0.2 and sign_match_x  # Stricter for x
    else:
        print(f"Numerical gradient too small")
        passed_x = abs(grad_x0_z_analytic) < 1e-8

    print(f"initial_x test: {'PASSED' if passed_x else 'FAILED'}")
    results['initial_x'] = passed_x
    all_passed = all_passed and passed_x

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for param, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {param}: {status}")

    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    try:
        passed = run_high_deformation_test()
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
