"""
Advanced MPM Solver Usage Examples

Demonstrates:
1. Using multi-term Ogden hyperelasticity
2. Configuring Maxwell viscoelasticity
3. Setting up contact with separate normal/tangent stiffness
4. Custom particle initialization
5. Energy tracking and analysis
"""

import numpy as np
import taichi as ti
from pathlib import Path

# Import MPM components
from xengym.mpm import MPMConfig, MPMSolver


def example_1_multi_ogden():
    """Example 1: Multi-term Ogden model (4 terms)"""
    print("=" * 60)
    print("Example 1: Multi-term Ogden Hyperelasticity")
    print("=" * 60)

    ti.init(arch=ti.cpu)

    # Create configuration with 4 Ogden terms
    config = MPMConfig()
    config.grid.grid_size = (32, 32, 32)
    config.grid.dx = 0.01
    config.time.dt = 1e-4
    config.time.num_steps = 500

    # Configure 4-term Ogden model
    config.material.ogden.mu = [1e5, 5e4, 2e4, 1e4]
    config.material.ogden.alpha = [3.0, 2.0, -1.0, -2.0]
    config.material.ogden.kappa = 1e6

    print(f"Ogden terms: {len(config.material.ogden.mu)}")
    print(f"mu = {config.material.ogden.mu}")
    print(f"alpha = {config.material.ogden.alpha}")

    # Create particles
    n_particles = 100
    positions = np.random.rand(n_particles, 3).astype(np.float32) * 0.1 + 0.15

    # Create and run solver
    solver = MPMSolver(config, n_particles)
    solver.initialize_particles(positions)

    for step in range(config.time.num_steps):
        solver.step()

        if step % 100 == 0:
            energy_data = solver.get_energy_data()
            print(f"Step {step}: E_elastic = {energy_data['E_elastic']:.6e}")

    print("✓ Multi-term Ogden example complete\n")


def example_2_maxwell_viscoelasticity():
    """Example 2: Maxwell viscoelasticity with multiple branches"""
    print("=" * 60)
    print("Example 2: Maxwell Viscoelasticity")
    print("=" * 60)

    ti.init(arch=ti.cpu)

    # Create configuration with Maxwell branches
    config = MPMConfig()
    config.grid.grid_size = (32, 32, 32)
    config.grid.dx = 0.01
    config.time.dt = 1e-4
    config.time.num_steps = 1000

    # Configure Maxwell branches with different relaxation times
    config.material.maxwell.G = [5e4, 2e4, 1e4]  # 3 branches
    config.material.maxwell.tau = [0.005, 0.02, 0.1]  # Fast, medium, slow

    print(f"Maxwell branches: {len(config.material.maxwell.G)}")
    print(f"G = {config.material.maxwell.G}")
    print(f"tau = {config.material.maxwell.tau}")

    # Create particles
    n_particles = 100
    positions = np.random.rand(n_particles, 3).astype(np.float32) * 0.1 + 0.15

    # Create solver
    solver = MPMSolver(config, n_particles)
    solver.initialize_particles(positions)

    # Apply initial deformation
    particle_data = solver.get_particle_data()
    F_init = particle_data['F'].copy()
    F_init[:, 0, 0] *= 1.15  # 15% stretch in x
    solver.fields.F.from_numpy(F_init)

    # Run and track viscous dissipation
    print("\nRunning viscoelastic relaxation...")
    for step in range(config.time.num_steps):
        solver.step()

        if step % 200 == 0:
            energy_data = solver.get_energy_data()
            print(f"Step {step}: E_elastic = {energy_data['E_elastic']:.6e}, "
                  f"E_viscous_cum = {energy_data['E_viscous_cum']:.6e}")

    print("✓ Maxwell viscoelasticity example complete\n")


def example_3_contact_friction():
    """Example 3: Contact with separate normal/tangent stiffness"""
    print("=" * 60)
    print("Example 3: Contact and Friction")
    print("=" * 60)

    ti.init(arch=ti.cpu)

    # Create configuration with contact
    config = MPMConfig()
    config.grid.grid_size = (64, 64, 32)
    config.grid.dx = 0.01
    config.time.dt = 1e-4
    config.time.num_steps = 500

    # Configure separate normal/tangent stiffness
    config.contact.enable_contact = True
    config.contact.normal_stiffness = 1e6   # Stiff normal contact
    config.contact.tangent_stiffness = 5e5  # Softer tangential friction
    config.contact.mu_s = 0.7  # Static friction
    config.contact.mu_k = 0.5  # Kinetic friction

    print(f"Normal stiffness: {config.contact.normal_stiffness:.2e}")
    print(f"Tangent stiffness: {config.contact.tangent_stiffness:.2e}")
    print(f"Friction coefficients: μ_s = {config.contact.mu_s}, μ_k = {config.contact.mu_k}")

    # Create particles (box on ground)
    n_particles = 200
    positions = np.random.rand(n_particles, 3).astype(np.float32)
    positions[:, 0] = positions[:, 0] * 0.1 + 0.25
    positions[:, 1] = positions[:, 1] * 0.1 + 0.25
    positions[:, 2] = positions[:, 2] * 0.05 + 0.05  # Just above ground

    # Initial horizontal velocity (sliding)
    velocities = np.zeros_like(positions)
    velocities[:, 0] = 0.3

    # Create and run solver
    solver = MPMSolver(config, n_particles)
    solver.initialize_particles(positions, velocities=velocities)

    print("\nRunning contact simulation...")
    for step in range(config.time.num_steps):
        solver.step()

        if step % 100 == 0:
            particle_data = solver.get_particle_data()
            v_avg = np.mean(particle_data['v'], axis=0)
            print(f"Step {step}: avg_vel_x = {v_avg[0]:.4f} m/s")

    print("✓ Contact friction example complete\n")


def example_4_energy_analysis():
    """Example 4: Comprehensive energy tracking"""
    print("=" * 60)
    print("Example 4: Energy Analysis")
    print("=" * 60)

    ti.init(arch=ti.cpu)

    # Create configuration
    config = MPMConfig()
    config.grid.grid_size = (32, 32, 32)
    config.grid.dx = 0.01
    config.time.dt = 1e-4
    config.time.num_steps = 500

    # Enable all energy tracking features
    config.material.maxwell.G = [3e4]
    config.material.maxwell.tau = [0.02]
    config.contact.enable_contact = True

    # Create particles
    n_particles = 100
    positions = np.random.rand(n_particles, 3).astype(np.float32) * 0.1 + 0.2
    velocities = np.random.randn(n_particles, 3).astype(np.float32) * 0.1

    # Create solver
    solver = MPMSolver(config, n_particles)
    solver.initialize_particles(positions, velocities=velocities)

    # Track energy history
    energy_history = []

    print("\nTracking all energy components...")
    for step in range(config.time.num_steps):
        solver.step()

        if step % 50 == 0:
            energy_data = solver.get_energy_data()
            energy_history.append({
                'step': step,
                'E_kin': energy_data['E_kin'],
                'E_elastic': energy_data['E_elastic'],
                'E_viscous_cum': energy_data['E_viscous_cum'],
                'E_proj_cum': energy_data['E_proj_cum']
            })

            # Compute total energy (kinetic + elastic)
            E_total = energy_data['E_kin'] + energy_data['E_elastic']

            print(f"Step {step}:")
            print(f"  E_kin       = {energy_data['E_kin']:.6e}")
            print(f"  E_elastic   = {energy_data['E_elastic']:.6e}")
            print(f"  E_total     = {E_total:.6e}")
            print(f"  E_visc_cum  = {energy_data['E_viscous_cum']:.6e}")
            print(f"  E_proj_cum  = {energy_data['E_proj_cum']:.6e}")

    # Save energy history to CSV
    import csv
    output_path = Path("output")
    output_path.mkdir(exist_ok=True)

    with open(output_path / "energy_analysis.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'E_kin', 'E_elastic', 'E_viscous_cum', 'E_proj_cum'])
        writer.writeheader()
        writer.writerows(energy_history)

    print(f"\n✓ Energy analysis complete. Results saved to output/energy_analysis.csv\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("MPM Solver Advanced Usage Examples")
    print("=" * 60 + "\n")

    examples = [
        example_1_multi_ogden,
        example_2_maxwell_viscoelasticity,
        example_3_contact_friction,
        example_4_energy_analysis
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"✗ Example failed: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("All examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
