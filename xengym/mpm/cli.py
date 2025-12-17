"""
CLI Entry Point for MPM Solver
"""
import argparse
import numpy as np
import taichi as ti
from pathlib import Path
from .config import MPMConfig, MaxwellBranchConfig
from .mpm_solver import MPMSolver
from .autodiff_wrapper import DifferentiableMPMSolver
from .stability import validate_config


def create_particle_box(center, size, spacing, density=1000.0):
    """
    Create particles in a box region

    Args:
        center: Box center (x, y, z)
        size: Box size (width, height, depth)
        spacing: Particle spacing
        density: Material density

    Returns:
        positions: (n_particles, 3) array
        volumes: (n_particles,) array
    """
    half_size = np.array(size) / 2
    min_corner = np.array(center) - half_size
    max_corner = np.array(center) + half_size

    # Generate grid of particles
    x = np.arange(min_corner[0], max_corner[0], spacing)
    y = np.arange(min_corner[1], max_corner[1], spacing)
    z = np.arange(min_corner[2], max_corner[2], spacing)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    # Particle volume
    volume = spacing ** 3
    volumes = np.full(len(positions), volume)

    return positions.astype(np.float32), volumes.astype(np.float32)


def run_simple_drop_test(config_path: str = None, output_dir: str = "output", arch=ti.gpu,
                         strict: bool = True):
    """
    Run a simple drop test: box of particles falling under gravity

    Args:
        config_path: Path to configuration file (optional)
        output_dir: Output directory for results
        arch: Taichi backend architecture
        strict: If True, block on invalid config; if False, warn only
    """
    # Initialize Taichi
    ti.init(arch=arch, device_memory_GB=4.0)

    # Load or create configuration
    if config_path:
        config = MPMConfig.from_json(config_path)
    else:
        # Default configuration
        config = MPMConfig()
        config.grid.grid_size = (64, 64, 64)
        config.grid.dx = 0.01
        config.time.dt = 1e-4
        config.time.num_steps = 1000
        config.material.density = 1000.0
        config.contact.enable_contact = True

    # Run stability validation (default-on per FR-8)
    print("\n" + "=" * 60)
    print("Configuration Validation (Drucker + Time Step)")
    print("=" * 60)
    is_valid, messages = validate_config(config, verbose=True)

    if not is_valid:
        if strict:
            print("\n✗ Configuration failed validation. Aborting.")
            print("  Pass strict=False to run with warnings only.")
            return
        else:
            print("\n⚠ Configuration has issues but proceeding (strict=False)")

    # Create particles
    positions, volumes = create_particle_box(
        center=[0.32, 0.32, 0.5],
        size=[0.1, 0.1, 0.1],
        spacing=config.grid.dx * 0.5
    )
    n_particles = len(positions)

    print(f"Created {n_particles} particles")

    # Create solver
    solver = MPMSolver(config, n_particles)
    solver.initialize_particles(positions, volumes=volumes)

    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save initial configuration
    config.save_json(str(output_path / "config.json"))

    # Run simulation
    print(f"Running simulation for {config.time.num_steps} steps...")

    energy_history = []

    for step in range(config.time.num_steps):
        solver.step()

        # Record energy (extended output per FR-6/7)
        if step % 10 == 0:
            energy_data = solver.get_energy_data()
            energy_history.append({
                'step': step,
                'time': step * config.time.dt,
                'E_kin': energy_data['E_kin'],
                'E_elastic': energy_data['E_elastic'],
                'E_viscous_step': energy_data['E_viscous_step'],
                'E_viscous_cum': energy_data['E_viscous_cum'],
                'E_proj_step': energy_data['E_proj_step'],  # ΔE_proj_step
                'E_proj_cum': energy_data['E_proj_cum']
            })

            if step % 100 == 0:
                print(f"Step {step}: E_kin={energy_data['E_kin']:.3e}, "
                      f"E_elastic={energy_data['E_elastic']:.3e}")

        # Save particle data periodically
        if step % config.output.output_interval == 0:
            particle_data = solver.get_particle_data()
            np.savez(
                str(output_path / f"particles_{step:06d}.npz"),
                x=particle_data['x'],
                v=particle_data['v'],
                F=particle_data['F']
            )

    # Save energy history (extended fields per FR-6/7)
    import csv
    fieldnames = ['step', 'time', 'E_kin', 'E_elastic',
                  'E_viscous_step', 'E_viscous_cum', 'E_proj_step', 'E_proj_cum']
    with open(output_path / "energy.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(energy_history)

    print(f"Simulation complete. Results saved to {output_path}")


def run_maxwell_relaxation_test(config_path: str = None, output_dir: str = "output", arch=ti.gpu,
                                 strict: bool = True):
    """
    Run Maxwell viscoelastic relaxation test: stretch then release

    Args:
        config_path: Path to configuration file (optional)
        output_dir: Output directory for results
        arch: Taichi backend architecture
        strict: If True, block on invalid config; if False, warn only
    """
    # Initialize Taichi
    ti.init(arch=arch, device_memory_GB=4.0)

    # Load or create configuration with Maxwell enabled
    if config_path:
        config = MPMConfig.from_json(config_path)
    else:
        config = MPMConfig()
        config.grid.grid_size = (32, 32, 32)
        config.grid.dx = 0.01
        config.time.dt = 1e-4
        config.time.num_steps = 2000
        # Enable Maxwell viscoelasticity with two branches
        config.material.maxwell_branches = [
            MaxwellBranchConfig(G=5e4, tau=0.01),
            MaxwellBranchConfig(G=2e4, tau=0.1)
        ]
        config.contact.enable_contact = False  # No contact for pure relaxation test

    # Run stability validation (default-on per FR-8)
    print("\n" + "=" * 60)
    print("Configuration Validation (Drucker + Time Step)")
    print("=" * 60)
    is_valid, messages = validate_config(config, verbose=True)

    if not is_valid:
        if strict:
            print("\n✗ Configuration failed validation. Aborting.")
            print("  Pass strict=False to run with warnings only.")
            return
        else:
            print("\n⚠ Configuration has issues but proceeding (strict=False)")

    # Create particles
    positions, volumes = create_particle_box(
        center=[0.16, 0.16, 0.16],
        size=[0.06, 0.06, 0.06],
        spacing=config.grid.dx * 0.5
    )
    n_particles = len(positions)

    print(f"Created {n_particles} particles for Maxwell relaxation test")
    if config.material.maxwell_branches:
        G_list = [br.G for br in config.material.maxwell_branches]
        tau_list = [br.tau for br in config.material.maxwell_branches]
        print(f"Maxwell branches: {len(config.material.maxwell_branches)}, G={G_list}, tau={tau_list}")
    else:
        print("No Maxwell branches configured")

    # Create solver
    solver = MPMSolver(config, n_particles)
    solver.initialize_particles(positions, volumes=volumes)

    # Apply initial deformation (stretch in x direction)
    particle_data = solver.get_particle_data()
    stretched_F = particle_data['F'].copy()
    stretched_F[:, 0, 0] *= 1.2  # 20% stretch in x
    solver.fields.F.from_numpy(stretched_F)

    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config.save_json(str(output_path / "config_maxwell.json"))

    # Run simulation and record relaxation
    print(f"Running Maxwell relaxation test for {config.time.num_steps} steps...")

    relaxation_data = []

    for step in range(config.time.num_steps):
        solver.step()

        if step % 10 == 0:
            particle_data = solver.get_particle_data()
            energy_data = solver.get_energy_data()

            # Measure average deformation
            F_avg = np.mean(particle_data['F'], axis=0)
            stretch_x = F_avg[0, 0]  # Track x-direction stretch

            relaxation_data.append({
                'step': step,
                'time': step * config.time.dt,
                'stretch_x': stretch_x,
                'E_elastic': energy_data['E_elastic'],
                'E_viscous_cum': energy_data['E_viscous_cum']
            })

            if step % 200 == 0:
                print(f"Step {step}: stretch_x={stretch_x:.4f}, "
                      f"E_elastic={energy_data['E_elastic']:.3e}")

    # Save relaxation curve
    import csv
    with open(output_path / "maxwell_relaxation.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'time', 'stretch_x', 'E_elastic', 'E_viscous_cum'])
        writer.writeheader()
        writer.writerows(relaxation_data)

    print(f"Maxwell relaxation test complete. Results saved to {output_path}")


def run_friction_test(config_path: str = None, output_dir: str = "output", arch=ti.gpu,
                      strict: bool = True):
    """
    Run contact friction test: box sliding on ground

    Args:
        config_path: Path to configuration file (optional)
        output_dir: Output directory for results
        arch: Taichi backend architecture
        strict: If True, block on invalid config; if False, warn only
    """
    # Initialize Taichi
    ti.init(arch=arch, device_memory_GB=4.0)

    # Load or create configuration with contact friction
    if config_path:
        config = MPMConfig.from_json(config_path)
    else:
        config = MPMConfig()
        config.grid.grid_size = (64, 64, 32)
        config.grid.dx = 0.01
        config.time.dt = 1e-4
        config.time.num_steps = 1000
        config.contact.enable_contact = True
        config.contact.contact_stiffness_normal = 1e6
        config.contact.contact_stiffness_tangent = 5e5  # Friction stiffness

    # Run stability validation (default-on per FR-8)
    print("\n" + "=" * 60)
    print("Configuration Validation (Drucker + Time Step)")
    print("=" * 60)
    is_valid, messages = validate_config(config, verbose=True)

    if not is_valid:
        if strict:
            print("\n✗ Configuration failed validation. Aborting.")
            print("  Pass strict=False to run with warnings only.")
            return
        else:
            print("\n⚠ Configuration has issues but proceeding (strict=False)")

    # Create particles (box positioned above ground)
    positions, volumes = create_particle_box(
        center=[0.32, 0.32, 0.08],  # Just above ground
        size=[0.1, 0.1, 0.04],
        spacing=config.grid.dx * 0.5
    )
    n_particles = len(positions)

    print(f"Created {n_particles} particles for friction test")
    print(f"Contact parameters: normal_k={config.contact.contact_stiffness_normal}, "
          f"tangent_k={config.contact.contact_stiffness_tangent}")

    # Create solver
    solver = MPMSolver(config, n_particles)

    # Initialize with horizontal velocity (sliding motion)
    velocities = np.zeros_like(positions)
    velocities[:, 0] = 0.5  # Velocity in x direction
    solver.initialize_particles(positions, velocities=velocities, volumes=volumes)

    # Output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config.save_json(str(output_path / "config_friction.json"))

    # Run simulation and record friction behavior
    print(f"Running friction test for {config.time.num_steps} steps...")

    friction_data = []
    initial_x = np.mean(positions[:, 0])  # Record initial position for displacement

    for step in range(config.time.num_steps):
        solver.step()

        if step % 10 == 0:
            particle_data = solver.get_particle_data()
            energy_data = solver.get_energy_data()

            # Measure average velocity and position
            v_avg = np.mean(particle_data['v'], axis=0)
            x_avg = np.mean(particle_data['x'], axis=0)
            tangent_disp = x_avg[0] - initial_x  # Tangential displacement

            friction_data.append({
                'step': step,
                'time': step * config.time.dt,
                'tangent_disp': tangent_disp,
                'tangent_vel': v_avg[0],
                'E_kin': energy_data['E_kin'],
                'E_elastic': energy_data['E_elastic'],
                'E_proj_step': energy_data['E_proj_step'],
                'E_proj_cum': energy_data['E_proj_cum']
            })

            if step % 100 == 0:
                print(f"Step {step}: vel_x={v_avg[0]:.4f}, disp_x={tangent_disp:.4f}")

    # Save friction curve (unified format with gelslim)
    import csv
    fieldnames = ['step', 'time', 'tangent_disp', 'tangent_vel',
                  'E_kin', 'E_elastic', 'E_proj_step', 'E_proj_cum']
    with open(output_path / "friction_curve.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(friction_data)

    print(f"Friction test complete. Results saved to {output_path}")


def run_dt_convergence_test(output_dir: str = "output", arch=ti.cpu,
                            dt_factors: list = None, base_dt: float = 2e-4,
                            base_steps: int = 400):
    """
    Run dt convergence study for E_proj/E_viscous ratio (FR-5)

    This tests that |E_proj_cum| / E_viscous_cum converges as O(dt),
    verifying the projection energy tracking implementation.

    Args:
        output_dir: Output directory for results
        arch: Taichi backend architecture (note: each run re-initializes Taichi)
        dt_factors: List of dt multipliers (default: [1.0, 0.5, 0.25, 0.125])
        base_dt: Base time step in seconds
        base_steps: Base number of steps
    """
    from .scripts.plot_dt_convergence import run_convergence_study, plot_convergence

    if dt_factors is None:
        dt_factors = [1.0, 0.5, 0.25, 0.125]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MPM Time Step Convergence Study (FR-5)")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"dt factors: {dt_factors}")
    print(f"Base dt: {base_dt:.2e} s")
    print(f"Base steps: {base_steps}")

    base_config = {
        'grid_size': (32, 32, 32),
        'dx': 0.02,
        'dt': base_dt,
        'density': 1000.0,
        'mu': [30000.0],
        'alpha': [2.0],
        'kappa': 300000.0
    }

    # Run convergence study
    results = run_convergence_study(
        dt_factors=sorted(dt_factors, reverse=True),
        output_dir=output_path,
        base_config=base_config
    )

    # Plot results
    print("\nGenerating convergence plot...")
    plot_convergence(results, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("Convergence Study Complete")
    print("=" * 60)

    if len(results) >= 2:
        import numpy as np
        # Filter valid results
        valid_results = [r for r in results if r.get('ratio_valid', True)]
        if len(valid_results) >= 2:
            dt_values = np.array([r['dt'] for r in valid_results])
            ratios = np.array([r['ratio'] for r in valid_results])
            valid_mask = ratios > 1e-15

            if np.sum(valid_mask) >= 2:
                log_dt = np.log(dt_values[valid_mask])
                log_ratio = np.log(ratios[valid_mask])
                slope, _ = np.polyfit(log_dt, log_ratio, 1)

                print(f"\nMeasured convergence order: {slope:.3f}")
                print(f"Expected (first-order): 1.0")

                if 0.8 <= slope <= 1.2:
                    print("✓ PASS: Convergence is approximately first-order")
                else:
                    print(f"⚠ WARNING: Convergence order ({slope:.2f}) differs from expected (1.0)")
        else:
            print("\n⚠ Insufficient valid data points for convergence analysis")

    print(f"\nResults saved to: {output_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="MPM Solver CLI - Verification Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenes:
  drop           - Basic drop test (particles falling under gravity)
  maxwell        - Maxwell viscoelastic relaxation test
  friction       - Contact friction sliding test
  uniaxial       - Uniaxial tension test (stress-strain vs Ogden theory)
  shear          - Pure shear test
  objectivity    - Objectivity test (stress invariant under rigid rotation)
  energy         - Energy conservation test
  energy_conv    - Energy convergence with projection tracking
  gelslim        - GelSlim-style stick-slip / incipient slip test
  hertz          - Hertzian contact / elastic impact convergence test
  dt_conv        - dt convergence study (|E_proj|/E_viscous ratio, FR-5)
  all_validation - Run all validation scenes

Examples:
  python -m xengym.mpm.cli --scene drop --arch cpu
  python -m xengym.mpm.cli --scene uniaxial --output results/
  python -m xengym.mpm.cli --scene all_validation --no-strict
        """
    )
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--scene', type=str, default='drop',
                        choices=['drop', 'maxwell', 'friction',
                                 'uniaxial', 'shear', 'objectivity',
                                 'energy', 'energy_conv', 'gelslim', 'hertz',
                                 'dt_conv', 'all_validation'],
                        help='Scene to simulate (see descriptions above)')
    parser.add_argument('--arch', type=str, default='gpu',
                        choices=['cpu', 'gpu', 'cuda', 'vulkan'],
                        help='Taichi backend architecture')
    parser.add_argument('--strict', dest='strict', action='store_true', default=True,
                        help='Enable strict stability checks (default: enabled)')
    parser.add_argument('--no-strict', dest='strict', action='store_false',
                        help='Disable strict stability checks (warn only)')

    args = parser.parse_args()

    # Map arch string to Taichi arch constant
    arch_map = {
        'cpu': ti.cpu,
        'gpu': ti.gpu,
        'cuda': ti.cuda,
        'vulkan': ti.vulkan
    }
    selected_arch = arch_map.get(args.arch, ti.gpu)

    print(f"Running scene: {args.scene} on arch: {args.arch}")
    print(f"Strict mode: {args.strict}")

    # Handle validation scenes
    validation_scenes = {
        'uniaxial': ['uniaxial'],
        'shear': ['shear'],
        'objectivity': ['objectivity'],
        'energy': ['energy'],
        'energy_conv': ['energy_convergence'],
        'gelslim': ['gelslim'],
        'hertz': ['hertz'],
        'all_validation': None  # None means all
    }

    if args.scene in validation_scenes:
        # Run validation scene(s)
        run_validation_scene(
            args.config, args.output, selected_arch,
            scenes=validation_scenes[args.scene],
            strict=args.strict
        )
    elif args.scene == 'drop':
        run_simple_drop_test(args.config, args.output, arch=selected_arch)
    elif args.scene == 'maxwell':
        run_maxwell_relaxation_test(args.config, args.output, arch=selected_arch)
    elif args.scene == 'friction':
        run_friction_test(args.config, args.output, arch=selected_arch)
    elif args.scene == 'dt_conv':
        # dt convergence study doesn't use arch since it re-initializes Taichi per run
        run_dt_convergence_test(args.output)
    else:
        print(f"Unknown scene: {args.scene}")


def run_validation_scene(config_path: str, output_dir: str, arch,
                         scenes: list = None, strict: bool = True):
    """
    Run validation scene(s) with stability checks

    Args:
        config_path: Path to configuration file (optional)
        output_dir: Output directory for results
        arch: Taichi backend architecture
        scenes: List of scene names to run
        strict: If True, block on invalid config; if False, warn only
    """
    from .validation import run_all_validations

    # Initialize Taichi
    ti.init(arch=arch, device_memory_GB=4.0)

    # Load or create configuration
    if config_path:
        config = MPMConfig.from_json(config_path)
    else:
        config = MPMConfig()
        config.grid.grid_size = (48, 48, 48)
        config.grid.dx = 0.01
        config.time.dt = 1e-4
        config.time.num_steps = 500
        config.material.density = 1000.0
        config.contact.enable_contact = True

    # Run stability validation
    print("\n" + "=" * 60)
    print("Configuration Validation")
    print("=" * 60)
    is_valid, messages = validate_config(config, verbose=True)

    if not is_valid:
        if strict:
            print("\n✗ Configuration failed validation in strict mode. Aborting.")
            print("  Use --no-strict to run with warnings only.")
            return
        else:
            print("\n⚠ Configuration has issues but proceeding (strict=False)")

    # Run validation scenes
    results = run_all_validations(config, output_dir, scenes=scenes)

    # Save config used
    output_path = Path(output_dir)
    config.save_json(str(output_path / "config_validation.json"))

    print(f"\nValidation complete. Config saved to {output_path / 'config_validation.json'}")


if __name__ == '__main__':
    main()
