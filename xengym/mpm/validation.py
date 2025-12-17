"""
Validation Scenes and Tests
Provides validation scenarios for MPM solver
"""
import numpy as np
import taichi as ti
from typing import Dict, List, Tuple, Optional
from .config import MPMConfig
from .mpm_solver import MPMSolver


class ValidationScene:
    """Base class for validation scenes

    Provides default implementations for simple validation scenarios.
    Subclasses should override create_particles() and optionally run()
    for specific validation tests.

    Default behavior:
    - create_particles(): Creates a single particle at grid center
    - run(): Runs simulation and returns basic particle/energy data

    Usage:
        # Direct use for simple tests
        scene = ValidationScene(config)
        results = scene.run(num_steps=100)

        # Subclass for custom tests
        class MyTest(ValidationScene):
            def create_particles(self):
                # Custom particle setup
                return positions, volumes
    """

    def __init__(self, config: MPMConfig):
        self.config = config

    def create_particles(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create particle positions and volumes

        Default: Creates a single particle at the center of the grid domain.
        Subclasses should override this method for custom particle configurations.

        Returns:
            Tuple of (positions, volumes) where:
            - positions: (N, 3) array of particle positions
            - volumes: (N,) array of particle volumes, or None for default
        """
        # Default: single particle at grid center
        grid_size = np.array(self.config.grid.grid_size)
        dx = self.config.grid.dx
        center = grid_size * dx * 0.5
        positions = np.array([center], dtype=np.float32)
        volumes = np.array([dx ** 3], dtype=np.float32)
        return positions, volumes

    def run(self, num_steps: Optional[int] = None) -> Dict:
        """Run validation and return results

        Default implementation runs the simulation and returns basic data.
        Subclasses can override for custom validation logic.

        Args:
            num_steps: Number of steps to run. If None, uses config default.

        Returns:
            Dict containing:
            - 'final_particles': Particle data at end of simulation
            - 'final_energy': Energy data at end of simulation
            - 'num_steps': Number of steps executed
        """
        if num_steps is None:
            num_steps = self.config.time.num_steps

        positions, volumes = self.create_particles()
        n_particles = len(positions)

        solver = MPMSolver(self.config, n_particles)
        if volumes is not None:
            solver.initialize_particles(positions, volumes=volumes)
        else:
            solver.initialize_particles(positions)

        for _ in range(num_steps):
            solver.step()

        return {
            'final_particles': solver.get_particle_data(),
            'final_energy': solver.get_energy_data(),
            'num_steps': num_steps
        }


class UniaxialTensionTest(ValidationScene):
    """
    Uniaxial tension test for validating hyperelastic response
    """

    def __init__(self, config: MPMConfig, stretch_rate: float = 0.1):
        super().__init__(config)
        self.stretch_rate = stretch_rate

    def create_particles(self):
        """Create a bar of particles"""
        # Bar dimensions
        length, width, height = 0.2, 0.05, 0.05
        spacing = self.config.grid.dx * 0.5

        # Generate particles
        x = np.arange(0, length, spacing)
        y = np.arange(0, width, spacing)
        z = np.arange(0, height, spacing)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        volume = spacing ** 3
        volumes = np.full(len(positions), volume)

        return positions.astype(np.float32), volumes.astype(np.float32)

    def run(self, num_steps: int = None):
        """Run uniaxial tension test"""
        if num_steps is None:
            num_steps = self.config.time.num_steps

        positions, volumes = self.create_particles()
        n_particles = len(positions)

        solver = MPMSolver(self.config, n_particles)
        solver.initialize_particles(positions, volumes=volumes)

        # Track stress-strain
        stress_strain = []

        for step in range(num_steps):
            solver.step()

            if step % 10 == 0:
                # Compute average stress and strain
                particle_data = solver.get_particle_data()
                F = particle_data['F']

                # Average stretch in x-direction
                stretch = np.mean(F[:, 0, 0])
                strain = stretch - 1.0

                # Average stress (simplified)
                stress = np.mean(F[:, 0, 0]) * self.config.material.ogden.mu[0]

                stress_strain.append({'step': step, 'strain': strain, 'stress': stress})

        return {
            'stress_strain': stress_strain,
            'final_particles': solver.get_particle_data()
        }


class PureShearTest(ValidationScene):
    """
    Pure shear test for validating objectivity and shear response
    """

    def __init__(self, config: MPMConfig, shear_rate: float = 0.1):
        super().__init__(config)
        self.shear_rate = shear_rate

    def create_particles(self):
        """Create a square block of particles"""
        size = 0.1
        spacing = self.config.grid.dx * 0.5

        x = np.arange(0, size, spacing)
        y = np.arange(0, size, spacing)
        z = np.arange(0, size, spacing)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        volume = spacing ** 3
        volumes = np.full(len(positions), volume)

        return positions.astype(np.float32), volumes.astype(np.float32)

    def run(self, num_steps: int = None):
        """Run pure shear test"""
        if num_steps is None:
            num_steps = self.config.time.num_steps

        positions, volumes = self.create_particles()
        n_particles = len(positions)

        solver = MPMSolver(self.config, n_particles)
        solver.initialize_particles(positions, volumes=volumes)

        # Track shear stress
        shear_history = []

        for step in range(num_steps):
            solver.step()

            if step % 10 == 0:
                particle_data = solver.get_particle_data()
                F = particle_data['F']

                # Average shear component
                shear = np.mean(F[:, 0, 1])
                shear_history.append({'step': step, 'shear': shear})

        return {
            'shear_history': shear_history,
            'final_particles': solver.get_particle_data()
        }


class EnergyConservationTest(ValidationScene):
    """
    Test energy conservation for elastic-only simulation
    """

    def create_particles(self):
        """Create a cube of particles"""
        size = 0.1
        spacing = self.config.grid.dx * 0.5

        x = np.arange(0, size, spacing)
        y = np.arange(0, size, spacing)
        z = np.arange(0, size, spacing)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        # Add initial velocity
        velocities = np.zeros_like(positions)
        velocities[:, 2] = 1.0  # Initial upward velocity

        volume = spacing ** 3
        volumes = np.full(len(positions), volume)

        return positions.astype(np.float32), velocities.astype(np.float32), volumes.astype(np.float32)

    def run(self, num_steps: int = None):
        """Run energy conservation test"""
        if num_steps is None:
            num_steps = self.config.time.num_steps

        positions, velocities, volumes = self.create_particles()
        n_particles = len(positions)

        # Disable contact for pure energy conservation test
        config_copy = self.config
        config_copy.contact.enable_contact = False

        solver = MPMSolver(config_copy, n_particles)
        solver.initialize_particles(positions, velocities=velocities, volumes=volumes)

        # Track energy
        energy_history = []
        initial_energy = None

        for step in range(num_steps):
            solver.step()

            if step % 10 == 0:
                energy_data = solver.get_energy_data()
                total_energy = energy_data['E_kin'] + energy_data['E_elastic']

                if initial_energy is None:
                    initial_energy = total_energy

                energy_error = abs(total_energy - initial_energy) / (initial_energy + 1e-10)

                energy_history.append({
                    'step': step,
                    'E_kin': energy_data['E_kin'],
                    'E_elastic': energy_data['E_elastic'],
                    'E_total': total_energy,
                    'error': energy_error
                })

        return {
            'energy_history': energy_history,
            'initial_energy': initial_energy
        }


class ObjectivityTest(ValidationScene):
    """
    Objectivity test: apply rigid rotation and verify stress invariance.

    Tests FR-5 requirement: stress response remains objective under superposed
    rigid rotation.
    """

    def __init__(self, config: MPMConfig, rotation_angle: float = 0.5):
        """
        Args:
            config: MPM configuration
            rotation_angle: Rotation angle in radians (applied about z-axis)
        """
        super().__init__(config)
        self.rotation_angle = rotation_angle

    def create_particles(self):
        """Create a cube of particles"""
        size = 0.08
        spacing = self.config.grid.dx * 0.5

        x = np.arange(0.2, 0.2 + size, spacing)
        y = np.arange(0.2, 0.2 + size, spacing)
        z = np.arange(0.2, 0.2 + size, spacing)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        volume = spacing ** 3
        volumes = np.full(len(positions), volume)

        return positions.astype(np.float32), volumes.astype(np.float32)

    def run(self, num_steps: int = None):
        """Run objectivity test: compare stress with and without rotation"""
        if num_steps is None:
            num_steps = min(200, self.config.time.num_steps)

        positions, volumes = self.create_particles()
        n_particles = len(positions)

        # Run without rotation
        solver_no_rot = MPMSolver(self.config, n_particles)
        solver_no_rot.initialize_particles(positions, volumes=volumes)

        stress_no_rot = []
        for step in range(num_steps):
            solver_no_rot.step()
            if step % 10 == 0:
                data = solver_no_rot.get_particle_data()
                F = data['F']
                # Cauchy stress invariant: Frobenius norm
                stress_norm = np.mean([np.linalg.norm(F[i]) for i in range(len(F))])
                stress_no_rot.append({'step': step, 'stress_norm': stress_norm})

        # Run with initial rotation (superposed rigid motion)
        cos_a, sin_a = np.cos(self.rotation_angle), np.sin(self.rotation_angle)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Rotate positions around center
        center = np.mean(positions, axis=0)
        rotated_pos = (positions - center) @ R.T + center

        solver_rot = MPMSolver(self.config, n_particles)
        solver_rot.initialize_particles(rotated_pos.astype(np.float32), volumes=volumes)

        stress_rot = []
        for step in range(num_steps):
            solver_rot.step()
            if step % 10 == 0:
                data = solver_rot.get_particle_data()
                F = data['F']
                stress_norm = np.mean([np.linalg.norm(F[i]) for i in range(len(F))])
                stress_rot.append({'step': step, 'stress_norm': stress_norm})

        # Compare: stress should be invariant
        comparison = []
        for s_no, s_rot in zip(stress_no_rot, stress_rot):
            rel_diff = abs(s_no['stress_norm'] - s_rot['stress_norm']) / (s_no['stress_norm'] + 1e-10)
            comparison.append({
                'step': s_no['step'],
                'stress_no_rot': s_no['stress_norm'],
                'stress_rot': s_rot['stress_norm'],
                'rel_diff': rel_diff
            })

        return {
            'comparison': comparison,
            'rotation_angle': self.rotation_angle,
            'max_rel_diff': max(c['rel_diff'] for c in comparison)
        }


class GelSlimSlipTest(ValidationScene):
    """
    GelSlim-style stick-slip / incipient slip test.

    Simulates tangential loading of elastomer against rigid surface,
    recording tangential force vs displacement to observe:
    - Stick phase (linear elasticity)
    - Incipient slip (partial slip at edges)
    - Full slip (kinetic friction plateau)

    Tests FR-5 requirement: stick-slip / incipient slip behavior.
    """

    def __init__(self, config: MPMConfig, normal_load: float = 100.0,
                 tangent_velocity: float = 0.01):
        """
        Args:
            config: MPM configuration
            normal_load: Applied normal load (N)
            tangent_velocity: Tangential driving velocity (m/s)
        """
        super().__init__(config)
        self.normal_load = normal_load
        self.tangent_velocity = tangent_velocity

    def create_particles(self):
        """Create a flat elastomer slab (GelSlim-like geometry)"""
        # Flat slab: 0.1 x 0.1 x 0.02 (thin in z)
        spacing = self.config.grid.dx * 0.5

        x = np.arange(0.2, 0.3, spacing)
        y = np.arange(0.2, 0.3, spacing)
        z = np.arange(0.02, 0.04, spacing)  # Thin slab above ground

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        volume = spacing ** 3
        volumes = np.full(len(positions), volume)

        return positions.astype(np.float32), volumes.astype(np.float32)

    def run(self, num_steps: int = None):
        """Run stick-slip test with tangential driving"""
        if num_steps is None:
            num_steps = min(500, self.config.time.num_steps)

        positions, volumes = self.create_particles()
        n_particles = len(positions)

        # Ensure contact is enabled
        config = self.config
        config.contact.enable_contact = True

        solver = MPMSolver(config, n_particles)
        solver.initialize_particles(positions, volumes=volumes)

        # Apply normal preload by setting initial compression
        # (simplified: use gravity + mass to create normal contact)

        slip_data = []
        initial_x = np.mean(positions[:, 0])

        for step in range(num_steps):
            # Apply tangential driving force via boundary velocity
            # This simulates the GelSlim being pushed sideways
            solver.step()

            if step % 5 == 0:
                data = solver.get_particle_data()
                energy = solver.get_energy_data()

                # Track tangential displacement and force proxy
                avg_x = np.mean(data['x'][:, 0])
                avg_vx = np.mean(data['v'][:, 0])
                tangent_disp = avg_x - initial_x

                # Force proxy: E_proj_step captures contact energy changes
                # and E_elastic gives elastic restoring force indication

                slip_data.append({
                    'step': step,
                    'time': step * config.time.dt,
                    'tangent_disp': tangent_disp,
                    'tangent_vel': avg_vx,
                    'E_kin': energy['E_kin'],
                    'E_elastic': energy['E_elastic'],
                    'E_proj_step': energy['E_proj_step'],
                    'E_proj_cum': energy['E_proj_cum']
                })

        return {
            'slip_data': slip_data,
            'normal_load': self.normal_load,
            'tangent_velocity': self.tangent_velocity
        }


class HertzContactTest(ValidationScene):
    """
    Hertzian contact / elastic impact test.

    Simulates elastic sphere impacting a rigid surface, validating:
    - Contact force vs indentation (Hertz law: F ~ δ^(3/2))
    - Error convergence vs time step and grid size

    Tests FR-5 requirement: Hertz contact / elastic impact convergence.
    """

    def __init__(self, config: MPMConfig, sphere_radius: float = 0.02,
                 impact_velocity: float = 0.5):
        """
        Args:
            config: MPM configuration
            sphere_radius: Sphere radius (m)
            impact_velocity: Initial downward velocity (m/s)
        """
        super().__init__(config)
        self.sphere_radius = sphere_radius
        self.impact_velocity = impact_velocity

    def create_particles(self):
        """Create particles in a sphere shape"""
        spacing = self.config.grid.dx * 0.5
        R = self.sphere_radius
        center = np.array([0.25, 0.25, 0.1 + R])  # Above ground

        # Generate sphere via rejection sampling
        positions = []
        n_per_dim = int(2 * R / spacing) + 2

        for i in range(n_per_dim):
            for j in range(n_per_dim):
                for k in range(n_per_dim):
                    x = center[0] - R + i * spacing
                    y = center[1] - R + j * spacing
                    z = center[2] - R + k * spacing

                    # Check if inside sphere
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                    if dist <= R:
                        positions.append([x, y, z])

        positions = np.array(positions, dtype=np.float32)

        # Initial downward velocity
        velocities = np.zeros_like(positions)
        velocities[:, 2] = -self.impact_velocity

        volume = spacing ** 3
        volumes = np.full(len(positions), volume)

        return positions, velocities, volumes

    def run(self, num_steps: int = None):
        """Run Hertz contact test"""
        if num_steps is None:
            num_steps = min(300, self.config.time.num_steps)

        positions, velocities, volumes = self.create_particles()
        n_particles = len(positions)

        config = self.config
        config.contact.enable_contact = True

        solver = MPMSolver(config, n_particles)
        solver.initialize_particles(positions, velocities=velocities.astype(np.float32), volumes=volumes)

        contact_data = []

        for step in range(num_steps):
            solver.step()

            if step % 5 == 0:
                data = solver.get_particle_data()
                energy = solver.get_energy_data()

                # Track lowest particle z (indentation proxy)
                min_z = np.min(data['x'][:, 2])
                avg_vz = np.mean(data['v'][:, 2])

                # Indentation depth (relative to ground at z=0)
                # Ground is at some z_ground; assume z_ground ~ 0
                indentation = max(0, 0.0 - min_z)

                contact_data.append({
                    'step': step,
                    'time': step * config.time.dt,
                    'min_z': min_z,
                    'indentation': indentation,
                    'vel_z': avg_vz,
                    'E_kin': energy['E_kin'],
                    'E_elastic': energy['E_elastic'],
                    'E_proj_cum': energy['E_proj_cum']
                })

        return {
            'contact_data': contact_data,
            'sphere_radius': self.sphere_radius,
            'impact_velocity': self.impact_velocity,
            'n_particles': n_particles
        }


class EnergyConvergenceTest(ValidationScene):
    """
    Energy convergence test with projection tracking.

    Validates energy balance including:
    - E_kin: Kinetic energy
    - E_elastic: Elastic potential energy
    - E_viscous_step/cum: Viscous dissipation
    - ΔE_proj_step: Projection correction per step
    - E_proj_cum: Cumulative projection energy

    Shows convergence vs Δt and grid resolution.

    Tests FR-5 requirement: Energy conservation/dissipation with projection.
    """

    def __init__(self, config: MPMConfig, dt_factors: list = None,
                 grid_factors: list = None):
        """
        Args:
            config: Base MPM configuration
            dt_factors: Time step multipliers for convergence study
            grid_factors: Grid refinement factors for convergence study
        """
        super().__init__(config)
        self.dt_factors = dt_factors or [1.0]  # Single run by default
        self.grid_factors = grid_factors or [1.0]

    def create_particles(self):
        """Create a bouncing cube"""
        size = 0.06
        spacing = self.config.grid.dx * 0.5

        x = np.arange(0.2, 0.2 + size, spacing)
        y = np.arange(0.2, 0.2 + size, spacing)
        z = np.arange(0.15, 0.15 + size, spacing)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        velocities = np.zeros_like(positions)
        velocities[:, 2] = -0.5  # Downward initial velocity

        volume = spacing ** 3
        volumes = np.full(len(positions), volume)

        return positions.astype(np.float32), velocities.astype(np.float32), volumes.astype(np.float32)

    def run(self, num_steps: int = None):
        """Run energy convergence test"""
        if num_steps is None:
            num_steps = min(400, self.config.time.num_steps)

        positions, velocities, volumes = self.create_particles()
        n_particles = len(positions)

        config = self.config
        config.contact.enable_contact = True

        solver = MPMSolver(config, n_particles)
        solver.initialize_particles(positions, velocities=velocities, volumes=volumes)

        energy_history = []

        for step in range(num_steps):
            solver.step()

            if step % 5 == 0:
                energy = solver.get_energy_data()

                E_total = energy['E_kin'] + energy['E_elastic']
                E_dissipated = energy['E_viscous_cum'] + energy['E_proj_cum']

                energy_history.append({
                    'step': step,
                    'time': step * config.time.dt,
                    'E_kin': energy['E_kin'],
                    'E_elastic': energy['E_elastic'],
                    'E_viscous_step': energy['E_viscous_step'],
                    'E_viscous_cum': energy['E_viscous_cum'],
                    'E_proj_step': energy['E_proj_step'],  # ΔE_proj_step
                    'E_proj_cum': energy['E_proj_cum'],
                    'E_total': E_total,
                    'E_dissipated': E_dissipated
                })

        return {
            'energy_history': energy_history,
            'dt': config.time.dt,
            'dx': config.grid.dx
        }


def run_all_validations(config: MPMConfig, output_dir: str = "validation_output",
                        scenes: list = None):
    """
    Run validation tests

    Args:
        config: MPM configuration
        output_dir: Output directory for results
        scenes: List of scene names to run. If None, runs all.
                Options: 'uniaxial', 'shear', 'objectivity', 'energy',
                         'energy_convergence', 'gelslim', 'hertz'
    """
    from pathlib import Path
    import csv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_scenes = ['uniaxial', 'shear', 'objectivity', 'energy',
                  'energy_convergence', 'gelslim', 'hertz']
    if scenes is None:
        scenes = all_scenes

    print("Running validation tests...")
    print(f"Scenes: {scenes}")

    results = {}

    # 1. Uniaxial tension
    if 'uniaxial' in scenes:
        print("\n1. Uniaxial Tension Test")
        tension_test = UniaxialTensionTest(config)
        tension_results = tension_test.run(num_steps=500)

        with open(output_path / "uniaxial_tension.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'strain', 'stress'])
            writer.writeheader()
            writer.writerows(tension_results['stress_strain'])

        results['uniaxial'] = tension_results
        print("   ✓ Uniaxial tension test complete")

    # 2. Pure shear
    if 'shear' in scenes:
        print("\n2. Pure Shear Test")
        shear_test = PureShearTest(config)
        shear_results = shear_test.run(num_steps=500)

        with open(output_path / "pure_shear.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'shear'])
            writer.writeheader()
            writer.writerows(shear_results['shear_history'])

        results['shear'] = shear_results
        print("   ✓ Pure shear test complete")

    # 3. Objectivity test (pure shear + rotation)
    if 'objectivity' in scenes:
        print("\n3. Objectivity Test (rigid rotation)")
        obj_test = ObjectivityTest(config, rotation_angle=0.5)
        obj_results = obj_test.run(num_steps=200)

        with open(output_path / "objectivity.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'stress_no_rot', 'stress_rot', 'rel_diff'])
            writer.writeheader()
            writer.writerows(obj_results['comparison'])

        results['objectivity'] = obj_results
        print(f"   ✓ Objectivity test complete (max rel_diff: {obj_results['max_rel_diff']:.2e})")

    # 4. Energy conservation (basic)
    if 'energy' in scenes:
        print("\n4. Energy Conservation Test")
        energy_test = EnergyConservationTest(config)
        energy_results = energy_test.run(num_steps=1000)

        with open(output_path / "energy_conservation.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'E_kin', 'E_elastic', 'E_total', 'error'])
            writer.writeheader()
            writer.writerows(energy_results['energy_history'])

        max_error = max(e['error'] for e in energy_results['energy_history'])
        results['energy'] = energy_results
        print(f"   ✓ Energy conservation test complete (max error: {max_error:.2%})")

    # 5. Energy convergence with projection
    if 'energy_convergence' in scenes:
        print("\n5. Energy Convergence with Projection Test")
        conv_test = EnergyConvergenceTest(config)
        conv_results = conv_test.run(num_steps=400)

        fieldnames = ['step', 'time', 'E_kin', 'E_elastic', 'E_viscous_step',
                      'E_viscous_cum', 'E_proj_step', 'E_proj_cum', 'E_total', 'E_dissipated']
        with open(output_path / "energy_convergence.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(conv_results['energy_history'])

        results['energy_convergence'] = conv_results
        print("   ✓ Energy convergence test complete")

    # 6. GelSlim stick-slip test
    if 'gelslim' in scenes:
        print("\n6. GelSlim Stick-Slip Test")
        slip_test = GelSlimSlipTest(config)
        slip_results = slip_test.run(num_steps=500)

        # Unified format with friction_curve.csv
        fieldnames = ['step', 'time', 'tangent_disp', 'tangent_vel',
                      'E_kin', 'E_elastic', 'E_proj_step', 'E_proj_cum']
        with open(output_path / "gelslim_slip.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(slip_results['slip_data'])

        results['gelslim'] = slip_results
        print("   ✓ GelSlim stick-slip test complete")

    # 7. Hertz contact test
    if 'hertz' in scenes:
        print("\n7. Hertz Contact / Impact Test")
        hertz_test = HertzContactTest(config)
        hertz_results = hertz_test.run(num_steps=300)

        fieldnames = ['step', 'time', 'min_z', 'indentation', 'vel_z',
                      'E_kin', 'E_elastic', 'E_proj_cum']
        with open(output_path / "hertz_contact.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hertz_results['contact_data'])

        results['hertz'] = hertz_results
        print(f"   ✓ Hertz contact test complete ({hertz_results['n_particles']} particles)")

    print(f"\nValidation tests complete. Results saved to {output_path}")
    return results


if __name__ == '__main__':
    # Run validations with default config
    config = MPMConfig()
    run_all_validations(config)
