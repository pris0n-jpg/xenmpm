"""
MPM vs FEM Visual Comparison Example

This script provides a side-by-side comparison of FEM and MPM simulations
for a block-on-sensor contact scenario.

Features:
- Runs both FEM (pre-computed) and MPM (real-time) simulations
- Visualizes deformation in the existing render pipeline
- Generates comparison curves (displacement vs force)

Usage:
    python example/mpm_fem_compare.py --mode fem       # FEM only
    python example/mpm_fem_compare.py --mode mpm       # MPM only
    python example/mpm_fem_compare.py --mode both      # Side-by-side

Requirements:
    - xengym conda environment with Taichi installed
    - FEM data file (default: assets/data/fem_data_gel_2035.npz)
"""
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import sys

# Add project root to path for standalone execution
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project imports
try:
    from xengym import PROJ_DIR
except ImportError:
    PROJ_DIR = _PROJECT_ROOT / "xengym"

# Attempt optional imports
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, curve plotting disabled")

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False
    print("Warning: Taichi not available, MPM mode disabled")

# Attempt optional render imports (existing ezgl/xengym render pipeline)
try:
    from xensesdk.ezgl import tb, Matrix4x4
    from xensesdk.ezgl.items.scene import Scene
    from xensesdk.ezgl.items import (
        GLBoxItem,
        GLScatterPlotItem,
        GLAxisItem,
        GLGridItem,
        DepthCamera,
        PointLight,
    )
    from xengym.render import VecTouchSim
    HAS_EZGL = True
    _EZGL_IMPORT_ERROR = None
except Exception as e:
    HAS_EZGL = False
    _EZGL_IMPORT_ERROR = e
    print(f"Warning: ezgl/xengym render not available, visualization disabled ({e})")


# ==============================================================================
# Scene Parameters (Task 2.1)
# ==============================================================================
# These parameters define the shared block-on-sensor scenario for both FEM and MPM

SCENE_PARAMS = {
    # Block geometry (approximate match to FEM gel layer)
    'block_size_mm': (19.4, 30.8, 5.0),  # width, height, depth in mm
    'block_center_mm': (0.0, 0.0, 2.5),  # initial center position

    # Material properties (Ogden soft gel approximation)
    'density': 1000.0,          # kg/m³
    'ogden_mu': [2500.0],       # Pa (shear modulus)
    'ogden_alpha': [2.0],       # Ogden exponent
    'ogden_kappa': 25000.0,     # Pa (bulk modulus)

    # Contact/sliding parameters
    'indentation_mm': 1.0,      # press depth
    'slide_distance_mm': 5.0,   # tangential travel distance (for visual comparison)
    'slide_steps': 200,         # steps used to complete the slide
    'slide_velocity_mm_s': 5.0, # tangential sliding speed

    # Simulation parameters
    'mpm_dt': 1e-4,             # MPM time step
    'mpm_steps': 200,           # simulation steps
    'mpm_grid_dx': 0.5,         # grid spacing in mm

    # Grid padding (cells) to keep particles away from sticky boundaries.
    # The core solver zeros velocity when I[d] < 3 or I[d] >= grid_size[d] - 3.
    # Use >= 6 cells of padding for x/y to avoid sampling boundary nodes.
    'mpm_grid_padding_cells_xy': 6,
    'mpm_grid_padding_cells_z_bottom': 4,
    'mpm_grid_padding_cells_z_top': 10,

    # Recording interval for visualization (store every N steps).
    'mpm_record_interval': 10,

    # Emulate gel bonded to sensor by fixing a thin bottom particle layer.
    'mpm_fix_base': True,
    'mpm_fix_base_thickness_cells': 2,

    # MPM contact/gravity tuning for mm-scale demo stability
    # These values intentionally lower the default penalty stiffness to
    # avoid particle ejection in small-scale explicit runs.
    'mpm_enable_contact': True,
    'mpm_contact_stiffness_normal': 1e3,   # N/m
    'mpm_contact_stiffness_tangent': 1e2,  # N/m
    'mpm_gravity': (0.0, 0.0, 0.0),        # m/s^2; set to zero for quasi-static press

    # Simple spherical indenter to create visible, localized indentation
    'indenter_radius_mm': 4.0,
    'indenter_gap_mm': 0.5,        # initial clearance above block top
    'indenter_press_mm': 1.0,      # target indentation depth
    'indenter_press_steps': 120,   # steps to reach target depth
}

if HAS_TAICHI:
    @ti.data_oriented
    class _MPMBaseFixer:
        """
        Simple particle constraint helper for visualization demos.

        This clamps a thin bottom layer of particles to their initial positions,
        emulating a gel block bonded to the sensor base. It reduces rigid-body
        drift so the indentation footprint is easier to compare with FEM.
        """

        def __init__(self, x_field, v_field, x0_np: np.ndarray, z_max: float):
            self.n_particles = int(x0_np.shape[0])
            self.x = x_field
            self.v = v_field
            self.x0 = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
            self.x0.from_numpy(x0_np.astype(np.float32))
            self.z_max = ti.field(dtype=ti.f32, shape=())
            self.z_max[None] = float(z_max)

        @ti.kernel
        def apply(self):
            z_max = self.z_max[None]
            for p in range(self.n_particles):
                if self.x0[p][2] <= z_max:
                    self.x[p] = self.x0[p]
                    self.v[p] = ti.Vector([0.0, 0.0, 0.0])


# ==============================================================================
# FEM Data Loader and Adapter (Task 3.2)
# ==============================================================================
class FEMDataAdapter:
    """Adapter for loading and processing FEM simulation data"""

    def __init__(self, fem_file: str):
        """
        Load FEM data from NPZ file

        Args:
            fem_file: Path to FEM NPZ file
        """
        self.fem_file = Path(fem_file)
        if not self.fem_file.exists():
            raise FileNotFoundError(f"FEM file not found: {fem_file}")

        # Load FEM data
        data = np.load(str(fem_file), allow_pickle=True)
        self.nodes = data['node']  # shape: (n_nodes, 3)
        self.elements = data['elements']  # shape: (n_elements, 8) hexahedral
        self.top_nodes = data['top_nodes']  # indices of top surface nodes
        self.top_indices = data['top_indices']  # surface quad indices

        # Extract gel dimensions from node positions
        self.gel_bounds = {
            'x_min': self.nodes[:, 0].min(),
            'x_max': self.nodes[:, 0].max(),
            'y_min': self.nodes[:, 1].min(),
            'y_max': self.nodes[:, 1].max(),
            'z_min': self.nodes[:, 2].min(),
            'z_max': self.nodes[:, 2].max(),
        }

        # Initial (reference) positions
        self.ref_positions = self.nodes.copy()
        self.current_displacements = np.zeros_like(self.nodes)

        print(f"FEM data loaded: {len(self.nodes)} nodes, {len(self.elements)} elements")
        print(f"  Gel bounds: X[{self.gel_bounds['x_min']:.2f}, {self.gel_bounds['x_max']:.2f}] mm")
        print(f"              Y[{self.gel_bounds['y_min']:.2f}, {self.gel_bounds['y_max']:.2f}] mm")
        print(f"              Z[{self.gel_bounds['z_min']:.2f}, {self.gel_bounds['z_max']:.2f}] mm")

    def get_top_surface_positions(self) -> np.ndarray:
        """Get current positions of top surface nodes"""
        return self.ref_positions[self.top_nodes] + self.current_displacements[self.top_nodes]

    def apply_indentation(
        self,
        depth_mm: float,
        contact_region: Optional[np.ndarray] = None,
        center_xy_mm: Optional[Tuple[float, float]] = None,
    ):
        """
        Apply a simple indentation displacement to the top surface

        Args:
            depth_mm: Indentation depth in mm (positive = pressing down)
            contact_region: Optional mask for contact nodes (default: center region)
            center_xy_mm: Optional (x, y) indentation center in mm (default: gel center)
        """
        if contact_region is None:
            # Default: center circular region.
            #
            # When an indenter radius is configured, approximate the contact patch
            # of a spherical indenter to make FEM/MPM indentation footprints
            # visually comparable.
            top_pos = self.ref_positions[self.top_nodes]
            if center_xy_mm is None:
                center_x = (self.gel_bounds['x_min'] + self.gel_bounds['x_max']) / 2
                center_y = (self.gel_bounds['y_min'] + self.gel_bounds['y_max']) / 2
            else:
                center_x = float(center_xy_mm[0])
                center_y = float(center_xy_mm[1])
            radius = None
            indenter_radius = SCENE_PARAMS.get("indenter_radius_mm", None)
            if indenter_radius is not None:
                try:
                    R = float(indenter_radius)
                    d = float(depth_mm)
                    # Sphere-plane intersection (mm): a^2 = 2 R d - d^2
                    a2 = max(0.0, 2.0 * R * d - d * d)
                    radius = float(np.sqrt(a2))
                except Exception:
                    radius = None

            if radius is None or radius <= 0.0:
                radius = min(
                    self.gel_bounds['x_max'] - self.gel_bounds['x_min'],
                    self.gel_bounds['y_max'] - self.gel_bounds['y_min']
                ) / 4

            dist = np.sqrt((top_pos[:, 0] - center_x)**2 + (top_pos[:, 1] - center_y)**2)
            contact_region = dist < radius

        # Apply indentation to contact region
        self.current_displacements[self.top_nodes[contact_region], 2] = -depth_mm

    def get_average_displacement(self) -> Tuple[float, float, float]:
        """Get average displacement of top surface"""
        top_disp = self.current_displacements[self.top_nodes]
        return tuple(top_disp.mean(axis=0))

    def get_contact_force_estimate(self) -> float:
        """
        Estimate contact force from displacement (simplified linear model)

        Returns:
            Estimated force in N (simplified)
        """
        # Simplified: F = k * delta, where k is estimated from material properties
        avg_disp = abs(self.current_displacements[self.top_nodes, 2].mean())
        area = (self.gel_bounds['x_max'] - self.gel_bounds['x_min']) * \
               (self.gel_bounds['y_max'] - self.gel_bounds['y_min']) * 1e-6  # m²
        E = SCENE_PARAMS['ogden_mu'][0] * 3  # Approximate Young's modulus
        thickness = self.gel_bounds['z_max'] - self.gel_bounds['z_min']
        k = E * area / (thickness * 1e-3)  # N/mm
        return k * avg_disp


# ==============================================================================
# MPM Simulation Adapter (Task 2.2, 3.3, 3.4)
# ==============================================================================
class MPMAdapter:
    """Adapter for running MPM simulation and extracting comparable data"""

    def __init__(self, scene_params: Dict):
        """
        Initialize MPM simulation for block-on-sensor scenario

        Args:
            scene_params: Scene configuration parameters
        """
        if not HAS_TAICHI:
            raise RuntimeError("Taichi not available, cannot run MPM simulation")

        self.params = scene_params
        self.solver = None
        self.positions_history: List[np.ndarray] = []
        self.forces_history: List[float] = []
        self.indenter_depth_history_mm: List[float] = []
        self.indenter_center_history_mm: List[np.ndarray] = []
        self._base_fixer = None

        # Initialize Taichi
        ti.init(arch=ti.cpu)  # Use CPU for compatibility

        self._setup_solver()

    def _setup_solver(self):
        """Set up MPM solver with matching parameters"""
        from xengym.mpm import (
            MPMConfig, GridConfig, TimeConfig, OgdenConfig,
            MaterialConfig, ContactConfig, OutputConfig, MPMSolver
        )
        from xengym.mpm.config import SDFConfig

        # Convert mm to m for MPM solver
        block_size_m = [s * 1e-3 for s in self.params['block_size_mm']]
        grid_dx = self.params['mpm_grid_dx'] * 1e-3

        # Grid size: keep a safety padding away from sticky boundary nodes.
        # NOTE: the solver applies boundary conditions when I[d] < 3 or
        # I[d] >= grid_size[d] - 3, so we must ensure particles never sample
        # those nodes. Compute grid cells based on block size + padding.
        pad_xy = int(self.params.get("mpm_grid_padding_cells_xy", 6))
        pad_z_bottom = int(self.params.get("mpm_grid_padding_cells_z_bottom", 4))
        pad_z_top = int(self.params.get("mpm_grid_padding_cells_z_top", 10))

        # Create particles first so we can place indenter relative to block
        n_particles = self._create_block_particles(
            grid_dx=grid_dx,
            pad_xy_cells=pad_xy,
            pad_z_bottom_cells=pad_z_bottom,
        )
        block_min = self.initial_positions.min(axis=0)
        block_max = self.initial_positions.max(axis=0)
        block_center = (block_min + block_max) / 2.0

        # Define explicit obstacles: ground plane + spherical indenter
        indenter_radius = float(self.params.get("indenter_radius_mm", 4.0)) * 1e-3
        indenter_gap = float(self.params.get("indenter_gap_mm", 0.5)) * 1e-3
        press_depth = float(self.params.get("indenter_press_mm", 0.0)) * 1e-3

        block_cells = [
            int(np.ceil(block_size_m[0] / grid_dx)),
            int(np.ceil(block_size_m[1] / grid_dx)),
            int(np.ceil(block_size_m[2] / grid_dx)),
        ]
        # Ensure enough headroom above the top surface for contact response.
        extra_top_cells = int(np.ceil((indenter_gap + press_depth) / max(grid_dx, 1e-12)))
        pad_z_top = max(pad_z_top, extra_top_cells + 6)
        grid_extent = [
            block_cells[0] + 2 * pad_xy + 1,
            block_cells[1] + 2 * pad_xy + 1,
            block_cells[2] + pad_z_bottom + pad_z_top + 1,
        ]

        indenter_center0 = (
            float(block_center[0]),
            float(block_center[1]),
            float(block_max[2] + indenter_radius + indenter_gap),
        )
        obstacles = [
            SDFConfig(sdf_type="plane", center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
            SDFConfig(sdf_type="sphere", center=indenter_center0, half_extents=(indenter_radius, 0.0, 0.0)),
        ]

        config = MPMConfig(
            grid=GridConfig(
                grid_size=grid_extent,
                dx=grid_dx
            ),
            time=TimeConfig(
                dt=self.params['mpm_dt'],
                num_steps=self.params['mpm_steps']
            ),
            material=MaterialConfig(
                density=self.params['density'],
                ogden=OgdenConfig(
                    mu=self.params['ogden_mu'],
                    alpha=self.params['ogden_alpha'],
                    kappa=self.params['ogden_kappa']
                ),
                maxwell_branches=[],
                enable_bulk_viscosity=False
            ),
            contact=ContactConfig(
                enable_contact=bool(self.params.get('mpm_enable_contact', True)),
                contact_stiffness_normal=float(self.params.get('mpm_contact_stiffness_normal', 1e5)),
                contact_stiffness_tangent=float(self.params.get('mpm_contact_stiffness_tangent', 1e4)),
                obstacles=obstacles,
            ),
            output=OutputConfig()
        )

        self.solver = MPMSolver(config, n_particles)

        # Override gravity if requested
        gravity_vec = self.params.get('mpm_gravity', (0.0, 0.0, -9.81))
        try:
            self.solver.gravity = ti.Vector(list(gravity_vec))
        except Exception:
            pass

        # Optional: fix a bottom layer of particles to emulate gel bonded to sensor.
        # This reduces rigid drift and prevents "floating" blocks when gravity is disabled.
        if bool(self.params.get("mpm_fix_base", False)):
            thickness_cells = int(self.params.get("mpm_fix_base_thickness_cells", 2))
            thickness_cells = max(1, thickness_cells)
            bottom_z = float(block_min[2])
            z_max = bottom_z + float(thickness_cells) * float(grid_dx)
            try:
                self._base_fixer = _MPMBaseFixer(
                    self.solver.fields.x,
                    self.solver.fields.v,
                    self.initial_positions,
                    z_max,
                )
            except Exception:
                self._base_fixer = None

        # Cache indenter setup for animation (obstacle index 1)
        self._block_center = block_center
        self._block_max_z = float(block_max[2])
        self._indenter_radius = indenter_radius
        self._indenter_center0 = np.array(indenter_center0, dtype=np.float32)

        print(f"MPM solver initialized: {n_particles} particles")
        print(f"  Grid: {grid_extent}, dx={grid_dx*1000:.2f}mm")
        print(
            f"  Padding cells: xy={pad_xy}, z_bottom={pad_z_bottom}, z_top={pad_z_top}"
        )

    def _create_block_particles(self, grid_dx: float, pad_xy_cells: int, pad_z_bottom_cells: int) -> int:
        """
        Create particles filling the block volume

        Returns:
            Number of particles created
        """
        # Particle spacing (2 particles per grid cell per dimension)
        spacing = grid_dx / 2

        block_size_m = [s * 1e-3 for s in self.params['block_size_mm']]

        # Generate particle positions
        nx = int(np.ceil(block_size_m[0] / spacing))
        ny = int(np.ceil(block_size_m[1] / spacing))
        nz = int(np.ceil(block_size_m[2] / spacing))

        x = np.linspace(-block_size_m[0]/2, block_size_m[0]/2, nx)
        y = np.linspace(-block_size_m[1]/2, block_size_m[1]/2, ny)
        z = np.linspace(0, block_size_m[2], nz)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

        # Shift block inside the positive grid domain with explicit padding.
        # The solver uses a fixed grid origin at (0,0,0); particles starting at
        # negative coordinates fall outside and may become unstable.
        target_min = np.array(
            [pad_xy_cells * grid_dx, pad_xy_cells * grid_dx, pad_z_bottom_cells * grid_dx],
            dtype=np.float32,
        )
        min_pos = positions.min(axis=0)
        positions += (target_min - min_pos)

        self.initial_positions = positions.copy()
        self.n_particles = len(positions)

        return self.n_particles

    def run_simulation(self, num_steps: Optional[int] = None) -> Dict:
        """
        Run MPM simulation

        Args:
            num_steps: Number of steps to run (default from params)

        Returns:
            Dict with simulation results
        """
        if num_steps is None:
            num_steps = self.params['mpm_steps']

        # Initialize particles (start from rest; indenter drives deformation)
        velocities = np.zeros_like(self.initial_positions, dtype=np.float32)
        self.solver.initialize_particles(self.initial_positions, velocities)

        # Run simulation and record history
        self.positions_history = []
        self.forces_history = []
        self.indenter_depth_history_mm = []
        self.indenter_center_history_mm = []

        print(f"Running MPM simulation for {num_steps} steps...")
        start_time = time.time()

        press_steps = int(self.params.get("indenter_press_steps", 0))
        press_depth = float(self.params.get("indenter_press_mm", 0.0)) * 1e-3
        slide_steps = int(self.params.get("slide_steps", 0))
        slide_distance = float(self.params.get("slide_distance_mm", 0.0)) * 1e-3

        record_interval = int(self.params.get("mpm_record_interval", 20))
        record_interval = max(1, record_interval)

        for step in range(num_steps):
            # Indenter schedule:
            # - press: move down to target depth
            # - slide: keep depth and move tangentially to create visible shear footprint
            depth_m = 0.0
            slide_dx = 0.0
            if press_steps > 0 and step <= press_steps:
                t_press = step / max(press_steps, 1)
                depth_m = press_depth * t_press
            else:
                depth_m = press_depth
                if slide_steps > 0 and step <= press_steps + slide_steps:
                    t_slide = (step - press_steps) / max(slide_steps, 1)
                    slide_dx = slide_distance * t_slide
                elif slide_steps > 0 and step > press_steps + slide_steps:
                    slide_dx = slide_distance

            if self.solver is not None:
                new_center = ti.Vector([
                    float(self._indenter_center0[0] + slide_dx),
                    float(self._indenter_center0[1]),
                    float(self._indenter_center0[2] - depth_m),
                ])
                try:
                    self.solver.obstacle_centers[1] = new_center
                except Exception:
                    pass

            self.solver.step()

            if self._base_fixer is not None:
                try:
                    self._base_fixer.apply()
                except Exception:
                    self._base_fixer = None

            if step % record_interval == 0:
                pos = self.solver.get_particle_data()['x']
                self.positions_history.append(pos.copy())

                # Estimate force from stress
                energy = self.solver.get_energy_data()
                self.forces_history.append(energy.get('E_elastic', 0.0))

                # Cache indentation depth schedule for visualization sync (mm)
                self.indenter_depth_history_mm.append(float(depth_m * 1000.0))
                self.indenter_center_history_mm.append(
                    np.array(
                        [
                            float(self._indenter_center0[0] + slide_dx),
                            float(self._indenter_center0[1]),
                            float(self._indenter_center0[2] - depth_m),
                        ],
                        dtype=np.float32,
                    )
                    * 1000.0
                )

        elapsed = time.time() - start_time
        print(f"MPM simulation complete in {elapsed:.2f}s")

        return {
            'positions_history': self.positions_history,
            'forces_history': self.forces_history,
            'indenter_depth_history_mm': self.indenter_depth_history_mm,
            'final_positions': self.solver.get_particle_data()['x'],
        }

    def get_top_surface_positions(self) -> np.ndarray:
        """Get positions of particles near top surface"""
        if self.solver is None:
            return np.array([])

        pos = self.solver.get_particle_data()['x']
        z_max = pos[:, 2].max()
        top_mask = pos[:, 2] > z_max - self.params['mpm_grid_dx'] * 1e-3
        return pos[top_mask]

    def get_average_displacement(self) -> Tuple[float, float, float]:
        """Get average displacement of all particles"""
        if self.solver is None or len(self.positions_history) == 0:
            return (0.0, 0.0, 0.0)

        current_pos = self.solver.get_particle_data()['x']
        displacement = current_pos - self.initial_positions
        return tuple(displacement.mean(axis=0) * 1000)  # Convert to mm


# ==============================================================================
# Comparison and Visualization (Task 4.1-4.3)
# ==============================================================================
class ComparisonEngine:
    """Engine for comparing FEM and MPM results"""

    def __init__(self, fem_adapter: Optional[FEMDataAdapter] = None,
                 mpm_adapter: Optional[MPMAdapter] = None):
        self.fem = fem_adapter
        self.mpm = mpm_adapter
        self.results: Dict = {}

    def run_comparison(self, indentation_depths: List[float]) -> Dict:
        """
        Run comparison across multiple indentation depths

        Args:
            indentation_depths: List of indentation depths in mm

        Returns:
            Dict with comparison data
        """
        fem_displacements = []
        fem_forces = []
        mpm_displacements = []
        mpm_forces = []

        for depth in indentation_depths:
            print(f"\n--- Indentation depth: {depth:.2f} mm ---")

            if self.fem is not None:
                self.fem.apply_indentation(depth)
                disp = self.fem.get_average_displacement()
                force = self.fem.get_contact_force_estimate()
                fem_displacements.append(disp[2])  # Z displacement
                fem_forces.append(force)
                print(f"  FEM: disp_z={disp[2]:.4f}mm, force={force:.2f}N")

            if self.mpm is not None:
                # For MPM, we'd need to re-run with different initial conditions
                # Simplified: use elastic energy as force proxy
                disp = self.mpm.get_average_displacement()
                force = self.mpm.forces_history[-1] if self.mpm.forces_history else 0.0
                mpm_displacements.append(disp[2])
                mpm_forces.append(force)
                print(f"  MPM: disp_z={disp[2]:.4f}mm, energy={force:.4f}J")

        self.results = {
            'indentation_depths': indentation_depths,
            'fem_displacements': fem_displacements,
            'fem_forces': fem_forces,
            'mpm_displacements': mpm_displacements,
            'mpm_forces': mpm_forces,
        }

        return self.results

    def plot_comparison(self, output_path: str = None):
        """
        Generate comparison plot

        Args:
            output_path: Path to save the plot (default: output/mpm_fem_compare.png)
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping plot")
            return

        if not self.results:
            print("No comparison results to plot")
            return

        if output_path is None:
            output_path = str(PROJ_DIR / "output" / "mpm_fem_compare.png")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        depths = self.results['indentation_depths']

        # Plot 1: Displacement vs Indentation
        ax1 = axes[0]
        if self.results['fem_displacements']:
            ax1.plot(depths, self.results['fem_displacements'], 'b-o', label='FEM', linewidth=2)
        if self.results['mpm_displacements']:
            ax1.plot(depths, self.results['mpm_displacements'], 'r-s', label='MPM', linewidth=2)
        ax1.set_xlabel('Indentation Depth (mm)')
        ax1.set_ylabel('Average Z Displacement (mm)')
        ax1.set_title('Displacement Response')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Force/Energy vs Indentation
        ax2 = axes[1]
        if self.results['fem_forces']:
            ax2.plot(depths, self.results['fem_forces'], 'b-o', label='FEM (Est. Force)', linewidth=2)
        if self.results['mpm_forces']:
            # Normalize MPM energy for comparison
            mpm_normalized = np.array(self.results['mpm_forces'])
            if mpm_normalized.max() > 0:
                mpm_normalized = mpm_normalized / mpm_normalized.max() * max(self.results['fem_forces']) if self.results['fem_forces'] else mpm_normalized
            ax2.plot(depths, mpm_normalized, 'r-s', label='MPM (Elastic Energy, normalized)', linewidth=2)
        ax2.set_xlabel('Indentation Depth (mm)')
        ax2.set_ylabel('Force (N) / Normalized Energy')
        ax2.set_title('Force/Energy Response')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\nComparison plot saved to: {output_path}")
        plt.close()


# ==============================================================================
# 3D Visualization using existing ezgl/render pipeline (Spec Requirement)
# ==============================================================================
if HAS_EZGL:
    class CompareScene(Scene):
        """
        Lightweight 3D viewer to compare FEM top-surface nodes vs MPM particles.

        This intentionally reuses the existing ezgl Scene/Item system used elsewhere
        in the project (demo_main / demo_simple_sensor), without introducing a new UI.
        """

        def __init__(
            self,
            fem_adapter: Optional[FEMDataAdapter],
            mpm_adapter: Optional[MPMAdapter],
            mode: str,
            fem_depths: List[float],
        ):
            super().__init__(win_width=1000, win_height=650, visible=True, title="MPM vs FEM Compare")

            self.fem = fem_adapter
            self.mpm = mpm_adapter
            self.mode = mode
            self.fem_depths = fem_depths or [SCENE_PARAMS.get("indentation_mm", 1.0)]

            # Offsets to show side-by-side when mode == "both"
            self.fem_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.mpm_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if mode == "both":
                self.fem_offset[0] = -25.0
                self.mpm_offset[0] = 25.0

            # Basic camera in mm units
            self.cameraLookAt([0, -80, 55], [0, 15, -1.5], [0, 0, 1])

            # Axes and grid for orientation
            self.axis = GLAxisItem(size=(20, 20, 20), tip_size=2.5)
            self.grid = GLGridItem(size=(10, 10), spacing=(5, 5), lineWidth=1).rotate(90, 1, 0, 0)

            # Playback/synchronization:
            # - FEM is a quasi-static indentation sequence (no true time series).
            # - MPM is a time series (positions_history).
            # When both are enabled, we sync FEM indentation depth to the MPM
            # indenter schedule (if available) and advance both at the same pace.
            self._sync_fem_to_mpm = (
                self.mode == "both"
                and self.fem is not None
                and self.mpm is not None
                and bool(getattr(self.mpm, "indenter_depth_history_mm", None))
            )

            # Align MPM cloud into FEM coordinate frame (mm)
            self._mpm_align_shift_mm = np.zeros(3, dtype=np.float32)
            if self.fem is not None and self.mpm is not None:
                fem_bounds = self.fem.gel_bounds
                fem_x_center = (fem_bounds["x_min"] + fem_bounds["x_max"]) / 2
                fem_y_center = (fem_bounds["y_min"] + fem_bounds["y_max"]) / 2
                fem_z_top = fem_bounds["z_max"]
                try:
                    mpm_init_mm = self.mpm.initial_positions * 1000.0
                    mpm_center = mpm_init_mm.mean(axis=0)
                    mpm_z_top = mpm_init_mm[:, 2].max()
                    self._mpm_align_shift_mm = np.array([
                        fem_x_center - mpm_center[0],
                        fem_y_center - mpm_center[1],
                        fem_z_top - mpm_z_top,
                    ], dtype=np.float32)
                except Exception:
                    self._mpm_align_shift_mm = np.zeros(3, dtype=np.float32)

            # Downsample MPM particles for visualization if needed
            self._mpm_sample_idx = None
            # Cache initial MPM positions in mm for depth-based coloring / surface selection
            self._mpm_initial_mm = None
            self._mpm_top_mask_full = None
            if self.mpm is not None:
                try:
                    self._mpm_initial_mm = self.mpm.initial_positions * 1000.0
                    # Select initial top-surface particles (within ~1 cell from top)
                    z_max_init = float(self._mpm_initial_mm[:, 2].max())
                    dz = float(SCENE_PARAMS.get("mpm_grid_dx", 0.5))  # mm
                    self._mpm_top_mask_full = self._mpm_initial_mm[:, 2] >= (z_max_init - dz * 1.2)
                except Exception:
                    self._mpm_initial_mm = None
                    self._mpm_top_mask_full = None

            if self.mpm is not None and self.mpm.positions_history:
                n = self.mpm.positions_history[0].shape[0]
                if self._mpm_top_mask_full is not None and self._mpm_top_mask_full.any():
                    top_idx = np.where(self._mpm_top_mask_full)[0]
                    sample_n = min(50000, top_idx.shape[0])
                    self._mpm_sample_idx = np.random.choice(top_idx, size=sample_n, replace=False)
                else:
                    sample_n = min(50000, n)
                    self._mpm_sample_idx = np.random.choice(n, size=sample_n, replace=False)

            # Stable color scale to reduce "flicker" caused by per-frame re-normalization.
            self._mpm_color_scale_mm = float(SCENE_PARAMS.get("indenter_press_mm", 1.0))
            if self.mpm is not None and self.mpm.positions_history and self._mpm_initial_mm is not None:
                try:
                    last_pos_mm_full = self.mpm.positions_history[-1] * 1000.0
                    if self._mpm_sample_idx is not None:
                        last_pos_mm = last_pos_mm_full[self._mpm_sample_idx]
                        init_mm = self._mpm_initial_mm[self._mpm_sample_idx]
                    else:
                        last_pos_mm = last_pos_mm_full
                        init_mm = self._mpm_initial_mm

                    depth = np.clip(init_mm[:, 2] - last_pos_mm[:, 2], 0.0, None)
                    baseline = float(np.percentile(depth, 10))
                    depth = np.clip(depth - baseline, 0.0, None)
                    scale = float(np.percentile(depth, 95))
                    if scale > 1e-6:
                        self._mpm_color_scale_mm = scale
                except Exception:
                    pass

            # Animation parameters (visual frame rate is managed in run_visualization)
            self._fem_hold_frames = 30  # hold each indentation level ~1s at 30fps

            # Control playback rate: by default, try to match the FEM cycle speed.
            self._mpm_hold_frames = 1
            if self.mpm is not None and self.mpm.positions_history:
                if self.fem is not None:
                    target_cycle_frames = max(1, len(self.fem_depths) * self._fem_hold_frames)
                    self._mpm_hold_frames = max(
                        1,
                        int(round(target_cycle_frames / max(len(self.mpm.positions_history), 1))),
                    )
                else:
                    self._mpm_hold_frames = 1

            self._init_sensor_boxes()
            self._init_point_clouds()

            # Animation state
            self._frame = 0
            self._fem_depth_idx = 0
            self._mpm_frame_idx = 0

        def _init_sensor_boxes(self):
            """Add a simple gel volume box based on FEM bounds (mm)."""
            if self.fem is None:
                return

            bounds = self.fem.gel_bounds
            width = bounds["x_max"] - bounds["x_min"]
            height = bounds["y_max"] - bounds["y_min"]
            thickness = bounds["z_max"] - bounds["z_min"]

            center = np.array([
                (bounds["x_min"] + bounds["x_max"]) / 2,
                (bounds["y_min"] + bounds["y_max"]) / 2,
                (bounds["z_min"] + bounds["z_max"]) / 2,
            ], dtype=np.float32)

            # FEM-side sensor box
            if self.mode in ("fem", "both"):
                self.fem_sensor_box = GLBoxItem(
                    size=(width, height, thickness),
                    color=(0.85, 0.85, 0.9),
                    glOptions="translucent",
                ).translate(*(center + self.fem_offset))

            # MPM-side sensor box
            if self.mode in ("mpm", "both"):
                self.mpm_sensor_box = GLBoxItem(
                    size=(width, height, thickness),
                    color=(0.9, 0.85, 0.85),
                    glOptions="translucent",
                ).translate(*(center + self.mpm_offset))

        def _compute_mpm_depth_colors(
            self,
            pos_mm: np.ndarray,
            sample_idx: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """
            Compute per-particle RGBA colors based on indentation depth.

            Depth is measured as positive downward displacement relative to the
            initial configuration. Colors map shallow->deep as blue->red.
            """
            if pos_mm.size == 0:
                return np.zeros((0, 4), dtype=np.float32)

            if self._mpm_initial_mm is None:
                # Fallback to constant red if initial state is unavailable
                return np.tile(np.array([[0.9, 0.2, 0.2, 0.9]], dtype=np.float32), (pos_mm.shape[0], 1))

            init_mm = self._mpm_initial_mm
            if sample_idx is not None:
                init_mm = init_mm[sample_idx]

            depth = np.clip(init_mm[:, 2] - pos_mm[:, 2], 0.0, None)  # mm, positive = indentation
            if depth.size == 0:
                return np.tile(np.array([[0.9, 0.2, 0.2, 0.9]], dtype=np.float32), (pos_mm.shape[0], 1))

            # Remove global rigid translation so local indentation pops visually.
            # Using a low percentile as baseline is robust to outliers.
            baseline = float(np.percentile(depth, 10))
            depth = np.clip(depth - baseline, 0.0, None)

            depth_max = float(getattr(self, "_mpm_color_scale_mm", 0.0)) or float(np.percentile(depth, 95))
            depth_max = max(depth_max, 1e-6)
            t = np.clip(depth / depth_max, 0.0, 1.0)

            r = t
            g = 0.2 * (1.0 - t)
            b = 1.0 - t
            a = np.full_like(t, 0.9)
            return np.stack([r, g, b, a], axis=1).astype(np.float32)

        def _compute_fem_depth_colors(self) -> np.ndarray:
            """
            Compute per-node RGBA colors for FEM top nodes based on indentation depth.

            FEM here is driven by an imposed displacement field (no full FEM solve),
            so we color by the imposed top-surface Z displacement magnitude.
            """
            if self.fem is None:
                return np.zeros((0, 4), dtype=np.float32)

            depth = np.clip(-self.fem.current_displacements[self.fem.top_nodes, 2], 0.0, None)
            if depth.size == 0:
                return np.zeros((0, 4), dtype=np.float32)

            depth_max = float(np.max(self.fem_depths)) if self.fem_depths else float(depth.max())
            depth_max = max(depth_max, 1e-6)
            t = np.clip(depth / depth_max, 0.0, 1.0)

            r = t
            g = 0.2 * (1.0 - t)
            b = 1.0 - t
            a = np.full_like(t, 0.9)
            return np.stack([r, g, b, a], axis=1).astype(np.float32)

        def _init_point_clouds(self):
            """Initialize scatter plots for FEM top nodes and MPM particles."""
            if self.mode in ("fem", "both") and self.fem is not None:
                fem_pos = self.fem.get_top_surface_positions() + self.fem_offset
                fem_colors = self._compute_fem_depth_colors()
                self.fem_cloud = GLScatterPlotItem(
                    pos=fem_pos.astype(np.float32),
                    color=fem_colors,
                    size=0.6,
                    glOptions="ontop",
                )

            if self.mode in ("mpm", "both") and self.mpm is not None:
                # If history exists, use the first frame
                if self.mpm.positions_history:
                    pos = self.mpm.positions_history[0]
                else:
                    pos = self.mpm.solver.get_particle_data()["x"]
                pos_mm_full = pos * 1000.0  # m -> mm
                if self._mpm_sample_idx is not None:
                    pos_mm = pos_mm_full[self._mpm_sample_idx]
                else:
                    pos_mm = pos_mm_full

                colors = self._compute_mpm_depth_colors(pos_mm, self._mpm_sample_idx)
                pos_mm = pos_mm + self._mpm_align_shift_mm + self.mpm_offset
                self.mpm_cloud = GLScatterPlotItem(
                    pos=pos_mm.astype(np.float32),
                    color=colors,
                    size=0.3,
                    glOptions="ontop",
                )

        def step_frame(self):
            """Advance animation and update point clouds."""
            self._frame += 1

            has_mpm_history = self.mpm is not None and bool(self.mpm.positions_history)
            should_advance_mpm = has_mpm_history and (self._frame % self._mpm_hold_frames == 0)

            # When both are enabled and MPM has a schedule, drive FEM from MPM.
            if self._sync_fem_to_mpm and should_advance_mpm:
                idx = self._mpm_frame_idx
                if idx < len(self.mpm.indenter_depth_history_mm):
                    depth = float(self.mpm.indenter_depth_history_mm[idx])
                else:
                    depth = float(self.fem_depths[self._fem_depth_idx])
                    self._fem_depth_idx = (self._fem_depth_idx + 1) % len(self.fem_depths)

                center_xy = None
                if idx < len(getattr(self.mpm, "indenter_center_history_mm", [])):
                    try:
                        center_mm = (
                            np.array(self.mpm.indenter_center_history_mm[idx], dtype=np.float32)
                            + self._mpm_align_shift_mm
                        )
                        center_xy = (float(center_mm[0]), float(center_mm[1]))
                    except Exception:
                        center_xy = None

                self.fem.apply_indentation(depth, center_xy_mm=center_xy)
                fem_pos = self.fem.get_top_surface_positions() + self.fem_offset
                fem_colors = self._compute_fem_depth_colors()
                self.fem_cloud.setData(fem_pos.astype(np.float32), color=fem_colors)

            # FEM-only mode: cycle fixed indentation depths slowly.
            if (not self._sync_fem_to_mpm) and self.mode in ("fem", "both") and self.fem is not None:
                if self._frame % self._fem_hold_frames == 0:
                    depth = self.fem_depths[self._fem_depth_idx]
                    self.fem.apply_indentation(depth)
                    fem_pos = self.fem.get_top_surface_positions() + self.fem_offset
                    fem_colors = self._compute_fem_depth_colors()
                    self.fem_cloud.setData(fem_pos.astype(np.float32), color=fem_colors)
                    self._fem_depth_idx = (self._fem_depth_idx + 1) % len(self.fem_depths)

            # MPM time history playback (throttled)
            if should_advance_mpm and self.mode in ("mpm", "both"):
                pos = self.mpm.positions_history[self._mpm_frame_idx]
                pos_mm_full = pos * 1000.0
                if self._mpm_sample_idx is not None:
                    pos_mm = pos_mm_full[self._mpm_sample_idx]
                else:
                    pos_mm = pos_mm_full

                colors = self._compute_mpm_depth_colors(pos_mm, self._mpm_sample_idx)
                pos_mm = pos_mm + self._mpm_align_shift_mm + self.mpm_offset
                self.mpm_cloud.setData(pos_mm.astype(np.float32), color=colors)
                self._mpm_frame_idx = (self._mpm_frame_idx + 1) % len(self.mpm.positions_history)


    def run_visualization(
        fem_adapter: Optional[FEMDataAdapter],
        mpm_adapter: Optional[MPMAdapter],
        mode: str,
        fem_depths: List[float],
    ):
        """Run a simple 3D visualization loop."""
        scene = CompareScene(fem_adapter, mpm_adapter, mode, fem_depths)
        print("\n[Visualization] Close the window or press ESC to exit.")

        # Main render loop (approx 30 FPS)
        while not scene.windowShouldClose():
            scene.update()
            scene.step_frame()
            time.sleep(1.0 / 30.0)
else:
    def run_visualization(
        fem_adapter: Optional[FEMDataAdapter],
        mpm_adapter: Optional[MPMAdapter],
        mode: str,
        fem_depths: List[float],
    ):
        print("Visualization skipped: ezgl/xengym render not available.")

# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='MPM vs FEM Visual Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--fem-file', type=str,
        default=str(PROJ_DIR / "assets/data/fem_data_gel_2035.npz"),
        help='Path to FEM NPZ file'
    )
    parser.add_argument(
        '--mode', type=str, choices=['fem', 'mpm', 'both'], default='both',
        help='Simulation mode: fem, mpm, or both'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for comparison plot'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip generating comparison plot'
    )
    parser.add_argument(
        '--mpm-steps', type=int, default=200,
        help='Number of MPM simulation steps'
    )
    parser.add_argument(
        '--no-vis', action='store_true',
        help='Skip 3D visualization even if render is available'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MPM vs FEM Visual Comparison")
    print("=" * 60)

    # Update scene params from args
    SCENE_PARAMS['mpm_steps'] = args.mpm_steps

    fem_adapter = None
    mpm_adapter = None

    # Initialize FEM adapter
    if args.mode in ['fem', 'both']:
        print("\n--- Initializing FEM ---")
        try:
            fem_adapter = FEMDataAdapter(args.fem_file)
        except Exception as e:
            print(f"Failed to load FEM data: {e}")
            if args.mode == 'fem':
                return 1
        else:
            # Keep MPM block geometry consistent with the FEM gel mesh bounds.
            # FEM data is stored in millimeters; MPM expects us to choose a matching
            # block size to make side-by-side visualization meaningful.
            bounds = fem_adapter.gel_bounds
            width_mm = float(bounds["x_max"] - bounds["x_min"])
            height_mm = float(bounds["y_max"] - bounds["y_min"])
            thickness_mm = float(bounds["z_max"] - bounds["z_min"])
            if width_mm > 0 and height_mm > 0 and thickness_mm > 0:
                SCENE_PARAMS["block_size_mm"] = (width_mm, height_mm, thickness_mm)
                SCENE_PARAMS["block_center_mm"] = (0.0, 0.0, thickness_mm / 2.0)
                print(
                    f"  Using FEM gel bounds for MPM block_size_mm: "
                    f"{width_mm:.2f} x {height_mm:.2f} x {thickness_mm:.2f} mm"
                )

    # Initialize MPM adapter
    if args.mode in ['mpm', 'both']:
        print("\n--- Initializing MPM ---")
        if not HAS_TAICHI:
            print("Taichi not available, skipping MPM")
            if args.mode == 'mpm':
                return 1
        else:
            try:
                mpm_adapter = MPMAdapter(SCENE_PARAMS)
                mpm_adapter.run_simulation()
            except Exception as e:
                print(f"Failed to run MPM: {e}")
                import traceback
                traceback.print_exc()
                if args.mode == 'mpm':
                    return 1

    # Run comparison
    print("\n--- Running Comparison ---")
    engine = ComparisonEngine(fem_adapter, mpm_adapter)

    # Test with multiple indentation depths
    depths = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = engine.run_comparison(depths)

    # Generate plot
    if not args.no_plot:
        engine.plot_comparison(args.output)

    # 3D visualization (reuse existing render pipeline)
    if not args.no_vis:
        run_visualization(fem_adapter, mpm_adapter, args.mode, depths)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if fem_adapter:
        print(f"FEM: {len(fem_adapter.nodes)} nodes loaded")
    if mpm_adapter:
        print(f"MPM: {mpm_adapter.n_particles} particles simulated")
    print(f"Comparison depths: {depths} mm")

    return 0


if __name__ == '__main__':
    exit(main())
