"""
Field Management for MPM Solver
Defines particle fields, grid fields, and global energy scalars.
Supports automatic differentiation via enable_grad parameter.
"""
import taichi as ti
from typing import Tuple
from .config import MPMConfig


@ti.data_oriented
class MPMFields:
    """
    Manages all fields for MPM simulation:
    - Particle fields: position, velocity, deformation gradient, APIC matrix, mass, volume, Maxwell internal variables, energy increments
    - Grid fields: mass, velocity, tangential displacement, contact mask, no-contact age
    - Global scalars: kinetic energy, elastic energy, viscous energy (step/cumulative), projection energy (step/cumulative)

    When enable_grad=True, particle state fields (x, v, F, b_bar_e) are created with needs_grad=True
    for automatic differentiation support.
    """

    def __init__(self, config: MPMConfig, n_particles: int, enable_grad: bool = False):
        """
        Initialize MPM fields

        Args:
            config: MPM configuration
            n_particles: Number of particles
            enable_grad: If True, create particle state fields with needs_grad=True for autodiff
        """
        self.config = config
        self.n_particles = n_particles
        self.n_maxwell = len(config.material.maxwell_branches)
        self.enable_grad = enable_grad

        # Grid dimensions
        self.grid_size = config.grid.grid_size
        self.dx = config.grid.dx

        # Particle fields - with optional gradient support
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles, needs_grad=enable_grad)  # Position
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles, needs_grad=enable_grad)  # Velocity
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles, needs_grad=enable_grad)  # Deformation gradient
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)  # APIC affine matrix (no grad needed)
        self.mass = ti.field(dtype=ti.f32, shape=n_particles)  # Mass (typically not optimized)
        self.volume = ti.field(dtype=ti.f32, shape=n_particles)  # Volume (typically not optimized)

        # Maxwell branch internal variables: b_bar_e[k] for each branch
        if self.n_maxwell > 0:
            self.b_bar_e = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(n_particles, self.n_maxwell), needs_grad=enable_grad)
        else:
            self.b_bar_e = None

        # Energy increments (particle-level)
        self.delta_E_viscous_step = ti.field(dtype=ti.f32, shape=n_particles)  # Viscous dissipation this step
        self.delta_E_proj_step = ti.field(dtype=ti.f32, shape=n_particles)  # Projection correction this step

        # Grid fields
        self.grid_m = ti.field(dtype=ti.f32, shape=self.grid_size)  # Grid mass
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_size)  # Grid velocity
        self.grid_ut = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_size)  # Tangential displacement (persistent)
        self.grid_contact_mask = ti.field(dtype=ti.i32, shape=self.grid_size)  # Contact flag
        self.grid_nocontact_age = ti.field(dtype=ti.i32, shape=self.grid_size)  # No-contact age counter

        # Global energy scalars
        self.E_kin = ti.field(dtype=ti.f32, shape=())  # Kinetic energy
        self.E_elastic = ti.field(dtype=ti.f32, shape=())  # Elastic energy
        self.E_viscous_step = ti.field(dtype=ti.f32, shape=())  # Viscous dissipation this step
        self.E_viscous_cum = ti.field(dtype=ti.f32, shape=())  # Cumulative viscous dissipation
        self.E_proj_step = ti.field(dtype=ti.f32, shape=())  # Projection correction this step
        self.E_proj_cum = ti.field(dtype=ti.f32, shape=())  # Cumulative projection correction

    @ti.kernel
    def clear_grid(self):
        """Clear grid fields (except grid_ut and grid_nocontact_age which are persistent)"""
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector([0.0, 0.0, 0.0])
            # Clear contact mask (will be set in grid_op if in contact)
            self.grid_contact_mask[I] = 0
            # Note: grid_ut and grid_nocontact_age are NOT cleared here (persistent across steps)

    @ti.kernel
    def clear_particle_energy_increments(self):
        """Clear particle-level energy increments"""
        for p in range(self.n_particles):
            self.delta_E_viscous_step[p] = 0.0
            self.delta_E_proj_step[p] = 0.0

    @ti.kernel
    def clear_global_energy_step(self):
        """Clear step-level global energy accumulators"""
        self.E_viscous_step[None] = 0.0
        self.E_proj_step[None] = 0.0

    def initialize_particles(self, positions, velocities=None, volumes=None, density=None):
        """
        Initialize particle data

        Args:
            positions: (n_particles, 3) array of positions
            velocities: (n_particles, 3) array of velocities (optional)
            volumes: (n_particles,) array of volumes (optional)
            density: Material density (optional, uses config if not provided)
        """
        if density is None:
            density = self.config.material.density

        self.x.from_numpy(positions)

        if velocities is not None:
            self.v.from_numpy(velocities)
        else:
            self.v.fill(0.0)

        # Initialize deformation gradient to identity
        @ti.kernel
        def init_F():
            for p in range(self.n_particles):
                self.F[p] = ti.Matrix.identity(ti.f32, 3)
                self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)

        init_F()

        # Initialize volumes and masses
        if volumes is not None:
            self.volume.from_numpy(volumes)
        else:
            # Default: uniform volume based on grid spacing
            vol = (self.dx * 0.5) ** 3
            self.volume.fill(vol)

        @ti.kernel
        def init_mass():
            for p in range(self.n_particles):
                self.mass[p] = self.volume[p] * density

        init_mass()

        # Initialize Maxwell internal variables to identity
        if self.b_bar_e is not None:
            @ti.kernel
            def init_maxwell():
                for p in range(self.n_particles):
                    for k in ti.static(range(self.n_maxwell)):
                        self.b_bar_e[p, k] = ti.Matrix.identity(ti.f32, 3)

            init_maxwell()

    def get_particle_data(self):
        """Get particle data as numpy arrays"""
        return {
            'x': self.x.to_numpy(),
            'v': self.v.to_numpy(),
            'F': self.F.to_numpy(),
            'mass': self.mass.to_numpy(),
            'volume': self.volume.to_numpy()
        }

    def get_energy_data(self):
        """Get energy data as dictionary"""
        return {
            'E_kin': self.E_kin[None],
            'E_elastic': self.E_elastic[None],
            'E_viscous_step': self.E_viscous_step[None],
            'E_viscous_cum': self.E_viscous_cum[None],
            'E_proj_step': self.E_proj_step[None],
            'E_proj_cum': self.E_proj_cum[None]
        }

    def reset_gradients(self):
        """Reset gradient fields to zero. Only effective when enable_grad=True."""
        if not self.enable_grad:
            return

        # Reset particle field gradients using fill
        self.x.grad.fill(0.0)
        self.v.grad.fill(0.0)
        self.F.grad.fill(0.0)

        if self.b_bar_e is not None:
            self.b_bar_e.grad.fill(0.0)
