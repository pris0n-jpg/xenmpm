"""
MPM Solver Configuration
Provides dataclass-based configuration for grid, time stepping, materials, contact, and output options.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json
import yaml
from pathlib import Path


@dataclass
class GridConfig:
    """Grid configuration for MPM solver"""
    grid_size: Tuple[int, int, int] = (64, 64, 64)  # Grid resolution
    dx: float = 0.01  # Grid spacing in meters
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Grid origin


@dataclass
class TimeConfig:
    """Time stepping configuration"""
    dt: float = 1e-4  # Time step size
    num_steps: int = 1000  # Number of simulation steps
    substeps: int = 1  # Number of substeps per step


@dataclass
class OgdenConfig:
    """Ogden hyperelastic model parameters"""
    mu: List[float] = field(default_factory=lambda: [1e5, 1e4])  # Shear moduli
    alpha: List[float] = field(default_factory=lambda: [2.0, -2.0])  # Exponents
    kappa: float = 1e6  # Bulk modulus for volumetric response


@dataclass
class MaxwellBranchConfig:
    """Single Maxwell branch configuration"""
    G: float = 1e4  # Shear modulus
    tau: float = 0.1  # Relaxation time


@dataclass
class MaterialConfig:
    """Material configuration"""
    density: float = 1000.0  # kg/m^3

    # Ogden hyperelastic parameters
    ogden: OgdenConfig = field(default_factory=OgdenConfig)

    # Maxwell viscoelastic branches
    maxwell_branches: List[MaxwellBranchConfig] = field(default_factory=list)

    # Optional Kelvin-Voigt bulk viscosity
    enable_bulk_viscosity: bool = False
    bulk_viscosity: float = 0.0  # Pa·s


@dataclass
class SDFConfig:
    """SDF obstacle configuration

    Supported types:
    - 'plane': Infinite plane defined by point and normal
    - 'sphere': Sphere defined by center and radius
    - 'box': Axis-aligned box defined by center and half_extents
    - 'cylinder': Capped cylinder aligned with Z axis, defined by center, radius and half_height

    Parameters (as tuples):
    - plane: point=(x, y, z), normal=(nx, ny, nz)
    - sphere: center=(x, y, z), radius=r (stored as half_extents=(r, 0, 0))
    - box: center=(x, y, z), half_extents=(hx, hy, hz)
    - cylinder: center=(x, y, z), radius=r, half_height=h (stored as half_extents=(r, r, h))
    """
    sdf_type: str = 'plane'  # 'plane', 'sphere', 'box', 'cylinder'
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center/point on plane
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # Normal for plane (unit vector)
    half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Half extents for box, or (radius, 0, 0) for sphere


@dataclass
class ContactConfig:
    """Contact and friction configuration"""
    enable_contact: bool = True
    contact_stiffness_normal: float = 1e5  # Normal penalty stiffness
    contact_stiffness_tangent: float = 1e4  # Tangential spring stiffness

    # Friction parameters
    mu_s: float = 0.5  # Static friction coefficient
    mu_k: float = 0.3  # Kinetic friction coefficient
    friction_transition_vel: float = 1e-3  # Velocity for tanh transition

    # Contact cleanup parameters
    K_clear: int = 10  # Hysteresis counter threshold for clearing u_t

    # SDF obstacles (default: ground plane at z=0)
    # If empty, uses default ground plane for backward compatibility
    obstacles: List[SDFConfig] = field(default_factory=list)

    # Backward compatibility
    @property
    def contact_stiffness(self) -> float:
        """Deprecated: use contact_stiffness_normal instead"""
        return self.contact_stiffness_normal


@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = "output"
    save_particles: bool = True
    save_energy: bool = True
    save_contact_data: bool = True
    output_interval: int = 10  # Save every N steps


@dataclass
class MPMConfig:
    """Complete MPM solver configuration"""
    grid: GridConfig = field(default_factory=GridConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MPMConfig':
        """Load configuration from dictionary"""
        grid_dict = config_dict.get('grid', {})
        # Convert lists to tuples for grid_size and origin
        if 'grid_size' in grid_dict and isinstance(grid_dict['grid_size'], list):
            grid_dict['grid_size'] = tuple(grid_dict['grid_size'])
        if 'origin' in grid_dict and isinstance(grid_dict['origin'], list):
            grid_dict['origin'] = tuple(grid_dict['origin'])
        grid = GridConfig(**grid_dict)
        time = TimeConfig(**config_dict.get('time', {}))

        # Parse material config
        mat_dict = config_dict.get('material', {})
        ogden = OgdenConfig(**mat_dict.get('ogden', {}))
        maxwell_branches = [
            MaxwellBranchConfig(**b) for b in mat_dict.get('maxwell_branches', [])
        ]
        material = MaterialConfig(
            density=mat_dict.get('density', 1000.0),
            ogden=ogden,
            maxwell_branches=maxwell_branches,
            enable_bulk_viscosity=mat_dict.get('enable_bulk_viscosity', False),
            bulk_viscosity=mat_dict.get('bulk_viscosity', 0.0)
        )

        # Parse contact config with backward compatibility
        contact_dict = config_dict.get('contact', {})
        # Handle old 'contact_stiffness' parameter
        if 'contact_stiffness' in contact_dict and 'contact_stiffness_normal' not in contact_dict:
            contact_dict['contact_stiffness_normal'] = contact_dict.pop('contact_stiffness')
            contact_dict['contact_stiffness_tangent'] = contact_dict['contact_stiffness_normal'] * 0.1

        # Parse SDF obstacles
        obstacles_list = contact_dict.pop('obstacles', [])
        obstacles = [
            SDFConfig(
                sdf_type=o.get('sdf_type', 'plane'),
                center=tuple(o.get('center', [0.0, 0.0, 0.0])),
                normal=tuple(o.get('normal', [0.0, 0.0, 1.0])),
                half_extents=tuple(o.get('half_extents', [0.0, 0.0, 0.0]))
            ) for o in obstacles_list
        ]
        contact = ContactConfig(**contact_dict, obstacles=obstacles)
        output = OutputConfig(**config_dict.get('output', {}))

        return cls(
            grid=grid,
            time=time,
            material=material,
            contact=contact,
            output=output
        )

    @classmethod
    def from_json(cls, json_path: str) -> 'MPMConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MPMConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'grid': {
                'grid_size': list(self.grid.grid_size),
                'dx': self.grid.dx,
                'origin': list(self.grid.origin)
            },
            'time': {
                'dt': self.time.dt,
                'num_steps': self.time.num_steps,
                'substeps': self.time.substeps
            },
            'material': {
                'density': self.material.density,
                'ogden': {
                    'mu': self.material.ogden.mu,
                    'alpha': self.material.ogden.alpha,
                    'kappa': self.material.ogden.kappa
                },
                'maxwell_branches': [
                    {'G': b.G, 'tau': b.tau} for b in self.material.maxwell_branches
                ],
                'enable_bulk_viscosity': self.material.enable_bulk_viscosity,
                'bulk_viscosity': self.material.bulk_viscosity
            },
            'contact': {
                'enable_contact': self.contact.enable_contact,
                'contact_stiffness_normal': self.contact.contact_stiffness_normal,
                'contact_stiffness_tangent': self.contact.contact_stiffness_tangent,
                'mu_s': self.contact.mu_s,
                'mu_k': self.contact.mu_k,
                'friction_transition_vel': self.contact.friction_transition_vel,
                'K_clear': self.contact.K_clear,
                'obstacles': [
                    {
                        'sdf_type': o.sdf_type,
                        'center': list(o.center),
                        'normal': list(o.normal),
                        'half_extents': list(o.half_extents)
                    } for o in self.contact.obstacles
                ]
            },
            'output': {
                'output_dir': self.output.output_dir,
                'save_particles': self.output.save_particles,
                'save_energy': self.output.save_energy,
                'save_contact_data': self.output.save_contact_data,
                'output_interval': self.output.output_interval
            }
        }

    def save_json(self, json_path: str):
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
