"""
Manual Adjoint Example - Differentiable MPM with Hand-written Adjoints

This example demonstrates the correct usage of ManualAdjointMPMSolver for
computing gradients w.r.t. material parameters and initial states.

IMPORTANT:
- configure_gradient_mode() must be called BEFORE ti.init()
- Only pure Ogden elastic materials are supported (no Maxwell/bulk viscosity)
- Numerical mode (default) is more accurate but slower (18 stress evals per particle)

Usage:
    python manual_adjoint_example.py
"""

# ============================================
# Step 1: Configure gradient mode BEFORE ti.init()
# ============================================
from xengym.mpm import configure_gradient_mode, validate_gradient_mode

# Configure: use numerical differentiation (accurate, recommended for calibration)
configure_gradient_mode(
    use_numerical=True,  # True: accurate but slow, False: fast but incomplete
    eps=1e-4             # Finite difference step size
)

# ============================================
# Step 2: Initialize Taichi AFTER configure
# ============================================
import taichi as ti
ti.init(arch=ti.cpu)  # Use ti.gpu for GPU acceleration

# ============================================
# Step 3: Create configuration (pure Ogden only!)
# ============================================
import numpy as np
from xengym.mpm import (
    MPMConfig,
    GridConfig,
    TimeConfig,
    MaterialConfig,
    OgdenConfig,
    ManualAdjointMPMSolver
)

# Grid configuration
grid_config = GridConfig(
    grid_size=(32, 32, 32),
    dx=0.01,
    origin=(0.0, 0.0, 0.0)
)

# Time configuration
time_config = TimeConfig(
    dt=1e-4,
    num_steps=100,
    substeps=1
)

# Material configuration - PURE OGDEN (no Maxwell branches!)
ogden_config = OgdenConfig(
    mu=[1000.0, 500.0],      # Shear moduli [Pa]
    alpha=[2.0, -2.0],       # Exponents
    kappa=10000.0            # Bulk modulus [Pa]
)

material_config = MaterialConfig(
    density=1000.0,
    ogden=ogden_config,
    maxwell_branches=[],           # MUST be empty for manual adjoint
    enable_bulk_viscosity=False    # MUST be False for manual adjoint
)

# Full configuration
config = MPMConfig(
    grid=grid_config,
    time=time_config,
    material=material_config
)

# Validate configuration (raises ValueError if Maxwell/bulk viscosity enabled)
try:
    validate_gradient_mode(config, strict=True)
    print("[OK] Configuration valid for manual adjoint")
except ValueError as e:
    print(f"[ERROR] {e}")
    exit(1)

# ============================================
# Step 4: Create particles
# ============================================
n_particles = 125  # 5x5x5 cube

# Create a cube of particles
positions = []
for i in range(5):
    for j in range(5):
        for k in range(5):
            x = 0.1 + i * 0.02
            y = 0.15 + j * 0.02  # Start above ground
            z = 0.1 + k * 0.02
            positions.append([x, y, z])
positions = np.array(positions, dtype=np.float32)

# Initial velocities (zero)
velocities = np.zeros((n_particles, 3), dtype=np.float32)

# ============================================
# Step 5: Create solver and initialize
# ============================================
solver = ManualAdjointMPMSolver(
    config,
    n_particles,
    max_grad_steps=100,       # Store up to 100 steps for backward pass
    maxwell_needs_grad=False  # No Maxwell gradients needed
)
solver.initialize_particles(positions, velocities)

# ============================================
# Step 6: Set target and run forward+backward
# ============================================
# Target: particles moved down by gravity
target_positions = positions.copy()
target_positions[:, 1] -= 0.05  # Target 5cm lower

solver.set_target_positions(target_positions)

# Run forward simulation with gradient computation
print("\nRunning forward simulation and backward pass...")
results = solver.solve_with_gradients(
    num_steps=50,
    loss_type='position',
    requires_grad={
        'ogden_mu': True,
        'ogden_alpha': True,
        'initial_x': True,
        'initial_v': True
    }
)

# ============================================
# Step 7: Examine results
# ============================================
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Loss (position MSE): {results['loss']:.6e}")
print(f"\nGradient w.r.t. Ogden mu:")
for i, g in enumerate(results['grad_ogden_mu']):
    print(f"  dL/d(mu[{i}]) = {g:.6e}")

print(f"\nGradient w.r.t. Ogden alpha:")
for i, g in enumerate(results['grad_ogden_alpha']):
    print(f"  dL/d(alpha[{i}]) = {g:.6e}")

print(f"\nGradient w.r.t. initial positions (first 3 particles):")
grad_x0 = results['grad_initial_x']
for p in range(min(3, n_particles)):
    print(f"  dL/d(x0[{p}]) = [{grad_x0[p, 0]:.6e}, {grad_x0[p, 1]:.6e}, {grad_x0[p, 2]:.6e}]")

# ============================================
# Step 8: Verify gradients (optional)
# ============================================
print("\n" + "="*50)
print("GRADIENT VERIFICATION (numerical vs analytical)")
print("="*50)

# Verify ogden_mu[0] gradient
verify_result = solver.verify_gradient_numerical(
    param_name='ogden_mu',
    param_idx=0,
    num_steps=10,
    loss_type='position',
    eps=1e-4
)
print(f"\nParameter: ogden_mu[0]")
print(f"  Analytical:  {verify_result['analytic']:.6e}")
print(f"  Numerical:   {verify_result['numerical']:.6e}")
print(f"  Rel. Error:  {verify_result['rel_error']:.2%}")

# Verify initial_x[0] gradient (particle 0, dimension 0)
verify_result = solver.verify_gradient_numerical(
    param_name='initial_x',
    param_idx=0,  # particle 0, dim 0 (x-component)
    num_steps=10,
    loss_type='position',
    eps=1e-4
)
print(f"\nParameter: initial_x[0, 0] (particle 0, x-component)")
print(f"  Analytical:  {verify_result['analytic']:.6e}")
print(f"  Numerical:   {verify_result['numerical']:.6e}")
print(f"  Rel. Error:  {verify_result['rel_error']:.2%}")

print("\n" + "="*50)
print("Example completed successfully!")
print("="*50)
