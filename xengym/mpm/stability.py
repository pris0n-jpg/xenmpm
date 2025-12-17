"""
Stability Checks for MPM Solver
Provides parameter validation and time step constraint checks
"""
import numpy as np
from typing import List, Tuple
from .config import MPMConfig


def check_ogden_drucker_stability(mu: List[float], alpha: List[float]) -> Tuple[bool, str]:
    """
    Check Drucker-type stability condition for Ogden model

    The Ogden model is stable if:
    1. All mu_k * alpha_k > 0 (same sign)
    2. Sum of mu_k * alpha_k > 0 (positive definite at identity)
    3. The model is convex along principal stretch paths

    Args:
        mu: List of shear moduli
        alpha: List of exponents

    Returns:
        is_stable: True if stable
        message: Diagnostic message
    """
    if len(mu) != len(alpha):
        return False, "Length mismatch between mu and alpha"

    # Check 1: Same sign condition
    products = [m * a for m, a in zip(mu, alpha)]

    if not all(p > 0 for p in products) and not all(p < 0 for p in products):
        return False, f"Ogden parameters violate sign consistency: mu*alpha = {products}"

    # Check 2: Positive definiteness at identity
    sum_products = sum(products)
    if sum_products <= 0:
        return False, f"Ogden model not positive definite at identity: sum(mu*alpha) = {sum_products}"

    # Check 3: Path scan for convexity (simplified check)
    # Sample principal stretches from 0.5 to 2.0
    lambda_samples = np.linspace(0.5, 2.0, 20)

    for lam in lambda_samples:
        # Second derivative of energy w.r.t. lambda
        d2W = 0.0
        for m, a in zip(mu, alpha):
            if a >= 2:
                d2W += m * a * (a - 1) * lam ** (a - 2)
            else:
                # For alpha < 2, check at specific points
                d2W += m * a * (a - 1) * lam ** (a - 2)

        if d2W < -1e-6:  # Allow small numerical tolerance
            return False, f"Ogden model may not be convex at lambda={lam:.2f}, d2W/dlambda2={d2W:.3e}"

    return True, "Ogden parameters satisfy Drucker-type stability conditions"


def compute_cfl_timestep(dx: float, E: float, rho: float, safety_factor: float = 0.5) -> float:
    """
    Compute CFL-limited time step for elastic wave propagation

    dt <= safety_factor * dx / c_wave
    where c_wave = sqrt(E / rho) is the wave speed

    Args:
        dx: Grid spacing
        E: Elastic modulus (approximate)
        rho: Density
        safety_factor: Safety factor (typically 0.5)

    Returns:
        dt_cfl: CFL-limited time step
    """
    c_wave = np.sqrt(E / rho)
    dt_cfl = safety_factor * dx / c_wave
    return dt_cfl


def compute_viscous_timestep(eta: float, E: float, safety_factor: float = 0.1) -> float:
    """
    Compute time step constraint for viscous relaxation

    dt <= safety_factor * eta / E

    Args:
        eta: Viscosity
        E: Elastic modulus
        safety_factor: Safety factor (typically 0.1)

    Returns:
        dt_visc: Viscous time step constraint
    """
    dt_visc = safety_factor * eta / E
    return dt_visc


def compute_contact_timestep(dx: float, k_contact: float, rho: float, safety_factor: float = 0.1) -> float:
    """
    Compute time step constraint for contact stiffness

    dt <= safety_factor * sqrt(rho * dx^3 / k_contact)

    Args:
        dx: Grid spacing
        k_contact: Contact stiffness
        rho: Density
        safety_factor: Safety factor (typically 0.1)

    Returns:
        dt_contact: Contact time step constraint
    """
    dt_contact = safety_factor * np.sqrt(rho * dx**3 / k_contact)
    return dt_contact


def check_timestep_constraints(config: MPMConfig) -> Tuple[bool, str]:
    """
    Check if time step satisfies all stability constraints

    Args:
        config: MPM configuration

    Returns:
        is_valid: True if time step is valid
        message: Diagnostic message with recommendations
    """
    dt = config.time.dt
    dx = config.grid.dx
    rho = config.material.density

    # Estimate elastic modulus from Ogden parameters
    E_approx = sum(config.material.ogden.mu)

    # 1. CFL constraint
    dt_cfl = compute_cfl_timestep(dx, E_approx, rho)

    # 2. Viscous constraint (if Maxwell branches present)
    dt_visc = float('inf')
    if config.material.maxwell_branches:
        for branch in config.material.maxwell_branches:
            dt_v = compute_viscous_timestep(branch.G * branch.tau, branch.G)
            dt_visc = min(dt_visc, dt_v)

    # 3. Contact constraint (if contact enabled)
    dt_contact = float('inf')
    if config.contact.enable_contact:
        dt_contact = compute_contact_timestep(dx, config.contact.contact_stiffness, rho)

    # Find most restrictive constraint
    dt_min = min(dt_cfl, dt_visc, dt_contact)

    messages = []
    messages.append(f"Time step constraints:")
    messages.append(f"  Current dt: {dt:.3e} s")
    messages.append(f"  CFL limit (elastic): {dt_cfl:.3e} s")

    if dt_visc < float('inf'):
        messages.append(f"  Viscous limit: {dt_visc:.3e} s")

    if dt_contact < float('inf'):
        messages.append(f"  Contact limit: {dt_contact:.3e} s")

    messages.append(f"  Recommended dt: {dt_min:.3e} s")

    if dt > dt_min:
        messages.append(f"⚠ WARNING: Time step exceeds stability limit by {dt/dt_min:.2f}x")
        return False, "\n".join(messages)
    else:
        messages.append(f"✓ Time step is within stability limits ({dt/dt_min:.2f}x of limit)")
        return True, "\n".join(messages)


def validate_config(config: MPMConfig, verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate complete MPM configuration

    Args:
        config: MPM configuration
        verbose: Print diagnostic messages

    Returns:
        is_valid: True if configuration is valid
        messages: List of diagnostic messages
    """
    messages = []
    is_valid = True

    # 1. Check Ogden parameters
    ogden_stable, ogden_msg = check_ogden_drucker_stability(
        config.material.ogden.mu,
        config.material.ogden.alpha
    )

    if not ogden_stable:
        is_valid = False
        messages.append(f"✗ Ogden stability check failed: {ogden_msg}")
    else:
        messages.append(f"✓ Ogden stability check passed")

    # 2. Check time step constraints
    dt_valid, dt_msg = check_timestep_constraints(config)

    if not dt_valid:
        is_valid = False
        messages.append(f"✗ Time step constraint check failed")
    else:
        messages.append(f"✓ Time step constraint check passed")

    messages.append(dt_msg)

    # 3. Check grid resolution
    if config.grid.dx <= 0:
        is_valid = False
        messages.append(f"✗ Invalid grid spacing: dx = {config.grid.dx}")
    else:
        messages.append(f"✓ Grid spacing: dx = {config.grid.dx} m")

    # 4. Check material parameters
    if config.material.density <= 0:
        is_valid = False
        messages.append(f"✗ Invalid density: rho = {config.material.density}")
    else:
        messages.append(f"✓ Density: rho = {config.material.density} kg/m³")

    if config.material.ogden.kappa <= 0:
        is_valid = False
        messages.append(f"✗ Invalid bulk modulus: kappa = {config.material.ogden.kappa}")
    else:
        messages.append(f"✓ Bulk modulus: kappa = {config.material.ogden.kappa} Pa")

    # 5. Check contact parameters
    if config.contact.enable_contact:
        if config.contact.mu_s < 0 or config.contact.mu_k < 0:
            is_valid = False
            messages.append(f"✗ Invalid friction coefficients: mu_s={config.contact.mu_s}, mu_k={config.contact.mu_k}")
        else:
            messages.append(f"✓ Friction coefficients: mu_s={config.contact.mu_s}, mu_k={config.contact.mu_k}")

        if config.contact.mu_k > config.contact.mu_s:
            messages.append(f"⚠ Warning: Kinetic friction > static friction (unusual)")

    if verbose:
        print("\n" + "="*60)
        print("MPM Configuration Validation")
        print("="*60)
        for msg in messages:
            print(msg)
        print("="*60)
        if is_valid:
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration has errors")
        print("="*60 + "\n")

    return is_valid, messages


if __name__ == '__main__':
    # Test validation
    from .config import MPMConfig

    print("Testing stability checks...")

    # Test 1: Valid configuration
    config = MPMConfig()
    is_valid, messages = validate_config(config, verbose=True)

    # Test 2: Invalid Ogden parameters
    print("\nTesting invalid Ogden parameters...")
    config_bad = MPMConfig()
    config_bad.material.ogden.mu = [1e5, -1e4]
    config_bad.material.ogden.alpha = [2.0, 2.0]  # Both positive alpha with mixed sign mu
    is_valid, messages = validate_config(config_bad, verbose=True)

    # Test 3: Time step too large
    print("\nTesting large time step...")
    config_large_dt = MPMConfig()
    config_large_dt.time.dt = 1e-2  # Very large time step
    is_valid, messages = validate_config(config_large_dt, verbose=True)
