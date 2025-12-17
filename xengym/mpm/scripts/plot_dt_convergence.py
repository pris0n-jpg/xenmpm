#!/usr/bin/env python
"""
Time Step Convergence Verification Script

Validates FR-5 requirement: |E_proj_cum| / E_viscous_cum convergence with dt.

This script:
1. Runs energy_convergence simulation at multiple dt values
2. Computes |E_proj_cum| / E_viscous_cum ratio at final time
3. Plots log-log convergence curve
4. Verifies first-order convergence (slope ~ 1)

Expected result:
    |E_proj_cum| / E_viscous_cum ~ O(dt)  (first-order convergence)
    In log-log plot, slope should be approximately 1.0

Usage:
    python -m xengym.mpm.scripts.plot_dt_convergence --output convergence_results/
    python -m xengym.mpm.scripts.plot_dt_convergence --dt-factors 1.0 0.5 0.25 0.125
"""
import argparse
import numpy as np
import csv
from pathlib import Path
from typing import List, Dict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")


def run_single_dt_simulation(dt_factor: float, base_config: dict,
                             num_steps_base: int = 400) -> Dict:
    """
    Run a single simulation with scaled dt.

    Args:
        dt_factor: Multiplier for base dt (smaller = finer)
        base_config: Base configuration dict
        num_steps_base: Base number of steps

    Returns:
        Dict with final energy values and ratio
    """
    import taichi as ti
    from xengym.mpm.config import MPMConfig
    from xengym.mpm.mpm_solver import MPMSolver

    # Scale dt and steps to maintain same total time
    config = MPMConfig()

    # Grid config
    config.grid.grid_size = base_config.get('grid_size', (32, 32, 32))
    config.grid.dx = base_config.get('dx', 0.02)

    # Time config - scale dt, adjust steps to keep total time constant
    base_dt = base_config.get('dt', 2e-4)
    config.time.dt = base_dt * dt_factor
    # More steps for smaller dt to reach same final time
    config.time.num_steps = int(num_steps_base / dt_factor)

    # Material config - must have Maxwell for viscous dissipation
    config.material.density = base_config.get('density', 1000.0)
    config.material.ogden.mu = base_config.get('mu', [30000.0])
    config.material.ogden.alpha = base_config.get('alpha', [2.0])
    config.material.ogden.kappa = base_config.get('kappa', 300000.0)

    # Add Maxwell branch for viscous dissipation
    from xengym.mpm.config import MaxwellBranch
    config.material.maxwell_branches = [
        MaxwellBranch(G=10000.0, tau=0.02)
    ]

    # Contact config
    config.contact.enable_contact = True
    config.contact.contact_stiffness_normal = 500000.0

    # Create particles (bouncing cube)
    size = 0.06
    spacing = config.grid.dx * 0.5
    x = np.arange(0.15, 0.15 + size, spacing)
    y = np.arange(0.15, 0.15 + size, spacing)
    z = np.arange(0.1, 0.1 + size, spacing)  # Start above ground

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    n_particles = len(positions)

    # Create solver and run
    solver = MPMSolver(config, n_particles)
    solver.initialize_particles(positions.astype(np.float32))

    # Run simulation
    energy_history = []
    for step in range(config.time.num_steps):
        solver.step()

        if step % max(1, config.time.num_steps // 50) == 0:
            energy = solver.get_energy_data()
            energy_history.append({
                'step': step,
                'time': step * config.time.dt,
                'E_viscous_cum': energy['E_viscous_cum'],
                'E_proj_cum': energy['E_proj_cum']
            })

    # Get final values
    final_energy = solver.get_energy_data()
    E_viscous_cum = final_energy['E_viscous_cum']
    E_proj_cum = final_energy['E_proj_cum']

    # Compute ratio with robust handling of E_viscous_cum ≈ 0
    MIN_VISCOUS_THRESHOLD = 1e-10
    ratio = None
    ratio_valid = True
    warning_msg = None

    if abs(E_viscous_cum) < MIN_VISCOUS_THRESHOLD:
        # E_viscous_cum is essentially zero - convergence ratio undefined
        ratio = float('nan')
        ratio_valid = False
        warning_msg = (
            f"E_viscous_cum ({E_viscous_cum:.2e}) is below threshold ({MIN_VISCOUS_THRESHOLD:.0e}). "
            f"This typically means no Maxwell viscosity is active or simulation is too short. "
            f"Convergence ratio is undefined."
        )
    else:
        ratio = abs(E_proj_cum) / abs(E_viscous_cum)

    return {
        'dt': config.time.dt,
        'dt_factor': dt_factor,
        'num_steps': config.time.num_steps,
        'final_time': config.time.num_steps * config.time.dt,
        'E_viscous_cum': E_viscous_cum,
        'E_proj_cum': E_proj_cum,
        'ratio': ratio,
        'ratio_valid': ratio_valid,
        'warning': warning_msg,
        'energy_history': energy_history
    }


def run_convergence_study(dt_factors: List[float], output_dir: Path,
                          base_config: dict = None) -> List[Dict]:
    """
    Run convergence study across multiple dt values.

    Args:
        dt_factors: List of dt multipliers (e.g., [1.0, 0.5, 0.25])
        output_dir: Output directory for results
        base_config: Base configuration parameters

    Returns:
        List of results for each dt
    """
    import taichi as ti

    if base_config is None:
        base_config = {
            'grid_size': (32, 32, 32),
            'dx': 0.02,
            'dt': 2e-4,
            'density': 1000.0,
            'mu': [30000.0],
            'alpha': [2.0],
            'kappa': 300000.0
        }

    results = []

    for i, dt_factor in enumerate(dt_factors):
        print(f"\n[{i+1}/{len(dt_factors)}] Running dt_factor = {dt_factor:.4f}")

        # Re-initialize Taichi for each run to reset state
        # (In practice, you might want to reset fields instead)
        try:
            ti.reset()
        except Exception:
            pass

        ti.init(arch=ti.cpu, default_fp=ti.f32)

        result = run_single_dt_simulation(dt_factor, base_config)
        results.append(result)

        # Print result with warning if applicable
        ratio_str = f"{result['ratio']:.4e}" if result['ratio_valid'] else "INVALID"
        print(f"    dt = {result['dt']:.2e}, "
              f"E_viscous_cum = {result['E_viscous_cum']:.4e}, "
              f"E_proj_cum = {result['E_proj_cum']:.4e}, "
              f"ratio = {ratio_str}")

        if result['warning']:
            print(f"    ⚠️  WARNING: {result['warning']}")

    # Check if any results have valid ratios
    valid_count = sum(1 for r in results if r['ratio_valid'])
    if valid_count == 0:
        print("\n" + "=" * 60)
        print("ERROR: No valid convergence ratios computed!")
        print("This typically means E_viscous_cum ≈ 0 for all runs.")
        print("Possible causes:")
        print("  1. No Maxwell branches configured")
        print("  2. Simulation too short for viscous dissipation")
        print("  3. Material parameters prevent viscous behavior")
        print("=" * 60)

    return results


def plot_convergence(results: List[Dict], output_dir: Path):
    """
    Plot dt convergence curve (log-log).

    Args:
        results: List of simulation results
        output_dir: Output directory for plots
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib required for plotting")
        return

    # Extract data, filtering for valid ratios only
    valid_results = [r for r in results if r.get('ratio_valid', True) and not np.isnan(r['ratio'])]

    if len(valid_results) < 2:
        print("ERROR: Need at least 2 valid data points for convergence plot.")
        print(f"  Valid points: {len(valid_results)} / {len(results)}")
        if len(results) > len(valid_results):
            print("  Some runs had E_viscous_cum ≈ 0, making ratio undefined.")
            print("  Ensure Maxwell viscosity is active in your configuration.")
        return

    dt_values = np.array([r['dt'] for r in valid_results])
    ratios = np.array([r['ratio'] for r in valid_results])

    # Additional filter for very small ratios (numerical noise)
    valid_mask = ratios > 1e-15
    if not np.any(valid_mask):
        print("Warning: All valid ratios are too small for log-log plot")
        return

    dt_valid = dt_values[valid_mask]
    ratio_valid = ratios[valid_mask]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Log-log convergence plot
    ax = axes[0]
    ax.loglog(dt_valid, ratio_valid, 'bo-', markersize=8, linewidth=2, label='Simulation')

    # Fit first-order reference line
    if len(dt_valid) >= 2:
        log_dt = np.log(dt_valid)
        log_ratio = np.log(ratio_valid)

        # Linear fit: log(ratio) = slope * log(dt) + intercept
        slope, intercept = np.polyfit(log_dt, log_ratio, 1)

        # Plot reference line
        dt_ref = np.logspace(np.log10(dt_valid.min()), np.log10(dt_valid.max()), 50)
        ratio_ref = np.exp(intercept) * dt_ref ** slope
        ax.loglog(dt_ref, ratio_ref, 'r--', linewidth=1.5,
                  label=f'Fit: slope = {slope:.2f}')

        # Plot ideal first-order line for comparison
        ratio_first = ratio_valid[0] * (dt_ref / dt_valid[0])
        ax.loglog(dt_ref, ratio_first, 'g:', linewidth=1.5,
                  label='Ideal O(dt)')

        ax.text(0.05, 0.95, f'Measured slope: {slope:.3f}\nExpected: 1.0',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Time step dt (s)', fontsize=12)
    ax.set_ylabel('|E_proj_cum| / E_viscous_cum', fontsize=12)
    ax.set_title('Projection Energy Convergence (FR-5)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, which='both')

    # Right: Table of values
    ax = axes[1]
    ax.axis('off')

    # Create table data
    table_data = [['dt (s)', 'E_viscous_cum (J)', 'E_proj_cum (J)', 'Ratio']]
    for r in results:
        table_data.append([
            f'{r["dt"]:.2e}',
            f'{r["E_viscous_cum"]:.4e}',
            f'{r["E_proj_cum"]:.4e}',
            f'{r["ratio"]:.4e}'
        ])

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', weight='bold')

    ax.set_title('Convergence Data', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'dt_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved: {output_dir / 'dt_convergence.png'}")

    # Also save CSV
    csv_path = output_dir / 'dt_convergence.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dt', 'dt_factor', 'num_steps',
                                                'final_time', 'E_viscous_cum',
                                                'E_proj_cum', 'ratio'])
        writer.writeheader()
        for r in results:
            writer.writerow({k: v for k, v in r.items() if k != 'energy_history'})

    print(f"  Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run dt convergence study for E_proj/E_viscous ratio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: 4 dt values (1.0, 0.5, 0.25, 0.125)
    python -m xengym.mpm.scripts.plot_dt_convergence

    # Custom dt factors
    python -m xengym.mpm.scripts.plot_dt_convergence --dt-factors 1.0 0.5 0.25

    # Specify output directory
    python -m xengym.mpm.scripts.plot_dt_convergence --output results/convergence/
"""
    )
    parser.add_argument('--output', '-o', type=str, default='convergence_output',
                        help='Output directory for results')
    parser.add_argument('--dt-factors', nargs='+', type=float,
                        default=[1.0, 0.5, 0.25, 0.125],
                        help='Time step multipliers (default: 1.0 0.5 0.25 0.125)')
    parser.add_argument('--base-dt', type=float, default=2e-4,
                        help='Base time step in seconds (default: 2e-4)')
    parser.add_argument('--base-steps', type=int, default=400,
                        help='Base number of steps (default: 400)')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MPM Time Step Convergence Study")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"dt factors: {args.dt_factors}")
    print(f"Base dt: {args.base_dt:.2e} s")
    print(f"Base steps: {args.base_steps}")

    base_config = {
        'grid_size': (32, 32, 32),
        'dx': 0.02,
        'dt': args.base_dt,
        'density': 1000.0,
        'mu': [30000.0],
        'alpha': [2.0],
        'kappa': 300000.0
    }

    # Run convergence study
    results = run_convergence_study(
        dt_factors=sorted(args.dt_factors, reverse=True),  # Largest first
        output_dir=output_dir,
        base_config=base_config
    )

    # Plot results
    print("\nGenerating convergence plot...")
    plot_convergence(results, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Convergence Study Complete")
    print("=" * 60)

    if len(results) >= 2:
        dt_values = np.array([r['dt'] for r in results])
        ratios = np.array([r['ratio'] for r in results])
        valid_mask = ratios > 1e-15

        if np.sum(valid_mask) >= 2:
            log_dt = np.log(dt_values[valid_mask])
            log_ratio = np.log(ratios[valid_mask])
            slope, _ = np.polyfit(log_dt, log_ratio, 1)

            print(f"\nMeasured convergence order: {slope:.3f}")
            print(f"Expected (first-order): 1.0")

            if 0.8 <= slope <= 1.2:
                print("PASS: Convergence is approximately first-order")
            else:
                print(f"WARNING: Convergence order ({slope:.2f}) differs from expected (1.0)")


if __name__ == '__main__':
    main()
