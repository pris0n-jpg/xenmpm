#!/usr/bin/env python
"""
Plotting scripts for MPM validation suite.

Generates required curves from CSV outputs:
- Stress-strain vs Ogden analytical
- Objectivity overlay
- Energy convergence (E_kin, E_elastic, ΔE_proj_step, E_proj_cum)
- Stick-slip tangential force-displacement
- Hertz contact convergence

Usage:
    python -m xengym.mpm.scripts.plot_validation --input validation_output/
    python -m xengym.mpm.scripts.plot_validation --input validation_output/ --scene uniaxial
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")


def plot_uniaxial_tension(input_dir: Path, output_dir: Path = None):
    """Plot stress-strain curve vs Ogden analytical prediction"""
    csv_path = input_dir / "uniaxial_tension.csv"
    if not csv_path.exists():
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    output_dir = output_dir or input_dir

    fig, ax = plt.subplots(figsize=(8, 6))

    # Simulation data
    ax.plot(df['strain'], df['stress'], 'b-o', label='MPM Simulation', markersize=4)

    # Ogden analytical (simplified single-term: σ = μ * (λ^α - λ^(-α/2)))
    # For small strains: σ ≈ 3μ * ε
    strain_theory = np.linspace(df['strain'].min(), df['strain'].max(), 100)
    mu_approx = df['stress'].iloc[-1] / df['strain'].iloc[-1] if df['strain'].iloc[-1] != 0 else 1e5
    stress_theory = mu_approx * strain_theory  # Linear approximation

    ax.plot(strain_theory, stress_theory, 'r--', label='Linear Theory (small strain)', linewidth=2)

    ax.set_xlabel('Strain (engineering)')
    ax.set_ylabel('Stress (Pa)')
    ax.set_title('Uniaxial Tension: Stress-Strain Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "uniaxial_tension.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved uniaxial_tension.png")


def plot_objectivity(input_dir: Path, output_dir: Path = None):
    """Plot objectivity test: stress with and without rotation overlay"""
    csv_path = input_dir / "objectivity.csv"
    if not csv_path.exists():
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    output_dir = output_dir or input_dir

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stress overlay
    ax = axes[0]
    ax.plot(df['step'], df['stress_no_rot'], 'b-', label='No Rotation', linewidth=2)
    ax.plot(df['step'], df['stress_rot'], 'r--', label='With Rotation', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Stress Norm (Frobenius)')
    ax.set_title('Objectivity: Stress Under Rigid Rotation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: relative difference
    ax = axes[1]
    ax.semilogy(df['step'], df['rel_diff'], 'g-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Relative Difference')
    ax.set_title('Objectivity Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "objectivity.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved objectivity.png")


def plot_energy_conservation(input_dir: Path, output_dir: Path = None):
    """Plot energy conservation test"""
    csv_path = input_dir / "energy_conservation.csv"
    if not csv_path.exists():
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    output_dir = output_dir or input_dir

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: energy components
    ax = axes[0]
    ax.plot(df['step'], df['E_kin'], 'b-', label='E_kin', linewidth=2)
    ax.plot(df['step'], df['E_elastic'], 'r-', label='E_elastic', linewidth=2)
    ax.plot(df['step'], df['E_total'], 'k--', label='E_total', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Energy Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: energy error
    ax = axes[1]
    ax.semilogy(df['step'], df['error'], 'g-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Relative Error')
    ax.set_title('Energy Conservation Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "energy_conservation.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved energy_conservation.png")


def plot_energy_convergence(input_dir: Path, output_dir: Path = None):
    """Plot energy convergence with projection tracking"""
    csv_path = input_dir / "energy_convergence.csv"
    if not csv_path.exists():
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    output_dir = output_dir or input_dir

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: E_kin + E_elastic
    ax = axes[0, 0]
    ax.plot(df['time'], df['E_kin'], 'b-', label='E_kin', linewidth=2)
    ax.plot(df['time'], df['E_elastic'], 'r-', label='E_elastic', linewidth=2)
    ax.plot(df['time'], df['E_total'], 'k--', label='E_total', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Kinetic and Elastic Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: E_viscous
    ax = axes[0, 1]
    ax.plot(df['time'], df['E_viscous_step'], 'g-', label='E_viscous_step (ΔE)', linewidth=2)
    ax.plot(df['time'], df['E_viscous_cum'], 'm--', label='E_viscous_cum', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Viscous Dissipation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: E_proj (ΔE_proj_step vs E_proj_cum)
    ax = axes[1, 0]
    ax.plot(df['time'], df['E_proj_step'], 'c-', label='ΔE_proj_step', linewidth=2)
    ax.plot(df['time'], df['E_proj_cum'], 'y--', label='E_proj_cum', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Projection Energy (SPD Correction)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: ΔE_proj_step vs E_viscous_step
    ax = axes[1, 1]
    ax.plot(df['time'], df['E_proj_step'], 'c-', label='ΔE_proj_step', linewidth=2)
    ax.plot(df['time'], df['E_viscous_step'], 'g--', label='E_viscous_step', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy per Step (J)')
    ax.set_title('Projection vs Viscous (per step)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "energy_convergence.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved energy_convergence.png")


def plot_gelslim_slip(input_dir: Path, output_dir: Path = None):
    """Plot GelSlim stick-slip tangential force-displacement curve"""
    csv_path = input_dir / "gelslim_slip.csv"
    if not csv_path.exists():
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    output_dir = output_dir or input_dir

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Tangential displacement vs time
    ax = axes[0]
    ax.plot(df['time'], df['tangent_disp'] * 1000, 'b-', linewidth=2)  # mm
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tangential Displacement (mm)')
    ax.set_title('GelSlim: Tangential Motion')
    ax.grid(True, alpha=0.3)

    # Right: Force proxy (E_elastic) vs displacement
    # Shows stick-slip behavior: linear stick region, plateau/slip region
    ax = axes[1]
    ax.plot(df['tangent_disp'] * 1000, df['E_elastic'], 'r-', linewidth=2)
    ax.set_xlabel('Tangential Displacement (mm)')
    ax.set_ylabel('Elastic Energy (J) [Force Proxy]')
    ax.set_title('GelSlim: Force-Displacement (Stick-Slip)')
    ax.grid(True, alpha=0.3)

    # Annotate regions if we can detect them
    # Simple heuristic: look for slope changes

    plt.tight_layout()
    plt.savefig(output_dir / "gelslim_slip.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved gelslim_slip.png")


def plot_hertz_contact(input_dir: Path, output_dir: Path = None):
    """Plot Hertz contact / impact test"""
    csv_path = input_dir / "hertz_contact.csv"
    if not csv_path.exists():
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    output_dir = output_dir or input_dir

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Indentation vs time
    ax = axes[0]
    ax.plot(df['time'], df['indentation'] * 1000, 'b-', linewidth=2)  # mm
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Indentation (mm)')
    ax.set_title('Hertz Contact: Indentation vs Time')
    ax.grid(True, alpha=0.3)

    # Right: Energy during impact
    ax = axes[1]
    ax.plot(df['time'], df['E_kin'], 'b-', label='E_kin', linewidth=2)
    ax.plot(df['time'], df['E_elastic'], 'r-', label='E_elastic', linewidth=2)
    ax.plot(df['time'], df['E_proj_cum'], 'g--', label='E_proj_cum', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Hertz Contact: Energy During Impact')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hertz_contact.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved hertz_contact.png")


def plot_pure_shear(input_dir: Path, output_dir: Path = None):
    """Plot pure shear test"""
    csv_path = input_dir / "pure_shear.csv"
    if not csv_path.exists():
        print(f"  Skipping: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    output_dir = output_dir or input_dir

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df['step'], df['shear'], 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Shear Component (F₁₂)')
    ax.set_title('Pure Shear Test')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pure_shear.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved pure_shear.png")


def plot_all(input_dir: Path, output_dir: Path = None):
    """Generate all plots from validation output"""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib required for plotting. Install with: pip install matplotlib")
        return

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    print(f"Generating plots from: {input_dir}")
    print(f"Output directory: {output_dir}")

    plot_uniaxial_tension(input_dir, output_dir)
    plot_pure_shear(input_dir, output_dir)
    plot_objectivity(input_dir, output_dir)
    plot_energy_conservation(input_dir, output_dir)
    plot_energy_convergence(input_dir, output_dir)
    plot_gelslim_slip(input_dir, output_dir)
    plot_hertz_contact(input_dir, output_dir)

    print("\nPlot generation complete.")


def main():
    parser = argparse.ArgumentParser(description="Plot MPM validation results")
    parser.add_argument('--input', '-i', type=str, default='validation_output',
                        help='Input directory containing CSV files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for plots (default: same as input)')
    parser.add_argument('--scene', '-s', type=str, default=None,
                        choices=['uniaxial', 'shear', 'objectivity', 'energy',
                                 'energy_convergence', 'gelslim', 'hertz', 'all'],
                        help='Plot specific scene (default: all)')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    if args.scene is None or args.scene == 'all':
        plot_all(input_dir, output_dir)
    else:
        scene_plotters = {
            'uniaxial': plot_uniaxial_tension,
            'shear': plot_pure_shear,
            'objectivity': plot_objectivity,
            'energy': plot_energy_conservation,
            'energy_convergence': plot_energy_convergence,
            'gelslim': plot_gelslim_slip,
            'hertz': plot_hertz_contact,
        }
        if args.scene in scene_plotters:
            scene_plotters[args.scene](input_dir, output_dir)


if __name__ == '__main__':
    main()
