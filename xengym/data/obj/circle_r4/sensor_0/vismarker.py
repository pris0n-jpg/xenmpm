#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marker Displacement Visualization Tool
Visualize and compare real and simulated marker displacement data
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_marker_data(pkl_path):
    """
    Load marker data
    
    Args:
        pkl_path: Path to pkl file
        
    Returns:
        dict: Dictionary containing 'marker_real' and 'marker_sim'
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_original_grid(marker_row_col=[20, 11], marker_dx_dy_mm=[1.31, 1.31]):
    """
    Get original marker grid coordinates (reference: MarkerInterp.py)
    
    Args:
        marker_row_col: Number of marker rows and columns [rows, cols]
        marker_dx_dy_mm: Marker spacing [dx, dy] in mm
        
    Returns:
        ndarray: Grid coordinates with shape (rows, cols, 2)
    """
    marker_row_col = np.array(marker_row_col)
    marker_dx_dy_mm = np.array(marker_dx_dy_mm)
    
    # Calculate grid boundaries
    y_ed, x_ed = marker_dx_dy_mm * (marker_row_col - 1) / 2
    x = np.linspace(-x_ed, x_ed, marker_row_col[1])
    y = np.linspace(-y_ed, y_ed, marker_row_col[0])
    X, Y = np.meshgrid(x, y)
    
    return np.stack([X, Y], axis=2)


def visualize_marker_displacement(pkl_path, save_path=None):
    """
    Visualize marker displacement data, comparing real and simulated results
    
    Args:
        pkl_path: Path to pkl file
        save_path: Path to save the image, if None the image will be displayed
    """
    # Load data
    data = load_marker_data(pkl_path)
    marker_real = data['marker_real']  # shape: (20, 11, 2)
    marker_sim = data['marker_sim']    # shape: (20, 11, 2)
    # marker_sim = -marker_sim[::-1, ::-1, :]  # Ensure only x,y components
    
    # Get original grid
    original_grid = get_original_grid(marker_row_col=[20, 11], marker_dx_dy_mm=[1.31, 1.31])
    
    # Calculate positions after displacement
    real_pos = original_grid + marker_real
    sim_pos = original_grid + marker_sim
    
    # Flatten to 2D arrays for plotting
    original_x = original_grid[:, :, 0].flatten()
    original_y = original_grid[:, :, 1].flatten()
    
    real_x = real_pos[:, :, 0].flatten()
    real_y = real_pos[:, :, 1].flatten()
    
    sim_x = sim_pos[:, :, 0].flatten()
    sim_y = sim_pos[:, :, 1].flatten()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Marker Displacement Visualization Comparison', fontsize=16, fontweight='bold')
    
    # Subplot 1: Real marker displacement
    ax1 = axes[0, 0]
    ax1.scatter(original_x, original_y, c='lightgray', s=30, alpha=0.5, label='Original Position')
    ax1.scatter(real_x, real_y, c='red', s=50, alpha=0.7, label='Real Displaced Position')
    # Draw displacement vectors
    for i in range(len(original_x)):
        ax1.arrow(original_x[i], original_y[i],
                 real_x[i] - original_x[i], real_y[i] - original_y[i],
                 head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('X (mm)', fontsize=12)
    ax1.set_ylabel('Y (mm)', fontsize=12)
    ax1.set_title('Real Marker Displacement', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Subplot 2: Simulated marker displacement
    ax2 = axes[0, 1]
    ax2.scatter(original_x, original_y, c='lightgray', s=30, alpha=0.5, label='Original Position')
    ax2.scatter(sim_x, sim_y, c='blue', s=50, alpha=0.7, label='Sim Displaced Position')
    # Draw displacement vectors
    for i in range(len(original_x)):
        ax2.arrow(original_x[i], original_y[i],
                 sim_x[i] - original_x[i], sim_y[i] - original_y[i],
                 head_width=0.3, head_length=0.2, fc='blue', ec='blue', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('X (mm)', fontsize=12)
    ax2.set_ylabel('Y (mm)', fontsize=12)
    ax2.set_title('Simulated Marker Displacement', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Subplot 3: Overlay comparison
    ax3 = axes[1, 0]
    ax3.scatter(original_x, original_y, c='lightgray', s=30, alpha=0.5, label='Original Position', zorder=1)
    ax3.scatter(real_x, real_y, c='red', s=50, alpha=0.6, label='Real', marker='o', zorder=2)
    ax3.scatter(sim_x, sim_y, c='blue', s=50, alpha=0.6, label='Sim', marker='^', zorder=3)
    ax3.set_xlabel('X (mm)', fontsize=12)
    ax3.set_ylabel('Y (mm)', fontsize=12)
    ax3.set_title('Real vs Sim Overlay Comparison', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Subplot 4: Error analysis (displacement difference)
    ax4 = axes[1, 1]
    diff_x = real_x - sim_x
    diff_y = real_y - sim_y
    diff_magnitude = np.sqrt(diff_x**2 + diff_y**2)
    
    scatter = ax4.scatter(original_x, original_y, c=diff_magnitude, s=100,
                         cmap='hot', alpha=0.8, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Displacement Error Magnitude (mm)', fontsize=10)
    
    # Draw error vectors
    for i in range(len(original_x)):
        if diff_magnitude[i] > 0.001:  # Only show significant errors
            ax4.arrow(original_x[i], original_y[i], diff_x[i], diff_y[i],
                     head_width=0.3, head_length=0.2, fc='purple', ec='purple',
                     alpha=0.5, linewidth=0.8)
    
    ax4.set_xlabel('X (mm)', fontsize=12)
    ax4.set_ylabel('Y (mm)', fontsize=12)
    ax4.set_title(f'Displacement Error Distribution (Mean Error: {diff_magnitude.mean():.4f} mm)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.tight_layout()
    
    # Show then save
    plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    # Print statistics
    print("\n=== Marker Displacement Statistics ===")
    print(f"Marker Grid Size: {marker_real.shape[0]} x {marker_real.shape[1]}")
    print(f"\nReal Displacement Statistics:")
    print(f"  X direction: mean={marker_real[:,:,0].mean():.4f} mm, std={marker_real[:,:,0].std():.4f} mm")
    print(f"  Y direction: mean={marker_real[:,:,1].mean():.4f} mm, std={marker_real[:,:,1].std():.4f} mm")
    print(f"  Magnitude: max={np.sqrt(marker_real[:,:,0]**2 + marker_real[:,:,1]**2).max():.4f} mm")
    
    print(f"\nSimulated Displacement Statistics:")
    print(f"  X direction: mean={marker_sim[:,:,0].mean():.4f} mm, std={marker_sim[:,:,0].std():.4f} mm")
    print(f"  Y direction: mean={marker_sim[:,:,1].mean():.4f} mm, std={marker_sim[:,:,1].std():.4f} mm")
    print(f"  Magnitude: max={np.sqrt(marker_sim[:,:,0]**2 + marker_sim[:,:,1]**2).max():.4f} mm")
    
    print(f"\nDisplacement Error Statistics:")
    print(f"  Mean error: {diff_magnitude.mean():.4f} mm")
    print(f"  Max error: {diff_magnitude.max():.4f} mm")
    print(f"  Error std: {diff_magnitude.std():.4f} mm")


def main():
    """Main function"""
    # Set file paths
    script_dir = Path(__file__).parent
    pkl_path = script_dir / "circle_r4.pkl"
    save_path = script_dir / "marker_visualization.png"
    
    # Check if file exists
    if not pkl_path.exists():
        print(f"Error: Data file not found {pkl_path}")
        return
    
    print(f"Loading data from: {pkl_path}")
    
    # Execute visualization
    visualize_marker_displacement(pkl_path, save_path=save_path)
    
    print("\nVisualization completed!")


if __name__ == "__main__":
    main()