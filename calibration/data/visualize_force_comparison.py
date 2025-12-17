#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Force and Marker Comparison Visualization Tool
Visualizes and compares force_xyz[2] (Z-force) and marker displacement
between real and simulated data for different objects and trajectories.

Features:
- Dual visualization modes: Force comparison and Marker comparison
- Interactive selection: object, trajectory, step
- Coordinate system validation and visualization
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, List, Tuple, Optional
import sys

try:
    from scipy.interpolate import UnivariateSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available, polynomial fitting will be used instead")


class ForceAndMarkerVisualizer:
    def __init__(self, real_data_path: str, sim_data_path: str):
        """
        Initialize the force and marker comparison visualizer
        
        Parameters:
        - real_data_path: Path to real_fit_data.pkl
        - sim_data_path: Path to sim_data.pkl
        """
        self.real_data = self.load_data(real_data_path)
        self.sim_data = self.load_data(sim_data_path)
        
        # Get union of objects
        self.objects = sorted(list(set(self.real_data.keys()) | set(self.sim_data.keys())))
        
        if not self.objects:
            raise ValueError("No objects found in either real or sim data")
        
        self.current_object = self.objects[0]
        self.current_trajectory = None
        self.current_step = 0
        self.trajectories = {}
        
        # Visualization mode: 'force' or 'marker'
        self.view_mode = 'force'
        
        # Marker grid parameters (standard size)
        self.marker_row_col = [20, 11]
        self.marker_dx_dy_mm = [1.31, 1.31]
        
        # Build trajectory dictionary
        for obj in self.objects:
            real_trajs = set(self.real_data.get(obj, {}).keys())
            sim_trajs = set(self.sim_data.get(obj, {}).keys())
            all_trajs = sorted(list(real_trajs | sim_trajs))
            self.trajectories[obj] = all_trajs
        
        self.current_trajectory = self.trajectories[self.current_object][0] if self.trajectories[self.current_object] else None
        
        # Setup the plot
        self.setup_plot()
        
    def load_data(self, file_path: str) -> Dict:
        """Load pickle data file"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def get_grid_for_shape(self, rows: int, cols: int):
        """
        Get marker grid coordinates for specific shape
        
        Grid layout:
        - Origin at center (0, 0)
        - X-axis: horizontal (left to right)
        - Y-axis: vertical (bottom to top)
        - Grid spans from negative to positive coordinates
        """
        marker_dx_dy_mm = np.array(self.marker_dx_dy_mm)
        
        # Calculate grid extent (centered at origin)
        y_extent = marker_dx_dy_mm[1] * (rows - 1) / 2
        x_extent = marker_dx_dy_mm[0] * (cols - 1) / 2
        
        # Create coordinate arrays
        x = np.linspace(-x_extent, x_extent, cols)
        y = np.linspace(-y_extent, y_extent, rows)
        X, Y = np.meshgrid(x, y)
        
        return np.stack([X, Y], axis=2)
    
    def extract_force_z(self, data: Dict, obj: str, traj: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract force_xyz[2] values and step numbers"""
        if obj not in data or traj not in data[obj]:
            return np.array([]), np.array([])
        
        traj_data = data[obj][traj]
        steps = []
        forces = []
        
        for step_key in sorted(traj_data.keys()):
            step_data = traj_data[step_key]
            if 'force_xyz' in step_data:
                force_z = step_data['force_xyz'][2]
                step_num = int(step_key.split('_')[-1])
                
                if step_num <= 9:
                    steps.append(step_num)
                    forces.append(float(force_z))
        
        if len(steps) > 0:
            # Add step 0 with 0 force
            steps = np.array([0] + [s + 1 for s in steps])
            forces = np.array([0.0] + forces)
            return steps, forces
        else:
            return np.array([]), np.array([])
    
    def extract_marker_displacement(self, data: Dict, obj: str, traj: str, step: int) -> Optional[np.ndarray]:
        """Extract marker displacement for a specific step"""
        if obj not in data or traj not in data[obj]:
            return None
        
        traj_data = data[obj][traj]
        step_key = f"step_{step:03d}"
        
        if step_key not in traj_data or 'marker_displacement' not in traj_data[step_key]:
            return None
        
        return traj_data[step_key]['marker_displacement']
    
    def fit_curve(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a smooth curve through the data points"""
        if len(x) < 2:
            return x, y
        
        x_dense = np.linspace(x.min(), x.max(), 200)
        
        if SCIPY_AVAILABLE and len(x) >= 4:
            try:
                k = min(3, len(x) - 1)
                s = len(x) * 0.1
                spline = UnivariateSpline(x, y, k=k, s=s)
                y_fitted = spline(x_dense)
                return x_dense, y_fitted
            except:
                pass
        
        try:
            degree = min(3, len(x) - 1)
            poly = np.polyfit(x, y, degree)
            y_fitted = np.polyval(poly, x_dense)
            return x_dense, y_fitted
        except:
            return x, y
    
    def setup_plot(self):
        """Setup the matplotlib figure and widgets"""
        self.fig = plt.figure(figsize=(16, 9), constrained_layout=False)
        self.fig.canvas.manager.set_window_title('Force & Marker Comparison: Real vs Simulation')
        
        # Main content area (will be redrawn based on mode)
        self.main_axes = []
        
        # Left panel - Selection controls
        self.ax_object = plt.subplot2grid((4, 5), (0, 0), rowspan=1)
        self.ax_object.set_title('Object', fontsize=10, fontweight='bold')
        
        self.ax_trajectory = plt.subplot2grid((4, 5), (1, 0), rowspan=1)
        self.ax_trajectory.set_title('Trajectory', fontsize=10, fontweight='bold')
        
        self.ax_step = plt.subplot2grid((4, 5), (2, 0), rowspan=1)
        self.ax_step.set_title('Step', fontsize=10, fontweight='bold')
        
        # Mode switch button
        self.ax_mode_button = plt.subplot2grid((4, 5), (3, 0), rowspan=1)
        self.ax_mode_button.set_title('View Mode', fontsize=10, fontweight='bold')
        
        # Create radio buttons
        self.radio_object = RadioButtons(self.ax_object, self.objects)
        self.radio_object.on_clicked(self.on_object_change)
        
        # Mode switch button
        self.radio_mode = RadioButtons(self.ax_mode_button, ['Force', 'Marker'])
        self.radio_mode.on_clicked(self.on_mode_change)
        
        self.update_trajectory_buttons()
        self.update_step_buttons()
        
        # Draw initial plot
        self.redraw_main_content()
        
        plt.tight_layout(pad=2.0)
    
    def update_trajectory_buttons(self):
        """Update trajectory radio buttons"""
        self.ax_trajectory.clear()
        self.ax_trajectory.set_title('Trajectory', fontsize=10, fontweight='bold')
        
        trajs = self.trajectories[self.current_object]
        if trajs:
            self.radio_trajectory = RadioButtons(self.ax_trajectory, trajs)
            self.radio_trajectory.on_clicked(self.on_trajectory_change)
            if self.current_trajectory not in trajs:
                self.current_trajectory = trajs[0]
        else:
            self.ax_trajectory.text(0.5, 0.5, 'No trajectories', 
                                   ha='center', va='center', transform=self.ax_trajectory.transAxes)
            self.current_trajectory = None
    
    def update_step_buttons(self):
        """Update step selection buttons"""
        self.ax_step.clear()
        self.ax_step.set_title('Step', fontsize=10, fontweight='bold')
        
        if self.current_trajectory:
            steps = []
            if self.current_object in self.real_data and self.current_trajectory in self.real_data[self.current_object]:
                steps.extend(self.real_data[self.current_object][self.current_trajectory].keys())
            if self.current_object in self.sim_data and self.current_trajectory in self.sim_data[self.current_object]:
                steps.extend(self.sim_data[self.current_object][self.current_trajectory].keys())
            
            if steps:
                step_nums = sorted(set(int(s.split('_')[-1]) for s in steps if s.startswith('step_')))
                step_labels = [f"step_{s}" for s in step_nums[:10]]
                
                if step_labels:
                    self.radio_step = RadioButtons(self.ax_step, step_labels)
                    self.radio_step.on_clicked(self.on_step_change)
                    return
        
        self.ax_step.text(0.5, 0.5, 'No steps', 
                         ha='center', va='center', transform=self.ax_step.transAxes)
    
    def on_object_change(self, label):
        """Callback when object selection changes"""
        self.current_object = label
        self.update_trajectory_buttons()
        self.update_step_buttons()
        self.redraw_main_content()
        plt.draw()
    
    def on_trajectory_change(self, label):
        """Callback when trajectory selection changes"""
        self.current_trajectory = label
        self.update_step_buttons()
        self.redraw_main_content()
        plt.draw()
    
    def on_step_change(self, label):
        """Callback when step selection changes"""
        self.current_step = int(label.split('_')[-1])
        self.redraw_main_content()
        plt.draw()
    
    def on_mode_change(self, label):
        """Callback when view mode changes"""
        self.view_mode = 'force' if label == 'Force' else 'marker'
        self.redraw_main_content()
        plt.draw()
    
    def redraw_main_content(self):
        """Redraw main content area based on current mode"""
        # Clear existing main axes
        for ax in self.main_axes:
            self.fig.delaxes(ax)
        self.main_axes = []
        
        if self.view_mode == 'force':
            self.draw_force_mode()
        else:
            self.draw_marker_mode()
    
    def draw_force_mode(self):
        """Draw force comparison mode"""
        # Large force plot
        ax_force = plt.subplot2grid((4, 5), (0, 1), colspan=4, rowspan=4)
        self.main_axes.append(ax_force)
        
        self.update_force_plot(ax_force)
    
    def draw_marker_mode(self):
        """Draw marker comparison mode"""
        # 手动创建三个完全相同大小的axes
        # 定义统一的axes尺寸
        ax_width = 0.18   # 每个图宽度（缩小为error图的colorbar留空间）
        ax_height = 0.55  # 每个图高度
        ax_bottom = 0.35  # 底部位置
        spacing = 0.05    # 图之间的间距
        
        # Real图
        ax_real = self.fig.add_axes([0.25, ax_bottom, ax_width, ax_height])
        
        # Sim图
        ax_sim = self.fig.add_axes([0.25 + ax_width + spacing, ax_bottom, ax_width, ax_height])
        
        # Error图 - 完全相同的尺寸
        ax_error = self.fig.add_axes([0.25 + 2*(ax_width + spacing), ax_bottom, ax_width, ax_height])
        
        # Info panel
        ax_info = self.fig.add_axes([0.25, 0.05, 0.65, 0.2])
        ax_info.axis('off')
        
        self.main_axes.extend([ax_real, ax_sim, ax_error, ax_info])
        
        self.update_marker_plot(ax_real, ax_sim, ax_error, ax_info)
    
    def update_force_plot(self, ax):
        """Update force comparison plot"""
        ax.clear()
        
        if self.current_trajectory is None:
            ax.text(0.5, 0.5, 'No trajectory selected', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, fontweight='bold')
            return
        
        try:
            real_steps, real_forces = self.extract_force_z(
                self.real_data, self.current_object, self.current_trajectory
            )
            sim_steps, sim_forces = self.extract_force_z(
                self.sim_data, self.current_object, self.current_trajectory
            )
            
            if len(real_steps) == 0 and len(sim_steps) == 0:
                ax.text(0.5, 0.5, 'No force data available',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=14, fontweight='bold')
                return
            
            # Plot real data
            if len(real_steps) > 0:
                ax.scatter(real_steps, real_forces, 
                          c='#2196F3', s=120, alpha=0.7, 
                          edgecolors='#1565C0', linewidth=2,
                          label='Real Data', zorder=3)
                
                if len(real_steps) >= 2:
                    x_fit, y_fit = self.fit_curve(real_steps, real_forces)
                    ax.plot(x_fit, y_fit, '#2196F3', 
                            linewidth=3, alpha=0.8,
                            label='Real Fitted')
            
            # Plot sim data
            if len(sim_steps) > 0:
                ax.scatter(sim_steps, sim_forces, 
                          c='#F44336', s=120, alpha=0.7, marker='s',
                          edgecolors='#C62828', linewidth=2,
                          label='Sim Data', zorder=3)
                
                if len(sim_steps) >= 2:
                    x_fit, y_fit = self.fit_curve(sim_steps, sim_forces)
                    ax.plot(x_fit, y_fit, '#F44336', 
                            linewidth=3, alpha=0.8, linestyle='--',
                            label='Sim Fitted')
            
            # Mark current step
            if self.current_step in real_steps:
                idx = np.where(real_steps == self.current_step)[0][0]
                ax.scatter([real_steps[idx]], [real_forces[idx]], 
                          c='#FFD700', s=300, marker='*', 
                          edgecolors='#FF6F00', linewidth=3, zorder=5,
                          label=f'Current (step_{self.current_step})')
            if self.current_step in sim_steps:
                idx = np.where(sim_steps == self.current_step)[0][0]
                ax.scatter([sim_steps[idx]], [sim_forces[idx]], 
                          c='#FFD700', s=300, marker='*', 
                          edgecolors='#FF6F00', linewidth=3, zorder=5)
            
            ax.set_xlabel('Step Number', fontsize=13, fontweight='bold')
            ax.set_ylabel('Z-Force (N)', fontsize=13, fontweight='bold')
            ax.set_title(
                f'Force Comparison: {self.current_object} / {self.current_trajectory}',
                fontsize=14, fontweight='bold', pad=15
            )
            ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            
            # Set axis limits
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(min(0, y_min), y_max * 1.1)
            x_min, x_max = ax.get_xlim()
            ax.set_xlim(-0.5, x_max)
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='red')
            print(f"Force plot error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_marker_plot(self, ax_real, ax_sim, ax_error, ax_info):
        """Update marker displacement visualization"""
        # 清除所有旧的 colorbar axes（避免叠加）
        fig = ax_real.figure
        # 获取当前所有axes，找出colorbar相关的并删除
        for ax_obj in fig.axes[:]:
            # colorbar的axes通常很窄，且不在main_axes列表中
            if ax_obj not in self.main_axes and ax_obj not in [self.ax_object, self.ax_trajectory, self.ax_step, self.ax_mode_button]:
                # 这可能是colorbar的axes
                if hasattr(ax_obj, 'get_position'):
                    bbox = ax_obj.get_position()
                    # colorbar通常宽度很小（< 0.05）
                    if bbox.width < 0.05:
                        fig.delaxes(ax_obj)
        
        ax_real.clear()
        ax_sim.clear()
        ax_error.clear()
        ax_info.clear()
        ax_info.axis('off')
        
        if self.current_trajectory is None:
            ax_info.text(0.5, 0.5, 'No trajectory selected',
                        ha='center', va='center', fontsize=12, fontweight='bold')
            return
        
        try:
            # Extract marker data
            marker_real = self.extract_marker_displacement(
                self.real_data, self.current_object, self.current_trajectory, self.current_step
            )
            marker_sim = self.extract_marker_displacement(
                self.sim_data, self.current_object, self.current_trajectory, self.current_step
            )
            
            if marker_real is None and marker_sim is None:
                for ax in [ax_real, ax_sim, ax_error]:
                    ax.text(0.5, 0.5, 'No marker data',
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax_info.text(0.5, 0.5, 'No marker data available',
                            ha='center', va='center', fontsize=12, color='orange')
                return
            
            # Collect info
            info_lines = []
            info_lines.append(f'Object: {self.current_object}')
            info_lines.append(f'Trajectory: {self.current_trajectory}')
            info_lines.append(f'Step: {self.current_step}')
            
            # Plot real marker
            if marker_real is not None:
                real_rows, real_cols = marker_real.shape[0], marker_real.shape[1]
                real_grid = self.get_grid_for_shape(real_rows, real_cols)
                self.plot_marker(ax_real, real_grid, marker_real, 'Real', '#F44336')
                info_lines.append(f'Real shape: {marker_real.shape}')
            else:
                ax_real.text(0.5, 0.5, 'No real data',
                            ha='center', va='center', transform=ax_real.transAxes, fontsize=10)
            
            # Plot sim marker
            if marker_sim is not None:
                sim_rows, sim_cols = marker_sim.shape[0], marker_sim.shape[1]
                sim_grid = self.get_grid_for_shape(sim_rows, sim_cols)
                self.plot_marker(ax_sim, sim_grid, marker_sim, 'Sim', '#2196F3')
                info_lines.append(f'Sim shape: {marker_sim.shape}')
            else:
                ax_sim.text(0.5, 0.5, 'No sim data',
                           ha='center', va='center', transform=ax_sim.transAxes, fontsize=10)
            
            # Plot error - only when both have same shape
            if marker_real is not None and marker_sim is not None:
                if marker_real.shape == marker_sim.shape:
                    real_grid = self.get_grid_for_shape(marker_real.shape[0], marker_real.shape[1])
                    self.plot_marker_error(ax_error, real_grid, marker_real, marker_sim)
                    
                    diff = marker_real - marker_sim
                    diff_magnitude = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)
                    info_lines.append(f'Mean error: {diff_magnitude.mean():.3f} mm')
                    info_lines.append(f'Max error: {diff_magnitude.max():.3f} mm')
                else:
                    ax_error.text(0.5, 0.5,
                                 f'Shape mismatch\nReal: {marker_real.shape}\nSim: {marker_sim.shape}',
                                 ha='center', va='center', transform=ax_error.transAxes,
                                 fontsize=10, color='orange', fontweight='bold')
                    info_lines.append('⚠️ Shape mismatch - cannot compute error')
            else:
                ax_error.text(0.5, 0.5, 'Need both datasets',
                             ha='center', va='center', transform=ax_error.transAxes, fontsize=10)
            
            # Display info
            info_text = '\n'.join(info_lines)
            ax_info.text(0.05, 0.5, info_text, ha='left', va='center', fontsize=10, 
                        family='monospace', transform=ax_info.transAxes)
        
        except Exception as e:
            print(f"Marker plot error: {e}")
            import traceback
            traceback.print_exc()
            ax_info.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                        fontsize=11, color='red')
    
    def plot_marker(self, ax, original_grid, marker_disp, title, color):
        """Plot marker displacement with arrows"""
        orig_x = original_grid[:, :, 0].flatten()
        orig_y = original_grid[:, :, 1].flatten()
        disp_x = marker_disp[:, :, 0].flatten()
        disp_y = marker_disp[:, :, 1].flatten()
        
        # 计算位移的最大值用于缩放
        disp_magnitude = np.sqrt(disp_x**2 + disp_y**2)
        max_disp = np.max(disp_magnitude) if np.max(disp_magnitude) > 0 else 1.0
        
        # 箭头缩放因子：使最大位移的箭头长度约为3mm
        arrow_scale = 3.0 / max_disp if max_disp > 0.5 else 5.0
        
        # 淡化显示原始marker位置（更小、更透明）
        ax.scatter(orig_x, orig_y, c='lightgray', s=8, alpha=0.2, label='Original', zorder=1)
        
        # 绘制所有位移箭头（缩放后）
        n_points = len(orig_x)
        for i in range(n_points):
            dx_scaled = disp_x[i] * arrow_scale
            dy_scaled = disp_y[i] * arrow_scale
            
            # 只绘制有明显位移的箭头
            if abs(disp_x[i]) > 0.005 or abs(disp_y[i]) > 0.005:
                ax.arrow(orig_x[i], orig_y[i], dx_scaled, dy_scaled,
                        head_width=0.4, head_length=0.3, fc=color, ec=color,
                        alpha=0.7, linewidth=1.2, zorder=2,
                        length_includes_head=True)
        
        ax.set_xlabel('X (mm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=10, fontweight='bold')
        
        # 标题中显示缩放信息
        ax.set_title(f'{title} (×{arrow_scale:.1f})', fontsize=11, fontweight='bold')
        
        # 图例显示缩放信息
        legend_labels = [
            'Original',
            f'Displacement (×{arrow_scale:.1f})'
        ]
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                      markersize=4, alpha=0.5),
            plt.Line2D([0], [0], marker='>', color=color, markersize=8, alpha=0.7, linewidth=0)
        ]
        ax.legend(legend_handles, legend_labels, fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # 固定坐标轴范围（基于标准20×11网格）
        # X轴范围: 11列 × 1.31mm spacing = 13.1mm 总宽度
        # Y轴范围: 20行 × 1.31mm spacing = 24.9mm 总高度
        # 添加10%边距用于显示位移
        x_margin = 13.1 * 0.6 / 2  # 左右各留60%边距
        y_margin = 24.9 * 0.6 / 2  # 上下各留60%边距
        ax.set_xlim(-13.1/2 - x_margin, 13.1/2 + x_margin)
        ax.set_ylim(-24.9/2 - y_margin, 24.9/2 + y_margin)
        ax.set_aspect('equal', adjustable='box')
        
        # Add coordinate system indicator
        ax.annotate('', xy=(0.9, 0.1), xytext=(0.8, 0.1),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                   annotation_clip=False)
        ax.text(0.85, 0.05, 'X', transform=ax.transAxes, fontsize=9, ha='center')
        
        ax.annotate('', xy=(0.8, 0.2), xytext=(0.8, 0.1),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                   annotation_clip=False)
        ax.text(0.75, 0.15, 'Y', transform=ax.transAxes, fontsize=9, ha='center')
    
    def plot_marker_error(self, ax, original_grid, marker_real, marker_sim):
        """Plot marker displacement error with arrows and heatmap"""
        # 计算误差向量
        diff = marker_real - marker_sim
        diff_magnitude = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)
        
        orig_x = original_grid[:, :, 0].flatten()
        orig_y = original_grid[:, :, 1].flatten()
        diff_x = diff[:, :, 0].flatten()
        diff_y = diff[:, :, 1].flatten()
        diff_mag_flat = diff_magnitude.flatten()
        
        # 计算误差的最大值用于缩放
        max_error = np.max(diff_mag_flat) if np.max(diff_mag_flat) > 0 else 1.0
        
        # 箭头缩放因子：使最大误差的箭头长度约为3mm
        arrow_scale = 3.0 / max_error if max_error > 0.5 else 5.0
        
        # 背景热力图：使用原始位置绘制误差大小
        scatter = ax.scatter(orig_x, orig_y, c=diff_mag_flat, s=50,
                           cmap='hot', alpha=0.6, edgecolors='none', zorder=1)
        
        # 创建独立的colorbar axes，完全不影响主axes
        # 获取error axes的位置
        pos = ax.get_position()
        # 在右侧创建colorbar axes（调整位置确保在可见范围内）
        cbar_ax = self.fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
        cbar = plt.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Error (mm)', fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
        
        # 绘制误差向量箭头
        n_points = len(orig_x)
        for i in range(n_points):
            dx_scaled = diff_x[i] * arrow_scale
            dy_scaled = diff_y[i] * arrow_scale
            
            # 只绘制明显的误差
            if abs(diff_x[i]) > 0.005 or abs(diff_y[i]) > 0.005:
                ax.arrow(orig_x[i], orig_y[i], dx_scaled, dy_scaled,
                        head_width=0.4, head_length=0.3,
                        fc='purple', ec='purple',
                        alpha=0.8, linewidth=1.2, zorder=2,
                        length_includes_head=True)
        
        ax.set_xlabel('X (mm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=10, fontweight='bold')
        ax.set_title(f'Error (×{arrow_scale:.1f})', fontsize=11, fontweight='bold')
        
        # 简化图例，避免与colorbar重叠
        # 只显示箭头说明，热力图已由colorbar说明
        legend_handle = plt.Line2D([0], [0], marker='>', color='purple',
                                   markersize=8, alpha=0.8, linewidth=0)
        ax.legend([legend_handle], [f'Error Vector (×{arrow_scale:.1f})'],
                 fontsize=9, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # 固定坐标轴范围（与plot_marker保持一致）
        x_margin = 13.1 * 0.6 / 2
        y_margin = 24.9 * 0.6 / 2
        ax.set_xlim(-13.1/2 - x_margin, 13.1/2 + x_margin)
        ax.set_ylim(-24.9/2 - y_margin, 24.9/2 + y_margin)
        ax.set_aspect('equal', adjustable='box')
    
    def show(self):
        """Display the interactive plot"""
        plt.show()


def main():
    """Main function"""
    print("🎯 Force & Marker Comparison Visualization Tool")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    
    real_data_path = script_dir / "real_fit_data.pkl"
    sim_data_path = script_dir / "sim_data.pkl"
    
    if not real_data_path.exists():
        print(f"❌ Error: real_fit_data.pkl not found at {real_data_path}")
        sys.exit(1)
    
    if not sim_data_path.exists():
        print(f"❌ Error: sim_data.pkl not found at {sim_data_path}")
        sys.exit(1)
    
    print(f"✓ Loading real data from: {real_data_path}")
    print(f"✓ Loading sim data from: {sim_data_path}")
    
    try:
        visualizer = ForceAndMarkerVisualizer(str(real_data_path), str(sim_data_path))
        
        print(f"\n📊 Visualization ready!")
        print(f"   Objects: {', '.join(visualizer.objects)}")
        print(f"   Total trajectories: {sum(len(v) for v in visualizer.trajectories.values())}")
        print(f"\n💡 Usage Instructions:")
        print(f"   - Left panel: Select object, trajectory, step, and view mode")
        print(f"   - Force Mode: Compare Z-force curves (Real vs Sim)")
        print(f"   - Marker Mode: Compare marker displacement (Real | Sim | Error)")
        print(f"   - Coordinate system: X=horizontal, Y=vertical, origin at center")
        print(f"\n🖥️  Opening visualization window...")
        
        visualizer.show()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()