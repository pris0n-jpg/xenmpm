#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三参数贝叶斯优化结果可视化脚本
用于可视化保存的calibration_results_*.json文件
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
from typing import Dict, Optional, Tuple
from scipy.interpolate import griddata
import pandas as pd


class OptimizationVisualizer:
    """贝叶斯优化结果可视化器"""
    
    def __init__(self, results_file: str):
        """
        初始化可视化器
        
        Parameters:
        - results_file: str, 结果JSON文件路径
        """
        self.results_file = Path(results_file)
        if not self.results_file.exists():
            raise FileNotFoundError(f"结果文件不存在: {results_file}")
        
        # 加载结果数据
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # 提取数据
        self.history = self.results['optimization_history']
        self.best_params = self.results['best_params']
        self.best_score = self.results['best_score']
        
        # 提取参数数组
        self.iterations = range(len(self.history))
        self.scores = [h['score'] for h in self.history]
        self.E_values = [h['params'][0] for h in self.history]
        self.nu_values = [h['params'][1] for h in self.history]
        self.coef_values = [h['params'][2] for h in self.history]
        
        # 提取真实值（如果存在）
        self.E_true = self.results.get('true_params', {}).get('E')
        self.nu_true = self.results.get('true_params', {}).get('nu')
        self.coef_true = self.results.get('true_params', {}).get('coef')
        
        print(f"✓ 加载优化结果: {len(self.history)} 次评估")
        print(f"✓ 最优参数: E={self.best_params['E']:.4f}, nu={self.best_params['nu']:.4f}, coef={self.best_params['coef']:.3f}")
        print(f"✓ 最优得分: {self.best_score:.6f}")
    
    def plot_optimization_overview(self, save_path: Optional[str] = None):
        """
        绘制优化概览图 - 与calibration.py第一张图相同的3x3布局
        """
        # 创建3x3布局的图形
        fig = plt.figure(figsize=(18, 15))
        fig.suptitle('Bayesian Optimization Results Overview (3 Parameters)', fontsize=16, fontweight='bold')
        
        # 1. 优化进程
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(self.iterations, self.scores, 'o-', linewidth=2, markersize=4)
        ax1.set_xlabel('Iteration', fontsize=10)
        ax1.set_ylabel('Objective Function Value', fontsize=10)
        ax1.set_title('Optimization Progress', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 参数收敛
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(self.iterations, self.E_values, 'o-', label='Young\'s Modulus E', linewidth=2, markersize=3)
        ax2.plot(self.iterations, self.nu_values, 's-', label='Poisson\'s Ratio ν', linewidth=2, markersize=3)
        ax2.plot(self.iterations, self.coef_values, 'd-', label='Nonlinear Coefficient', linewidth=2, markersize=3)
        
        # 添加真实值线
        if self.E_true is not None:
            ax2.axhline(y=self.E_true, color='red', linestyle='--', linewidth=2, 
                       label=f'True E={self.E_true:.4f}')
        if self.nu_true is not None:
            ax2.axhline(y=self.nu_true, color='blue', linestyle='--', linewidth=2, 
                       label=f'True ν={self.nu_true:.4f}')
        if self.coef_true is not None:
            ax2.axhline(y=self.coef_true, color='green', linestyle='--', linewidth=2, 
                       label=f'True coef={self.coef_true:.3f}')
        
        ax2.set_xlabel('Iteration', fontsize=10)
        ax2.set_ylabel('Parameter Value', fontsize=10)
        ax2.set_title('Parameter Convergence', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. 3D参数空间
        ax3 = plt.subplot(3, 3, 3, projection='3d')
        scatter = ax3.scatter(self.E_values, self.nu_values, self.coef_values, 
                             c=self.scores, cmap='viridis', s=30, alpha=0.7)
        
        # 标记最优点
        best_idx = np.argmin(self.scores)
        ax3.scatter(self.E_values[best_idx], self.nu_values[best_idx], self.coef_values[best_idx],
                   c='red', s=200, marker='*', edgecolors='black', linewidth=2, label='Best Solution')
        
        # 标记真实值
        if self.E_true is not None and self.nu_true is not None and self.coef_true is not None:
            ax3.scatter(self.E_true, self.nu_true, self.coef_true, c='blue', s=150, marker='o',
                       label='True Values', edgecolors='black', linewidths=1.5)
        
        ax3.set_xlabel('Young\'s Modulus E', fontsize=10)
        ax3.set_ylabel('Poisson\'s Ratio ν', fontsize=10)
        ax3.set_zlabel('Nonlinear Coefficient', fontsize=10)
        ax3.set_title('3D Parameter Space', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        
        # 4. E vs nu 投影
        ax4 = plt.subplot(3, 3, 4)
        scatter4 = ax4.scatter(self.E_values, self.nu_values, c=self.scores, cmap='viridis', s=40, alpha=0.7)
        plt.colorbar(scatter4, ax=ax4, label='Score', shrink=0.8)
        ax4.scatter(self.E_values[best_idx], self.nu_values[best_idx], c='red', s=200, 
                   marker='*', edgecolors='black', linewidth=2, label='Best')
        if self.E_true is not None and self.nu_true is not None:
            ax4.scatter(self.E_true, self.nu_true, c='blue', s=150, marker='o', 
                       label='True', edgecolors='black', linewidths=1.5)
        ax4.set_xlabel('Young\'s Modulus E', fontsize=10)
        ax4.set_ylabel('Poisson\'s Ratio ν', fontsize=10)
        ax4.set_title('E vs ν Projection', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. E vs coef 投影
        ax5 = plt.subplot(3, 3, 5)
        scatter5 = ax5.scatter(self.E_values, self.coef_values, c=self.scores, cmap='viridis', s=40, alpha=0.7)
        plt.colorbar(scatter5, ax=ax5, label='Score', shrink=0.8)
        ax5.scatter(self.E_values[best_idx], self.coef_values[best_idx], c='red', s=200, 
                   marker='*', edgecolors='black', linewidth=2, label='Best')
        if self.E_true is not None and self.coef_true is not None:
            ax5.scatter(self.E_true, self.coef_true, c='blue', s=150, marker='o', 
                       label='True', edgecolors='black', linewidths=1.5)
        ax5.set_xlabel('Young\'s Modulus E', fontsize=10)
        ax5.set_ylabel('Nonlinear Coefficient', fontsize=10)
        ax5.set_title('E vs Coef Projection', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. nu vs coef 投影
        ax6 = plt.subplot(3, 3, 6)
        scatter6 = ax6.scatter(self.nu_values, self.coef_values, c=self.scores, cmap='viridis', s=40, alpha=0.7)
        plt.colorbar(scatter6, ax=ax6, label='Score', shrink=0.8)
        ax6.scatter(self.nu_values[best_idx], self.coef_values[best_idx], c='red', s=200, 
                   marker='*', edgecolors='black', linewidth=2, label='Best')
        if self.nu_true is not None and self.coef_true is not None:
            ax6.scatter(self.nu_true, self.coef_true, c='blue', s=150, marker='o', 
                       label='True', edgecolors='black', linewidths=1.5)
        ax6.set_xlabel('Poisson\'s Ratio ν', fontsize=10)
        ax6.set_ylabel('Nonlinear Coefficient', fontsize=10)
        ax6.set_title('ν vs Coef Projection', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. 累积最优得分
        ax7 = plt.subplot(3, 3, 7)
        best_scores = [min(self.scores[:i+1]) for i in range(len(self.scores))]
        ax7.plot(self.iterations, best_scores, 'o-', color='green', linewidth=2, markersize=4)
        ax7.set_xlabel('Iteration', fontsize=10)
        ax7.set_ylabel('Best Score Achieved', fontsize=10)
        ax7.set_title('Cumulative Best Score', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. 参数分布
        ax8 = plt.subplot(3, 3, 8)
        n_bins = min(15, len(self.E_values)//3)
        ax8.hist(self.E_values, bins=n_bins, alpha=0.7, label='E', density=True, color='red')
        ax8.hist(self.nu_values, bins=n_bins, alpha=0.7, label='ν', density=True, color='blue')
        ax8.hist(self.coef_values, bins=n_bins, alpha=0.7, label='coef', density=True, color='green')
        
        # 添加真实值线
        if self.E_true is not None:
            ax8.axvline(x=self.E_true, color='red', linestyle='--', linewidth=2, 
                       label=f'True E={self.E_true:.4f}')
        if self.nu_true is not None:
            ax8.axvline(x=self.nu_true, color='blue', linestyle='--', linewidth=2, 
                       label=f'True ν={self.nu_true:.4f}')
        if self.coef_true is not None:
            ax8.axvline(x=self.coef_true, color='green', linestyle='--', linewidth=2, 
                       label=f'True coef={self.coef_true:.3f}')
        
        ax8.set_xlabel('Parameter Value', fontsize=10)
        ax8.set_ylabel('Density', fontsize=10)
        ax8.set_title('Parameter Distribution', fontsize=12, fontweight='bold')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        
        # 9. 参数相关性热图
        ax9 = plt.subplot(3, 3, 9)
        param_df = pd.DataFrame({
            'E': self.E_values,
            'ν': self.nu_values,
            'coef': self.coef_values,
            'score': self.scores
        })
        correlation_matrix = param_df.corr()
        
        im = ax9.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # 添加相关系数文本
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax9.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax9.set_xticks(range(len(correlation_matrix.columns)))
        ax9.set_yticks(range(len(correlation_matrix.columns)))
        ax9.set_xticklabels(correlation_matrix.columns, fontsize=10)
        ax9.set_yticklabels(correlation_matrix.columns, fontsize=10)
        ax9.set_title('Parameter Correlation Matrix', fontsize=12, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax9, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 优化概览图已保存: {save_path}")
        
        plt.show()
    
    def plot_parameter_gradients(self, save_path: Optional[str] = None, resolution: int = 30):
        """
        绘制参数梯度图 - 三个参数组合的3D曲面图（分别在三个窗口中显示）
        
        Parameters:
        - save_path: str, optional, 保存路径
        - resolution: int, 网格分辨率
        """
        print("📈 生成3D参数曲面图...")
        
        # 创建保存路径的基础名称
        if save_path:
            save_dir = Path(save_path).parent
            base_name = Path(save_path).stem
        
        # 1. E vs ν 3D曲面图和等高线图
        print("  📊 生成 E vs ν 3D曲面图和等高线图...")
        fig1, (ax1_left, ax1_right) = plt.subplots(1, 2, figsize=(16, 6))
        fig1.suptitle('E vs ν Parameter Space Analysis', fontsize=16, fontweight='bold')
        
        # 左侧：等高线图
        self._plot_contour_map(
            ax1_left, self.E_values, self.nu_values, self.scores,
            'Young\'s Modulus E', 'Poisson\'s Ratio ν', 'E vs ν Contour Map',
            self.E_true, self.nu_true, resolution
        )
        
        # 右侧：3D曲面图
        ax1_right = fig1.add_subplot(122, projection='3d')
        self._plot_3d_surface_map(
            ax1_right, self.E_values, self.nu_values, self.scores,
            'Young\'s Modulus E', 'Poisson\'s Ratio ν', 'E vs ν 3D Surface',
            self.E_true, self.nu_true, resolution
        )
        
        if save_path:
            save_path1 = save_dir / f"{base_name}_E_vs_nu.png"
            plt.savefig(save_path1, dpi=300, bbox_inches='tight')
            print(f"    ✓ E vs ν 曲面图已保存: {save_path1}")
        
        plt.show()
        
        # 2. E vs coef 3D曲面图和等高线图
        print("  📊 生成 E vs coef 3D曲面图和等高线图...")
        fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(16, 6))
        fig2.suptitle('E vs Coef Parameter Space Analysis', fontsize=16, fontweight='bold')
        
        # 左侧：等高线图
        self._plot_contour_map(
            ax2_left, self.E_values, self.coef_values, self.scores,
            'Young\'s Modulus E', 'Nonlinear Coefficient', 'E vs Coef Contour Map',
            self.E_true, self.coef_true, resolution
        )
        
        # 右侧：3D曲面图
        ax2_right = fig2.add_subplot(122, projection='3d')
        self._plot_3d_surface_map(
            ax2_right, self.E_values, self.coef_values, self.scores,
            'Young\'s Modulus E', 'Nonlinear Coefficient', 'E vs Coef 3D Surface',
            self.E_true, self.coef_true, resolution
        )
        
        if save_path:
            save_path2 = save_dir / f"{base_name}_E_vs_coef.png"
            plt.savefig(save_path2, dpi=300, bbox_inches='tight')
            print(f"    ✓ E vs coef 曲面图已保存: {save_path2}")
        
        plt.show()
        
        # 3. ν vs coef 3D曲面图和等高线图
        print("  📊 生成 ν vs coef 3D曲面图和等高线图...")
        fig3, (ax3_left, ax3_right) = plt.subplots(1, 2, figsize=(16, 6))
        fig3.suptitle('ν vs Coef Parameter Space Analysis', fontsize=16, fontweight='bold')
        
        # 左侧：等高线图
        self._plot_contour_map(
            ax3_left, self.nu_values, self.coef_values, self.scores,
            'Poisson\'s Ratio ν', 'Nonlinear Coefficient', 'ν vs Coef Contour Map',
            self.nu_true, self.coef_true, resolution
        )
        
        # 右侧：3D曲面图
        ax3_right = fig3.add_subplot(122, projection='3d')
        self._plot_3d_surface_map(
            ax3_right, self.nu_values, self.coef_values, self.scores,
            'Poisson\'s Ratio ν', 'Nonlinear Coefficient', 'ν vs Coef 3D Surface',
            self.nu_true, self.coef_true, resolution
        )
        
        if save_path:
            save_path3 = save_dir / f"{base_name}_nu_vs_coef.png"
            plt.savefig(save_path3, dpi=300, bbox_inches='tight')
            print(f"    ✓ ν vs coef 曲面图已保存: {save_path3}")
        
        plt.show()
        
        print("✅ 所有3D参数曲面图生成完成！")
    
    def _plot_contour_map(self, ax, x_vals, y_vals, z_vals, xlabel, ylabel, title, 
                         x_true=None, y_true=None, resolution=30):
        """
        绘制等高线图（深度场图）
        
        Parameters:
        - ax: matplotlib轴对象
        - x_vals, y_vals, z_vals: 参数和目标函数值
        - xlabel, ylabel, title: 标签和标题
        - x_true, y_true: 真实值（可选）
        - resolution: 网格分辨率
        """
        # 检查数据点数量
        if len(x_vals) < 4:
            print(f"⚠️ 数据点不足，无法绘制等高线图: {title}")
            return
        
        # 创建网格
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        # 扩展边界以获得更好的可视化
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        xi = np.linspace(x_min, x_max, resolution)
        yi = np.linspace(y_min, y_max, resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # 插值生成等高线场
        points = np.column_stack((x_vals, y_vals))
        try:
            zi_grid = griddata(points, z_vals, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
            
            # 如果线性插值失败，尝试最近邻插值
            if np.all(np.isnan(zi_grid)):
                zi_grid = griddata(points, z_vals, (xi_grid, yi_grid), method='nearest')
        except Exception as e:
            print(f"⚠️ 插值失败: {e}, 使用最近邻插值")
            zi_grid = griddata(points, z_vals, (xi_grid, yi_grid), method='nearest')
        
        # 绘制等高线图
        try:
            # 填充等高线
            contour_filled = ax.contourf(xi_grid, yi_grid, zi_grid, levels=20, cmap='viridis', alpha=0.8)
            
            # 等高线
            contour_lines = ax.contour(xi_grid, yi_grid, zi_grid, levels=10, colors='white', alpha=0.6, linewidths=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=8, colors='white')
            
            # 添加颜色条
            try:
                cbar = plt.colorbar(contour_filled, ax=ax, shrink=0.8)
                cbar.set_label('Objective Function Value', fontsize=10)
            except:
                pass  # 如果颜色条添加失败，忽略
        except Exception as e:
            print(f"⚠️ 等高线绘制失败: {e}")
        
        # 绘制采样点
        scatter = ax.scatter(x_vals, y_vals, c=z_vals, cmap='viridis', s=50, alpha=0.9, 
                            edgecolors='black', linewidths=1)
        
        # 标记最优点
        best_idx = np.argmin(z_vals)
        ax.scatter(x_vals[best_idx], y_vals[best_idx], c='red', s=200, marker='*', 
                  edgecolors='black', linewidth=2, label='Best Solution', zorder=10)
        
        # 标记真实值
        if x_true is not None and y_true is not None:
            ax.scatter(x_true, y_true, c='blue', s=150, marker='o', 
                      label='True Values', edgecolors='black', linewidths=2, zorder=10)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_3d_surface_map(self, ax, x_vals, y_vals, z_vals, xlabel, ylabel, title, 
                            x_true=None, y_true=None, resolution=30):
        """
        绘制单个3D曲面图
        
        Parameters:
        - ax: matplotlib 3D轴对象
        - x_vals, y_vals, z_vals: 参数和目标函数值
        - xlabel, ylabel, title: 标签和标题
        - x_true, y_true: 真实值（可选）
        - resolution: 网格分辨率
        """
        # 检查数据点数量
        if len(x_vals) < 4:
            print(f"⚠️ 数据点不足，无法绘制3D曲面图: {title}")
            return
        
        # 创建网格
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        # 扩展边界以获得更好的可视化
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        xi = np.linspace(x_min, x_max, resolution)
        yi = np.linspace(y_min, y_max, resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # 插值生成3D曲面
        points = np.column_stack((x_vals, y_vals))
        try:
            zi_grid = griddata(points, z_vals, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
            
            # 如果线性插值失败，尝试最近邻插值
            if np.all(np.isnan(zi_grid)):
                zi_grid = griddata(points, z_vals, (xi_grid, yi_grid), method='nearest')
        except Exception as e:
            print(f"⚠️ 插值失败: {e}, 使用最近邻插值")
            zi_grid = griddata(points, z_vals, (xi_grid, yi_grid), method='nearest')
        
        # 绘制3D曲面
        try:
            # 过滤掉NaN值
            mask = ~np.isnan(zi_grid)
            if np.any(mask):
                surf = ax.plot_surface(xi_grid, yi_grid, zi_grid, cmap='viridis', 
                                     alpha=0.7, linewidth=0, antialiased=True)
                
                # 添加颜色条
                try:
                    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
                except:
                    pass  # 如果颜色条添加失败，忽略
        except Exception as e:
            print(f"⚠️ 3D曲面绘制失败: {e}")
        
        # 绘制采样点
        ax.scatter(x_vals, y_vals, z_vals, c='red', s=50, alpha=1.0, edgecolors='black', linewidths=1)
        
        # 标记最优点
        best_idx = np.argmin(z_vals)
        ax.scatter(x_vals[best_idx], y_vals[best_idx], z_vals[best_idx], 
                  c='yellow', s=200, marker='*', edgecolors='black', linewidth=2, 
                  label='Best Solution')
        
        # 标记真实值
        if x_true is not None and y_true is not None:
            # 通过插值计算真实值对应的目标函数值
            try:
                true_z = griddata(points, z_vals, np.array([[x_true, y_true]]), method='linear')[0]
                if np.isnan(true_z):
                    true_z = griddata(points, z_vals, np.array([[x_true, y_true]]), method='nearest')[0]
                
                ax.scatter(x_true, y_true, true_z, c='blue', s=150, marker='o', 
                          edgecolors='black', linewidths=2, label='True Values')
            except:
                # 如果插值失败，使用平均值
                true_z = np.mean(z_vals)
                ax.scatter(x_true, y_true, true_z, c='blue', s=150, marker='o', 
                          edgecolors='black', linewidths=2, label='True Values')
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_zlabel('Objective Function Value', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 添加图例
        try:
            ax.legend(fontsize=8)
        except:
            pass
    
    def save_summary_report(self, save_path: Optional[str] = None):
        """保存优化结果摘要报告"""
        if save_path is None:
            save_path = self.results_file.parent / f"summary_{self.results_file.stem}.txt"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("贝叶斯优化结果摘要报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"结果文件: {self.results_file.name}\n")
            f.write(f"评估次数: {len(self.history)}\n")
            f.write(f"优化时间: {self.results.get('timestamp', 'N/A')}\n\n")
            
            f.write("最优参数:\n")
            f.write(f"  杨氏模量 E = {self.best_params['E']:.4f}\n")
            f.write(f"  泊松比 ν = {self.best_params['nu']:.4f}\n")
            f.write(f"  非线性系数 coef = {self.best_params['coef']:.3f}\n")
            f.write(f"  最小误差 = {self.best_score:.6f}\n\n")
            
            if self.E_true is not None and self.nu_true is not None:
                f.write("真实参数:\n")
                f.write(f"  杨氏模量 E = {self.E_true:.4f}\n")
                f.write(f"  泊松比 ν = {self.nu_true:.4f}\n")
                if self.coef_true is not None:
                    f.write(f"  非线性系数 coef = {self.coef_true:.3f}\n")
                
                f.write("\n参数误差:\n")
                E_error = abs(self.best_params['E'] - self.E_true)
                nu_error = abs(self.best_params['nu'] - self.nu_true)
                f.write(f"  E误差 = {E_error:.4f} ({100*E_error/self.E_true:.1f}%)\n")
                f.write(f"  ν误差 = {nu_error:.4f} ({100*nu_error/self.nu_true:.1f}%)\n")
                if self.coef_true is not None:
                    coef_error = abs(self.best_params['coef'] - self.coef_true)
                    f.write(f"  coef误差 = {coef_error:.3f} ({100*coef_error/self.coef_true:.1f}%)\n")
            
            f.write("\n统计信息:\n")
            f.write(f"  得分均值 = {np.mean(self.scores):.6f}\n")
            f.write(f"  得分标准差 = {np.std(self.scores):.6f}\n")
            f.write(f"  得分范围 = [{min(self.scores):.6f}, {max(self.scores):.6f}]\n")
            f.write(f"  改进幅度 = {self.scores[0] - self.best_score:.6f}\n")
        
        print(f"✓ 摘要报告已保存: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='三参数贝叶斯优化结果可视化工具')
    parser.add_argument('--results_file', default=None, help='结果JSON文件路径')
    parser.add_argument('--save-plots', action='store_true', help='保存图表到文件')
    parser.add_argument('--resolution', type=int, default=50, help='梯度图分辨率 (默认: 50)')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录 (默认: 与结果文件同目录)')
    
    args = parser.parse_args()
    
    # 如果未指定文件，自动查找最新的结果文件
    if args.results_file is None:
        print("🔍 未指定结果文件，自动查找最新的结果文件...")
        
        # 可能的搜索路径
        search_patterns = [
            "calibration_results_*.json",                           # 当前目录
            "calibration/calibration_results_*.json",               # calibration子目录
            "../calibration/calibration_results_*.json",            # 上级目录的calibration子目录
            "results/calibration_results_*.json",                   # results目录
            "calibration/results/calibration_results_*.json",       # calibration/results子目录
        ]
        
        found_files = []
        for pattern in search_patterns:
            import glob
            files = glob.glob(pattern)
            if files:
                # 按修改时间排序，获取最新的文件
                files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                found_files.extend(files)
                print(f"  在 {pattern} 中找到 {len(files)} 个文件")
        
        if found_files:
            # 选择最新的文件
            args.results_file = found_files[0]
            print(f"✓ 使用最新的结果文件: {args.results_file}")
        else:
            print("❌ 未找到任何结果文件")
            print("请确保标定脚本已运行并生成了结果文件，或使用 --results_file 指定文件路径")
            return
    
    # 检查文件是否存在
    if not Path(args.results_file).exists():
        print(f"❌ 结果文件不存在: {args.results_file}")
        return
    
    # 设置输出目录
    results_path = Path(args.results_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = results_path.parent
    
    # 创建可视化器
    try:
        visualizer = OptimizationVisualizer(args.results_file)
    except Exception as e:
        print(f"❌ 加载结果文件失败: {e}")
        return
    
    print(f"\n🎨 开始生成可视化图表...")
    
    # 生成图表
    if args.save_plots:
        base_name = results_path.stem
        overview_path = output_dir / f"{base_name}_overview.png"
        gradient_path = output_dir / f"{base_name}_gradients.png"
        
        print("📊 生成优化概览图...")
        visualizer.plot_optimization_overview(save_path=str(overview_path))
        
        print("📈 生成参数3D曲面图...")
        visualizer.plot_parameter_gradients(save_path=str(gradient_path), resolution=args.resolution)
        
        print("📝 生成摘要报告...")
        visualizer.save_summary_report()
        
        print(f"\n✅ 所有图表已保存至: {output_dir}")
    else:
        print("📊 显示优化概览图...")
        visualizer.plot_optimization_overview()
        
        print("📈 显示参数3D曲面图...")
        visualizer.plot_parameter_gradients(resolution=args.resolution)
    
    print("\n🎉 可视化完成！")


if __name__ == '__main__':
    main() 