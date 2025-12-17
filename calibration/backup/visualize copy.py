#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization tool module
For visualization analysis and plotting of Bayesian optimization material parameter calibration
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Try to import Chinese display support
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("⚠️ Chinese display support may be incomplete")

class CalibrationVisualizer:
    """Calibration visualization tool"""
    
    def __init__(self, results: Optional[Dict] = None):
        """
        Initialize visualization tool
        
        Args:
            results: Calibration result dictionary, if None then needs to be loaded later
        """
        self.results = results
        self.color_palette = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6A994E',
            'light': '#F4A261'
        }
    
    def load_results(self, results: Dict):
        """Load calibration results"""
        self.results = results
    
    def load_from_file(self, file_path: str):
        """Load calibration results from file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Result file does not exist: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            print(f"✓ Results loaded from file: {file_path}")
            
            # 检查是否存在真实参数
            if 'true_params' in self.results:
                print(f"✓ Found true parameters in results file: E={self.results['true_params']['E']:.4f}, ν={self.results['true_params']['nu']:.4f}")
        except Exception as e:
            raise ValueError(f"Failed to load results: {e}")
    
    def plot_optimization_history(self, show_params: bool = True, save_path: Optional[str] = None):
        """
        Plot optimization history
        
        Args:
            show_params: Whether to display parameter information
            save_path: Save path, if None then do not save
        """
        if self.results is None:
            raise ValueError("No calibration results loaded")
        
        history = self.results['optimization_history']
        
        # 从结果中检查真实值
        E_true = None
        nu_true = None
        if 'true_params' in self.results:
            E_true = self.results['true_params'].get('E')
            nu_true = self.results['true_params'].get('nu')
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bayesian Optimization Calibration Analysis', fontsize=18, fontweight='bold')
        
        # Subplot 1: Optimization Progress
        iterations = range(len(history))
        scores = [h['score'] for h in history]
        best_scores = [min(scores[:i+1]) for i in range(len(scores))]
        
        ax1.plot(iterations, scores, 'o-', color=self.color_palette['primary'], 
                markersize=4, linewidth=1.5, alpha=0.7, label='Current Value')
        ax1.plot(iterations, best_scores, 's-', color=self.color_palette['success'], 
                markersize=4, linewidth=2, alpha=0.9, label='Best Value')
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Objective Function Value', fontsize=12)
        ax1.set_title('Optimization Progress', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Add optimization information
        improvement = ((scores[0] - best_scores[-1]) / scores[0] * 100) if scores[0] > 0 else 0
        info_text = f"Total Evaluations: {len(history)}\nImprovement: {improvement:.1f}%\nFinal Error: {best_scores[-1]:.6f}"
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Subplot2: Parameter Space Exploration
        E_values = [h['params'][0] for h in history]
        nu_values = [h['params'][1] for h in history]
        
        scatter = ax2.scatter(E_values, nu_values, c=scores, cmap='viridis', 
                           s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Mark best point
        best_idx = np.argmin(scores)
        ax2.scatter(E_values[best_idx], nu_values[best_idx], c='red', s=300, 
                   marker='*', edgecolors='black', linewidth=2, label='Best Parameters')
        
        # 如果存在真实值，添加到参数空间绘图中
        if E_true is not None and nu_true is not None:
            ax2.scatter(E_true, nu_true, c='blue', s=200, marker='o', 
                       edgecolors='black', linewidth=1.5, 
                       label=f'True Parameters (E={E_true:.4f}, ν={nu_true:.4f})')
        
        ax2.set_xlabel('Youngs Modulus E', fontsize=12)
        ax2.set_ylabel('Poissons Ratio ν', fontsize=12)
        ax2.set_title('Parameter Space Exploration', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Objective Function Value', fontsize=10)
        
        # Subplot3: Parameter distribution histogram
        ax3.hist(E_values, bins=20, alpha=0.7, color=self.color_palette['primary'], 
                label='E Distribution', edgecolor='black')
        
        # 如果存在真实E值，在直方图中添加
        if E_true is not None:
            ax3.axvline(x=E_true, color='blue', linestyle='--', linewidth=2, 
                       label=f'True E={E_true:.4f}')
        
        ax3.set_xlabel('Youngs Modulus E', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('E Parameter Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Subplot4: Error Distribution
        errors = [h['score'] for h in history]
        ax4.hist(errors, bins=25, alpha=0.7, color=self.color_palette['warning'], 
                edgecolor='black')
        ax4.set_xlabel('Error Value', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Optimization history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_parameter_convergence(self, save_path: Optional[str] = None):
        """
        Plot Parameter Convergence Process
        
        Args:
            save_path: Save path
        """
        if self.results is None:
            raise ValueError("No calibration results loaded")
        
        history = self.results['optimization_history']
        
        # 从结果中检查真实值
        E_true = None
        nu_true = None
        if 'true_params' in self.results:
            E_true = self.results['true_params'].get('E')
            nu_true = self.results['true_params'].get('nu')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Parameter Convergence Analysis', fontsize=16, fontweight='bold')
        
        iterations = range(len(history))
        E_values = [h['params'][0] for h in history]
        nu_values = [h['params'][1] for h in history]
        
        # E Parameter convergence
        ax1.plot(iterations, E_values, 'o-', color=self.color_palette['primary'], 
                markersize=4, linewidth=1.5, alpha=0.7)
        ax1.axhline(y=self.results['best_params']['E'], color='red', linestyle='--', 
                   linewidth=2, label=f'Optimal value: {self.results["best_params"]["E"]:.4f}')
        
        # 如果存在真实E值，在E参数收敛图中添加
        if E_true is not None:
            ax1.axhline(y=E_true, color='blue', linestyle='-.', 
                       linewidth=2, label=f'True value: {E_true:.4f}')
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Youngs Modulus E', fontsize=12)
        ax1.set_title('E Parameter Convergence', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # nu Parameter convergence
        ax2.plot(iterations, nu_values, 'o-', color=self.color_palette['secondary'], 
                markersize=4, linewidth=1.5, alpha=0.7)
        ax2.axhline(y=self.results['best_params']['nu'], color='red', linestyle='--', 
                   linewidth=2, label=f'Optimal value: {self.results["best_params"]["nu"]:.4f}')
        
        # 如果存在真实nu值，在nu参数收敛图中添加
        if nu_true is not None:
            ax2.axhline(y=nu_true, color='blue', linestyle='-.', 
                       linewidth=2, label=f'True value: {nu_true:.4f}')
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Poissons Ratio ν', fontsize=12)
        ax2.set_title('ν Parameter Convergence', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Parameter convergence plot saved to: {save_path}")
        
        plt.show()
    
    def plot_optimization_surface(self, save_path: Optional[str] = None):
        """
        Plot optimization surface (If data is sufficient)
        
        Args:
            save_path: Save path
        """
        if self.results is None:
            raise ValueError("No calibration results loaded")
        
        history = self.results['optimization_history']
        
        if len(history) < 10:
            print("⚠️ Insufficient data points, cannot plot optimization surface")
            return
        
        # 从结果中检查真实值
        E_true = None
        nu_true = None
        if 'true_params' in self.results:
            E_true = self.results['true_params'].get('E')
            nu_true = self.results['true_params'].get('nu')
        
        # Extract parameters and scores
        E_values = [h['params'][0] for h in history]
        nu_values = [h['params'][1] for h in history]
        scores = [h['score'] for h in history]
        
        # Create grid
        E_min, E_max = min(E_values), max(E_values)
        nu_min, nu_max = min(nu_values), max(nu_values)
        
        E_range = np.linspace(E_min, E_max, 30)
        nu_range = np.linspace(nu_min, nu_max, 30)
        E_grid, nu_grid = np.meshgrid(E_range, nu_range)
        
        # Simple interpolation（More complex methods can be used in practical applications）
        from scipy.interpolate import griddata
        
        points = np.column_stack([E_values, nu_values])
        score_grid = griddata(points, scores, (E_grid, nu_grid), method='linear')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Optimization Surface Analysis', fontsize=16, fontweight='bold')
        
        # Left plot：Contour plot
        contour = ax1.contour(E_grid, nu_grid, score_grid, levels=20, alpha=0.6)
        ax1.clabel(contour, inline=True, fontsize=8)
        
        # Plot data points
        scatter = ax1.scatter(E_values, nu_values, c=scores, cmap='viridis', 
                           s=80, edgecolors='black', linewidth=1, alpha=0.8)
        
        # Mark best point
        best_idx = np.argmin(scores)
        ax1.scatter(E_values[best_idx], nu_values[best_idx], c='red', s=300, 
                   marker='*', edgecolors='black', linewidth=2, label='Optimal Solution')
        
        # 如果存在真实值，在等高线图中添加
        if E_true is not None and nu_true is not None:
            ax1.scatter(E_true, nu_true, c='blue', s=200, marker='o', 
                       edgecolors='black', linewidth=1.5, 
                       label=f'True Parameters (E={E_true:.4f}, ν={nu_true:.4f})')
        
        ax1.set_xlabel('Youngs Modulus E', fontsize=12)
        ax1.set_ylabel('Poissons Ratio ν', fontsize=12)
        ax1.set_title('Objective Function Contour', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot：3DSurface plot
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(E_grid, nu_grid, score_grid, cmap='viridis', 
                               alpha=0.8, linewidth=0, antialiased=True)
        
        ax2.scatter(E_values, nu_values, scores, c='red', s=50, alpha=1.0)
        ax2.scatter(E_values[best_idx], nu_values[best_idx], scores[best_idx], 
                   c='yellow', s=200, marker='*', edgecolors='black', linewidth=2)
        
        # 如果存在真实值，在3D曲面图中添加
        if E_true is not None and nu_true is not None:
            # 计算真实值对应的评分（通过插值）
            true_score = griddata(points, scores, np.array([[E_true, nu_true]]), method='linear')[0]
            ax2.scatter(E_true, nu_true, true_score, c='blue', s=200, marker='o', 
                       edgecolors='black', linewidth=1.5)
        
        ax2.set_xlabel('Youngs Modulus E', fontsize=12)
        ax2.set_ylabel('Poissons Ratio ν', fontsize=12)
        ax2.set_zlabel('Objective Function Value', fontsize=12)
        ax2.set_title('Objective Function 3D Surface', fontsize=14, fontweight='bold')
        
        # Add color bar
        plt.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Optimization surface plot saved to: {save_path}")
        
        plt.show()
    
    def plot_comparison_with_known(self, E_true: Optional[float] = None, nu_true: Optional[float] = None,
                                  save_path: Optional[str] = None):
        """
        Plot comparison with True Parameters comparison
        
        Args:
            E_true: True Youngs Modulus Value
            nu_true: True Poissons Ratio Value
            save_path: Save path
        """
        if self.results is None:
            raise ValueError("No calibration results loaded")
        
        if E_true is None or nu_true is None:
            print("⚠️ Not provided True Parameters, cannot plot comparison")
            return
        
        best_E = self.results['best_params']['E']
        best_nu = self.results['best_params']['nu']
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Calibration Results vs True Parameters', fontsize=16, fontweight='bold')
        
        # Subplot1: Parameter Comparison
        parameters = ['E', 'nu']
        true_values = [E_true, nu_true]
        calibrated_values = [best_E, best_nu]
        
        x = np.arange(len(parameters))
        width = 0.35
        
        ax1.bar(x - width/2, true_values, width, label='True Value', 
                color=self.color_palette['primary'], alpha=0.8)
        ax1.bar(x + width/2, calibrated_values, width, label='Calibrated Value', 
                color=self.color_palette['success'], alpha=0.8)
        
        ax1.set_xlabel('parameters', fontsize=12)
        ax1.set_ylabel('Parameter values', fontsize=12)
        ax1.set_title('Parameter Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(parameters)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add numerical labels
        for i, (true_val, calib_val) in enumerate(zip(true_values, calibrated_values)):
            ax1.text(i - width/2, true_val + 0.001, f'{true_val:.4f}', 
                    ha='center', va='bottom', fontsize=10)
            ax1.text(i + width/2, calib_val + 0.001, f'{calib_val:.4f}', 
                    ha='center', va='bottom', fontsize=10)
        
        # Subplot2: Error analysis
        errors = [abs(best_E - E_true), abs(best_nu - nu_true)]
        error_percent = [errors[0]/E_true*100, errors[1]/nu_true*100]
        
        ax2.bar(parameters, errors, color=self.color_palette['warning'], alpha=0.8)
        ax2.set_xlabel('parameters', fontsize=12)
        ax2.set_ylabel('Absolute Error', fontsize=12)
        ax2.set_title('Parameter Error', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add error percentage labels
        for i, (error, percent) in enumerate(zip(errors, error_percent)):
            ax2.text(i, error + 0.0001, f'{error:.4f}\n({percent:.2f}%)', 
                    ha='center', va='bottom', fontsize=10)
        
        # Subplot3: Parameter Space Comparison
        ax3.scatter(E_true, nu_true, s=200, c='blue', marker='o', 
                   edgecolors='black', linewidth=1.5, label='True Parameters')
        ax3.scatter(best_E, best_nu, s=300, c='red', marker='*', 
                   edgecolors='black', linewidth=2, label='Calibrated Parameters')
        
        # Add error circles
        circle = plt.Circle((best_E, best_nu), max(errors), fill=False, 
                          color='red', linestyle='--', alpha=0.5, label='Error Range')
        ax3.add_patch(circle)
        
        ax3.set_xlabel('Youngs Modulus E', fontsize=12)
        ax3.set_ylabel('Poissons Ratio ν', fontsize=12)
        ax3.set_title('Parameter Space Comparison', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Subplot4: Detailed Statistics
        ax4.axis('off')
        stats_text = f"""
        Calibration Statistics
        ──────────────────
        True Parameters: E = {E_true:.4f}, ν = {nu_true:.4f}
        Calibrated Parameters: E = {best_E:.4f}, ν = {best_nu:.4f}
        
        E Parameter Error: {abs(best_E - E_true):.6f} ({abs(best_E - E_true)/E_true*100:.2f}%)
        ν Parameter Error: {abs(best_nu - nu_true):.6f} ({abs(best_nu - nu_true)/nu_true*100:.2f}%)
        
        Overall Error: {np.sqrt(errors[0]**2 + errors[1]**2):.6f}
        Calibration Accuracy: {(1 - np.sqrt(errors[0]**2 + errors[1]**2)/np.sqrt(E_true**2 + nu_true**2))*100:.2f}%
        
        Evaluation Count: {self.results['n_evaluations']}
        Final Error: {self.results['best_score']:.6f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Parameter ComparisonPlot saved to: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, save_path: Optional[str] = None):
        """
        Generate calibration result summary report
        
        Args:
            save_path: Save path
        """
        if self.results is None:
            raise ValueError("No calibration results loaded")
        
        # Create summary report
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.axis('off')
        
        # Calculate statistical information
        history = self.results['optimization_history']
        scores = [h['score'] for h in history]
        E_values = [h['params'][0] for h in history]
        nu_values = [h['params'][1] for h in history]
        
        improvement = ((scores[0] - scores[-1]) / scores[0] * 100) if scores[0] > 0 else 0
        
        report_text = f"""
        Bayesian Optimization Material Parameters Calibration Report
        ══════════════════════════════════════════════════════════════════════════════
        
        📊 Calibration results
        ──────────────────────────────────────────────────────────────────────────
        • Optimal Youngs Modulus E: {self.results['best_params']['E']:.6f}
        • Optimal Poissons Ratio ν: {self.results['best_params']['nu']:.6f}
        • Final Error: {self.results['best_score']:.8f}
        • Number of evaluations: {self.results['n_evaluations']}
        
        📈 Optimization process analysis
        ──────────────────────────────────────────────────────────────────────────
        • Initial error: {scores[0]:.8f}
        • Final Error: {scores[-1]:.8f}
        • Improvement: {improvement:.2f}%
        • Average improvement per iteration: {improvement/self.results['n_evaluations']:.3f}%
        
        📋 Parameter statistics
        ──────────────────────────────────────────────────────────────────────────
        • E parameters range: [{min(E_values):.6f}, {max(E_values):.6f}]
        • ν parameters range: [{min(nu_values):.6f}, {max(nu_values):.6f}]
        • E Parameter standard deviation: {np.std(E_values):.6f}
        • ν Parameter standard deviation: {np.std(nu_values):.6f}
        
        🎯 Performance evaluation
        ──────────────────────────────────────────────────────────────────────────
        • Optimization efficiency: {'High' if improvement > 50 else 'Medium' if improvement > 20 else 'Low'}
        • Parameter stability: {'High' if np.std(E_values) < 0.01 and np.std(nu_values) < 0.01 else 'Medium' if np.std(E_values) < 0.05 and np.std(nu_values) < 0.05 else 'Low'}
        • Convergence: {'Good' if scores[-1] < scores[0] * 0.1 else 'Average' if scores[-1] < scores[0] * 0.5 else 'Poor'}
        
        📝 Recommendations
        ──────────────────────────────────────────────────────────────────────────
        """
        
        # Provide recommendations based on results
        if improvement > 50:
            report_text += "• Optimization effect is significant, current parameter settings are reasonable\n"
        elif improvement > 20:
            report_text += "• Optimization effect is good, consider increasing iterations for further optimization\n"
        else:
            report_text += "• Optimization effect is limited, suggest adjusting parameter range or optimization strategy\n"
        
        if self.results['best_score'] < 1e-6:
            report_text += "• Error is extremely small, calibration accuracy is very high\n"
        elif self.results['best_score'] < 1e-4:
            report_text += "• Error is small, calibration accuracy is good\n"
        else:
            report_text += "• Error is large, suggest checking data quality or model settings\n"
        
        report_text += f"""
        • Calibration time: {self.results['timestamp']}
        • Data format: JSON
        """
        
        ax.text(0.05, 0.95, report_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Summary report saved to: {save_path}")
        
        plt.show()
    
    def plot_all_visualizations(self, E_true: Optional[float] = None, nu_true: Optional[float] = None,
                              save_dir: Optional[str] = None):
        """
        Plot all visualization charts
        
        Args:
            E_true: True Youngs Modulus value
            nu_true: True Poissons Ratio value
            save_dir: Save directory
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        # 从结果中检查真实值（如果未提供）
        if (E_true is None or nu_true is None) and 'true_params' in self.results:
            E_true = self.results['true_params'].get('E', E_true)
            nu_true = self.results['true_params'].get('nu', nu_true)
            
        print("🎯 Starting to generate all visualization charts...")
        if E_true is not None and nu_true is not None:
            print(f"   Using true parameters: E={E_true:.4f}, ν={nu_true:.4f}")
        
        # 1. Optimization history
        save_path = save_dir / "optimization_history.png" if save_dir else None
        self.plot_optimization_history(save_path=save_path)
        
        # 2. Parameter convergence
        save_path = save_dir / "parameter_convergence.png" if save_dir else None
        self.plot_parameter_convergence(save_path=save_path)
        
        # 3. Optimization surface
        save_path = save_dir / "optimization_surface.png" if save_dir else None
        self.plot_optimization_surface(save_path=save_path)
        
        # 4. Parameter Comparison (if true values are available)
        if E_true is not None and nu_true is not None:
            save_path = save_dir / "parameter_comparison.png" if save_dir else None
            self.plot_comparison_with_known(E_true, nu_true, save_path=save_path)
        
        # 5. Summary report
        save_path = save_dir / "summary_report.png" if save_dir else None
        self.generate_summary_report(save_path=save_path)
        
        print("🎉 All visualization charts generated successfully!")


def quick_visualize(results_file: str, E_true: Optional[float] = None, nu_true: Optional[float] = None):
    """
    Quick visualization function
    
    Args:
        results_file: Results file path
        E_true: True Youngs Modulus value
        nu_true: True Poissons Ratio value
    """
    visualizer = CalibrationVisualizer()
    visualizer.load_from_file(results_file)
    
    # 如果未提供真实值但结果文件中有，则使用结果文件中的真实值
    if (E_true is None or nu_true is None) and 'true_params' in visualizer.results:
        E_true = visualizer.results['true_params'].get('E', E_true)
        nu_true = visualizer.results['true_params'].get('nu', nu_true)
        print(f"✓ Using true parameters from results file: E={E_true:.4f}, ν={nu_true:.4f}")
        
    visualizer.plot_all_visualizations(E_true, nu_true)


if __name__ == "__main__":
    # Example usage
    print("🎯 Calibration Visualization Tool")
    print("=" * 50)
    
    # 处理命令行参数
    import sys
    if len(sys.argv) > 1:
        # 如果提供了命令行参数，直接使用提供的文件路径
        result_file = sys.argv[1]
        if Path(result_file).exists():
            print(f"✓ 使用指定的结果文件: {result_file}")
            visualizer = CalibrationVisualizer()
            visualizer.load_from_file(result_file)
            visualizer.plot_all_visualizations(save_dir="visualization_results")
            sys.exit(0)
        else:
            print(f"❌ 指定的文件不存在: {result_file}")
    
    # 如果没有命令行参数或提供的文件不存在，尝试自动查找
    # Check if result files exist
    import os
    print(f"当前工作目录: {os.getcwd()}")
    
    result_files = [
        "calibration_results_*.json",
        "../calibration_results_*.json",
        "./calibration_results_*.json",
        "calibration/calibration_results_*.json",
        "./calibration/calibration_results_*.json",
        str(Path(__file__).parent / "calibration_results_*.json")
    ]
    
    found_file = None
    for pattern in result_files:
        import glob
        print(f"搜索模式: {pattern}")
        files = glob.glob(pattern)
        if files:
            found_file = files[0]
            print(f"找到文件: {found_file}")
            break
    
    if found_file:
        print(f"✓ 找到结果文件: {found_file}")
        
        # 直接查找当前目录中特定的文件
        specific_file = "calibration_results_20250916_223827.json"
        if Path(specific_file).exists():
            found_file = specific_file
            print(f"✓ 使用特定结果文件: {specific_file}")
        elif Path("calibration/" + specific_file).exists():
            found_file = "calibration/" + specific_file
            print(f"✓ 使用特定结果文件: {found_file}")
        
        # Create visualization tool
        visualizer = CalibrationVisualizer()
        visualizer.load_from_file(found_file)
        
        # Generate all visualizations
        visualizer.plot_all_visualizations(save_dir="visualization_results")
    else:
        print("❌ 没有找到结果文件")
        print("请确保校准脚本已运行并生成了结果文件")
        print("使用方法:")
        print("  python visualize.py <结果文件路径>")
        print("  或在代码中:")
        print("  from visualize import CalibrationVisualizer")
        print("  visualizer = CalibrationVisualizer()")
        print("  visualizer.load_from_file('results.json')")
        print("  visualizer.plot_all_visualizations()")