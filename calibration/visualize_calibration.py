#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
标定结果可视化工具
用于读取calibration/results目录下的标定结果并生成可视化图表
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Union
import argparse

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("❌ Matplotlib 不可用，无法生成可视化")
    exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_calibration_results(file_path: Union[str, Path]) -> Dict:
    """加载标定结果JSON文件"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_optimization_summary(results: Dict, save_path: Optional[str] = None):
    """创建优化过程总结图表（6个子图）"""
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ Matplotlib 不可用")
        return
    
    history = results['optimization_history']
    scores = [h['score'] for h in history]
    E_values = [h['params'][0] for h in history]
    nu_values = [h['params'][1] for h in history]
    
    E_true = results.get('true_params', {}).get('E')
    nu_true = results.get('true_params', {}).get('nu')
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Bayesian Optimization Summary', fontsize=16, fontweight='bold')
    
    # 1. 优化进度
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(range(len(scores)), scores, 'o-', linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optimization Progress')
    ax1.grid(True, alpha=0.3)
    
    # 2. 参数空间
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(E_values, nu_values, c=scores, cmap='viridis', s=40, alpha=0.7)
    ax2.scatter(results['best_params']['E'], results['best_params']['nu'],
               c='red', s=200, marker='*', label='Best', edgecolors='black', linewidths=1.5)
    
    if E_true is not None and nu_true is not None:
        ax2.scatter(E_true, nu_true, c='blue', s=200, marker='o',
                   label='True', edgecolors='black', linewidths=1.5)
    
    ax2.set_xlabel('Young\'s Modulus E')
    ax2.set_ylabel('Poisson\'s Ratio ν')
    ax2.set_title('Parameter Space')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2)
    
    # 3. 统计信息
    ax3 = plt.subplot(2, 3, 3)
    ax3.text(0.1, 0.9, 'Statistics:', fontsize=14, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.1, 0.8, f'Evaluations: {len(scores)}', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.7, f'Best Score: {results["best_score"]:.6f}', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.6, f'Best E: {results["best_params"]["E"]:.4f}', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.5, f'Best ν: {results["best_params"]["nu"]:.4f}', fontsize=12, transform=ax3.transAxes)
    
    if E_true is not None and nu_true is not None:
        E_error = abs(results["best_params"]["E"] - E_true)
        nu_error = abs(results["best_params"]["nu"] - nu_true)
        ax3.text(0.1, 0.3, f'E Error: {E_error:.4f}', fontsize=12, color='darkgreen', transform=ax3.transAxes)
        ax3.text(0.1, 0.2, f'ν Error: {nu_error:.4f}', fontsize=12, color='darkgreen', transform=ax3.transAxes)
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # 4. 收敛分析
    ax4 = plt.subplot(2, 3, 4)
    convergence = [min(scores[:i+1]) for i in range(len(scores))]
    ax4.plot(range(len(convergence)), convergence, 'o-', color='green', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Best Score')
    ax4.set_title('Convergence')
    ax4.grid(True, alpha=0.3)
    
    # 5. 参数演化
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(range(len(E_values)), E_values, 'o-', label='E', linewidth=2, markersize=3)
    ax5.plot(range(len(nu_values)), nu_values, 's-', label='ν', linewidth=2, markersize=3)
    
    if E_true is not None:
        ax5.axhline(y=E_true, color='red', linestyle='--', alpha=0.7, label='True E')
    if nu_true is not None:
        ax5.axhline(y=nu_true, color='blue', linestyle='--', alpha=0.7, label='True ν')
    
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Parameter Value')
    ax5.set_title('Parameter Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 相关性矩阵
    ax6 = plt.subplot(2, 3, 6)
    if PANDAS_AVAILABLE:
        param_df = pd.DataFrame({'E': E_values, 'ν': nu_values, 'score': scores})
        corr = param_df.corr()
    else:
        corr = np.array([
            [1.0, np.corrcoef(E_values, nu_values)[0, 1], np.corrcoef(E_values, scores)[0, 1]],
            [np.corrcoef(nu_values, E_values)[0, 1], 1.0, np.corrcoef(nu_values, scores)[0, 1]],
            [np.corrcoef(scores, E_values)[0, 1], np.corrcoef(scores, nu_values)[0, 1], 1.0]
        ])
    
    im = ax6.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    param_names = ['E', 'ν', 'score']
    
    if PANDAS_AVAILABLE and hasattr(corr, 'columns'):
        for i in range(3):
            for j in range(3):
                ax6.text(j, i, f'{corr.iloc[i, j]:.2f}', ha="center", va="center",
                        color="black", fontweight='bold')
    else:
        for i in range(3):
            for j in range(3):
                ax6.text(j, i, f'{corr[i, j]:.2f}', ha="center", va="center",
                        color="black", fontweight='bold')
    
    ax6.set_xticks(range(3))
    ax6.set_yticks(range(3))
    ax6.set_xticklabels(param_names)
    ax6.set_yticklabels(param_names)
    ax6.set_title('Correlation Matrix')
    plt.colorbar(im, ax=ax6, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表保存至: {save_path}")
    
    plt.show()


def create_gp_surface_plots(results: Dict, E_bounds=None, nu_bounds=None, save_path: Optional[str] = None):
    """创建GP曲面分析图（4个子图），与calibration_Ev.py中的plot_gp_surface完全相同"""
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ Matplotlib 不可用")
        return
    
    history = results['optimization_history']
    E_values = np.array([h['params'][0] for h in history])
    nu_values = np.array([h['params'][1] for h in history])
    scores = np.array([h['score'] for h in history])
    
    # 自动确定边界
    if E_bounds is None:
        E_bounds = (E_values.min() * 0.9, E_values.max() * 1.1)
    if nu_bounds is None:
        nu_bounds = (nu_values.min() * 0.98, nu_values.max() * 1.02)
    
    # 创建网格
    n_grid = 50
    E_range = np.linspace(E_bounds[0], E_bounds[1], n_grid)
    nu_range = np.linspace(nu_bounds[0], nu_bounds[1], n_grid)
    E_grid, nu_grid = np.meshgrid(E_range, nu_range)
    
    best_idx = np.argmin(scores)
    best_E, best_nu, best_score = E_values[best_idx], nu_values[best_idx], scores[best_idx]
    
    # 使用RBF来近似GP预测
    try:
        from scipy.interpolate import Rbf
        rbf_predictor = Rbf(E_values, nu_values, scores, function='multiquadric', smooth=0.1)
        mean_pred = rbf_predictor(E_grid, nu_grid)
        # 简单的不确定性估计
        std_pred = np.zeros_like(mean_pred)
        for i in range(E_grid.shape[0]):
            for j in range(E_grid.shape[1]):
                min_dist = np.min(np.sqrt((E_values - E_grid[i, j])**2 + (nu_values - nu_grid[i, j])**2))
                std_pred[i, j] = min_dist * 0.5
    except ImportError:
        mean_pred = np.ones_like(E_grid) * np.mean(scores)
        std_pred = np.ones_like(E_grid) * np.std(scores)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gaussian Process Analysis', fontsize=16, fontweight='bold')
    
    # 左上：均值预测
    contour1 = ax1.contourf(E_grid, nu_grid, mean_pred, levels=20, cmap='viridis', alpha=0.8)
    ax1.contour(E_grid, nu_grid, mean_pred, levels=20, colors='white', alpha=0.4, linewidths=0.5)
    ax1.scatter(E_values, nu_values, c=scores, cmap='viridis', s=80, edgecolors='black', linewidth=1)
    ax1.scatter(best_E, best_nu, c='red', s=200, marker='*',
               edgecolors='white', linewidth=2, label='Best')
    ax1.set_xlabel('E')
    ax1.set_ylabel('ν')
    ax1.set_title('GP Mean Prediction')
    ax1.legend()
    plt.colorbar(contour1, ax=ax1)
    
    # 右上：不确定性
    contour2 = ax2.contourf(E_grid, nu_grid, std_pred, levels=20, cmap='plasma', alpha=0.8)
    ax2.scatter(E_values, nu_values, c='black', s=50, alpha=0.7)
    ax2.set_xlabel('E')
    ax2.set_ylabel('ν')
    ax2.set_title('GP Uncertainty')
    plt.colorbar(contour2, ax=ax2)
    
    # 左下：3D GP表面
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax3.plot_surface(E_grid, nu_grid, mean_pred, cmap='viridis', alpha=0.7)
    ax3.plot(E_values, nu_values, scores, 'r-', linewidth=2, marker='o', markersize=4)
    ax3.set_xlabel('E')
    ax3.set_ylabel('ν')
    ax3.set_zlabel('Objective')
    ax3.set_title('3D GP Surface')
    
    # 右下：点数据插值曲面（多级回退策略）
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    interpolation_success = False
    method_used = "Unknown"
    
    # 方法1: RBF插值（首选）
    try:
        from scipy.interpolate import Rbf
        
        print("🔄 尝试RBF插值...")
        rbf = Rbf(E_values, nu_values, scores, function='thin_plate', smooth=0)
        Z_interp = rbf(E_grid, nu_grid)
        
        if np.all(np.isfinite(Z_interp)):
            surf_interp = ax4.plot_surface(E_grid, nu_grid, Z_interp, cmap='viridis',
                                           alpha=0.7, edgecolor='none')
            method_used = "RBF (thin_plate)"
            interpolation_success = True
            print("✓ RBF插值成功")
        else:
            print("⚠️ RBF结果包含无效值，尝试下一个方法")
            
    except Exception as e:
        print(f"⚠️ RBF插值失败: {e}")
    
    # 方法2: griddata linear插值
    if not interpolation_success:
        try:
            from scipy.interpolate import griddata
            
            print("🔄 尝试griddata linear插值...")
            points = np.column_stack([E_values, nu_values])
            Z_interp = griddata(points, scores, (E_grid, nu_grid),
                               method='linear', fill_value=np.nan)
            
            # 填充NaN值
            if np.any(np.isnan(Z_interp)):
                Z_interp_nearest = griddata(points, scores, (E_grid, nu_grid),
                                            method='nearest')
                Z_interp = np.where(np.isnan(Z_interp), Z_interp_nearest, Z_interp)
            
            if np.all(np.isfinite(Z_interp)):
                surf_interp = ax4.plot_surface(E_grid, nu_grid, Z_interp, cmap='viridis',
                                               alpha=0.7, edgecolor='none')
                method_used = "griddata (linear)"
                interpolation_success = True
                print("✓ griddata linear插值成功")
            else:
                print("⚠️ griddata结果包含无效值，尝试下一个方法")
                
        except Exception as e:
            print(f"⚠️ griddata插值失败: {e}")
    
    # 方法3: griddata cubic插值
    if not interpolation_success:
        try:
            from scipy.interpolate import griddata
            
            print("🔄 尝试griddata cubic插值...")
            points = np.column_stack([E_values, nu_values])
            Z_interp = griddata(points, scores, (E_grid, nu_grid),
                               method='cubic', fill_value=np.nan)
            
            if np.any(np.isnan(Z_interp)):
                Z_interp_nearest = griddata(points, scores, (E_grid, nu_grid),
                                            method='nearest')
                Z_interp = np.where(np.isnan(Z_interp), Z_interp_nearest, Z_interp)
            
            if np.all(np.isfinite(Z_interp)):
                surf_interp = ax4.plot_surface(E_grid, nu_grid, Z_interp, cmap='viridis',
                                               alpha=0.7, edgecolor='none')
                method_used = "griddata (cubic)"
                interpolation_success = True
                print("✓ griddata cubic插值成功")
            else:
                print("⚠️ griddata cubic结果包含无效值，尝试下一个方法")
                
        except Exception as e:
            print(f"⚠️ griddata cubic插值失败: {e}")
    
    # 方法4: CloughTocher2D插值
    if not interpolation_success:
        try:
            from scipy.interpolate import CloughTocher2DInterpolator
            
            print("🔄 尝试CloughTocher2D插值...")
            points = np.column_stack([E_values, nu_values])
            interp = CloughTocher2DInterpolator(points, scores)
            Z_interp = interp(E_grid, nu_grid)
            
            if np.any(np.isnan(Z_interp)):
                from scipy.interpolate import griddata
                Z_interp_nearest = griddata(points, scores, (E_grid, nu_grid),
                                            method='nearest')
                Z_interp = np.where(np.isnan(Z_interp), Z_interp_nearest, Z_interp)
            
            if np.all(np.isfinite(Z_interp)):
                surf_interp = ax4.plot_surface(E_grid, nu_grid, Z_interp, cmap='viridis',
                                               alpha=0.7, edgecolor='none')
                method_used = "CloughTocher2D"
                interpolation_success = True
                print("✓ CloughTocher2D插值成功")
            else:
                print("⚠️ CloughTocher2D结果包含无效值")
                
        except Exception as e:
            print(f"⚠️ CloughTocher2D插值失败: {e}")
    
    # 最终回退：散点图
    if not interpolation_success:
        print("⚠️ 所有插值方法失败，使用散点图")
        ax4.scatter(E_values, nu_values, scores, c=scores, cmap='viridis',
                   s=100, edgecolors='black', linewidth=1.5, depthshade=True)
        method_used = "Scatter (fallback)"
    
    # 绘制观测点（所有成功情况）
    if interpolation_success:
        ax4.scatter(E_values, nu_values, scores, c='red', s=100,
                   edgecolors='black', linewidth=1.5, depthshade=True,
                   label='Observations', alpha=0.8)
        fig.colorbar(surf_interp, ax=ax4, shrink=0.5, aspect=5)
    
    # 标记最优点
    ax4.scatter([best_E], [best_nu], [best_score], c='yellow', s=300, marker='*',
               edgecolors='black', linewidth=2, label='Best', depthshade=True, zorder=10)
    
    ax4.set_title(f'Data-Driven Surface ({method_used})')
    ax4.legend(loc='upper right', fontsize=8)
    
    ax4.set_xlabel('E')
    ax4.set_ylabel('ν')
    ax4.set_zlabel('Objective')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ GP 图表保存至: {save_path}")
    
    plt.show()


def main():
    """主函数：解析命令行参数并生成可视化"""
    parser = argparse.ArgumentParser(
        description='标定结果可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 可视化最新的结果文件
  python visualize_calibration.py
  
  # 可视化指定的结果文件
  python visualize_calibration.py --file calibration/results/calibration_results_20231014_120000.json
  
  # 生成并保存所有图表
  python visualize_calibration.py --save
  
  # 只生成优化总结图
  python visualize_calibration.py --summary-only
        '''
    )
    
    parser.add_argument('--file', '-f', type=str, default=None,
                       help='标定结果JSON文件路径（默认：calibration/results/optimization_results.json）')
    parser.add_argument('--save', '-s', action='store_true',
                       help='保存图表到文件')
    parser.add_argument('--output-dir', '-o', type=str, default='calibration/results/plots',
                       help='图表输出目录（默认：calibration/results/plots）')
    parser.add_argument('--summary-only', action='store_true',
                       help='只生成优化总结图（6子图）')
    parser.add_argument('--gp-only', action='store_true',
                       help='只生成GP曲面图（4子图）')
    
    args = parser.parse_args()
    
    # 确定结果文件路径
    if args.file:
        results_file = Path(args.file)
    else:
        results_file = Path('calibration/results/optimization_results.json')
    
    # 加载结果
    try:
        print(f"📂 加载标定结果: {results_file}")
        results = load_calibration_results(results_file)
        print(f"✓ 成功加载 {len(results['optimization_history'])} 次迭代的数据")
        print(f"  最优参数: E={results['best_params']['E']:.4f}, ν={results['best_params']['nu']:.4f}")
        print(f"  最优目标值: {results['best_score']:.6f}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"提示：请确保已运行calibration_Ev.py生成结果文件")
        return
    except Exception as e:
        print(f"❌ 加载结果失败: {e}")
        return
    
    # 准备输出路径
    output_dir = Path(args.output_dir)
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = Path(results_file).stem.replace('calibration_results_', '').replace('optimization_results', 'current')
        summary_path = output_dir / f"summary_{timestamp}.png"
        gp_path = output_dir / f"gp_surface_{timestamp}.png"
    else:
        summary_path = None
        gp_path = None
    
    # 生成可视化
    print("\n📊 生成可视化图表...")
    
    if not args.gp_only:
        print("  → 优化总结图（6子图）")
        create_optimization_summary(results, save_path=summary_path)
    
    if not args.summary_only:
        print("  → GP曲面分析图（4子图）")
        create_gp_surface_plots(results, save_path=gp_path)
    
    if args.save:
        print(f"\n✓ 图表已保存至: {output_dir}")
    
    print("\n✅ 可视化完成！")


if __name__ == "__main__":
    main()