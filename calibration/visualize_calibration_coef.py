#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coef参数标定结果可视化工具
用于读取calibration/results_coef目录下的标定结果并生成可视化图表
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
    """创建优化过程总结图表（6个子图）- 专门针对coef参数"""
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ Matplotlib 不可用")
        return
    
    history = results['optimization_history']
    scores = [h['score'] for h in history]
    coef_values = [h['params'][0] for h in history]
    
    E_fixed = results['best_params']['E']
    nu_fixed = results['best_params']['nu']
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Coef Parameter Calibration - Bayesian Optimization Summary', 
                fontsize=16, fontweight='bold')
    
    # 1. 优化进度
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(range(len(scores)), scores, 'o-', linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optimization Progress')
    ax1.grid(True, alpha=0.3)
    
    # 2. coef参数演化
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(range(len(coef_values)), coef_values, c=scores, 
                        cmap='viridis', s=60, alpha=0.7)
    best_idx = np.argmin(scores)
    ax2.scatter(best_idx, coef_values[best_idx], c='red', s=300, marker='*', 
               label='Best', edgecolors='black', linewidths=2)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('coef Value')
    ax2.set_title('Coef Parameter Evolution')
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Objective Value')
    ax2.grid(True, alpha=0.3)
    
    # 3. 统计信息
    ax3 = plt.subplot(2, 3, 3)
    ax3.text(0.1, 0.9, 'Statistics:', fontsize=14, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.1, 0.8, f'Evaluations: {len(scores)}', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.7, f'Best Score: {results["best_score"]:.6f}', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.6, f'Best coef: {results["best_params"]["coef"]:.4f}', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.5, f'Fixed E: {E_fixed:.4f}', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.4, f'Fixed ν: {nu_fixed:.4f}', fontsize=12, transform=ax3.transAxes)
    
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
    
    # 5. Coef值分布直方图
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(coef_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(results['best_params']['coef'], color='red', linestyle='--', linewidth=2, label='Best')
    if coef_true is not None:
        ax5.axvline(coef_true, color='blue', linestyle='--', linewidth=2, label='True')
    ax5.set_xlabel('coef Value')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Coef Value Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Coef vs Objective 曲线图
    ax6 = plt.subplot(2, 3, 6)
    # 按coef值排序以便绘制平滑曲线
    sorted_indices = np.argsort(coef_values)
    coef_sorted = np.array(coef_values)[sorted_indices]
    scores_sorted = np.array(scores)[sorted_indices]
    
    ax6.plot(coef_sorted, scores_sorted, '-', linewidth=2, alpha=0.6, color='gray', label='Trajectory')
    ax6.scatter(coef_values, scores, s=50, alpha=0.7, c='skyblue', edgecolors='black', linewidth=0.5, label='Evaluations')
    ax6.scatter(results['best_params']['coef'], results['best_score'],
               c='red', s=200, marker='*', label='Best', edgecolors='black', linewidths=2, zorder=5)
    ax6.set_xlabel('coef Value')
    ax6.set_ylabel('Objective Value')
    ax6.set_title('Coef vs Objective')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表保存至: {save_path}")
    
    plt.show()


def create_coef_1d_analysis(results: Dict, coef_bounds=None, save_path: Optional[str] = None):
    """创建coef参数的1D分析图（4个子图）"""
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ Matplotlib 不可用")
        return
    
    history = results['optimization_history']
    coef_values = np.array([h['params'][0] for h in history])
    scores = np.array([h['score'] for h in history])
    
    # 自动确定边界
    if coef_bounds is None:
        coef_bounds = (coef_values.min() - 0.05, coef_values.max() + 0.05)
    
    best_idx = np.argmin(scores)
    best_coef, best_score = coef_values[best_idx], scores[best_idx]
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Coef Parameter 1D Analysis', fontsize=16, fontweight='bold')
    
    # 1. 左上：所有评估点散点图
    ax1 = plt.subplot(2, 2, 1)
    scatter1 = ax1.scatter(coef_values, scores, c=range(len(scores)), 
                          cmap='viridis', s=80, alpha=0.7, edgecolors='black', linewidth=1)
    ax1.scatter(best_coef, best_score, c='red', s=300, marker='*',
               edgecolors='white', linewidth=2, label='Best', zorder=5)
    ax1.set_xlabel('coef Value')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('All Evaluations (colored by iteration)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Iteration')
    
    # 2. 右上：拟合曲线（使用样条插值或RBF）
    ax2 = plt.subplot(2, 2, 2)
    
    # 尝试使用RBF或多项式拟合
    coef_dense = np.linspace(coef_bounds[0], coef_bounds[1], 200)
    interpolation_success = False
    
    try:
        from scipy.interpolate import Rbf
        rbf = Rbf(coef_values, scores, function='multiquadric', smooth=0.1)
        scores_dense = rbf(coef_dense)
        interpolation_success = True
        method_name = "RBF"
    except:
        try:
            from scipy.interpolate import UnivariateSpline
            # 排序数据
            sort_idx = np.argsort(coef_values)
            spline = UnivariateSpline(coef_values[sort_idx], scores[sort_idx], s=0.5)
            scores_dense = spline(coef_dense)
            interpolation_success = True
            method_name = "Spline"
        except:
            # 回退到多项式拟合
            try:
                poly = np.poly1d(np.polyfit(coef_values, scores, deg=min(3, len(coef_values)-1)))
                scores_dense = poly(coef_dense)
                interpolation_success = True
                method_name = "Polynomial"
            except:
                pass
    
    if interpolation_success:
        ax2.plot(coef_dense, scores_dense, 'b-', linewidth=2, alpha=0.6, label=f'{method_name} Fit')
    
    ax2.scatter(coef_values, scores, c='gray', s=50, alpha=0.5, label='Observations')
    ax2.scatter(best_coef, best_score, c='red', s=300, marker='*',
               edgecolors='black', linewidth=2, label='Best', zorder=5)
    ax2.set_xlabel('coef Value')
    ax2.set_ylabel('Objective Value')
    ax2.set_title('Objective Function Approximation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 左下：时序演化图
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(range(len(coef_values)), coef_values, 'o-', linewidth=2, 
            markersize=6, alpha=0.7, label='coef evolution')
    ax3.axhline(y=best_coef, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='Best coef')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('coef Value')
    ax3.set_title('Coef Parameter Evolution Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 右下：残差分析（如果有拟合）
    ax4 = plt.subplot(2, 2, 4)
    
    if interpolation_success:
        try:
            # 计算残差
            if method_name == "RBF":
                fitted_scores = rbf(coef_values)
            elif method_name == "Spline":
                fitted_scores = spline(coef_values)
            else:
                fitted_scores = poly(coef_values)
            
            residuals = scores - fitted_scores
            
            ax4.scatter(coef_values, residuals, c=range(len(residuals)), 
                       cmap='viridis', s=80, alpha=0.7, edgecolors='black', linewidth=1)
            ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax4.set_xlabel('coef Value')
            ax4.set_ylabel('Residual')
            ax4.set_title('Residual Plot')
            ax4.grid(True, alpha=0.3)
        except:
            # 如果残差计算失败，显示局部不确定性
            ax4.text(0.5, 0.5, 'Residual analysis unavailable', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    else:
        # 显示采样密度
        ax4.hist(coef_values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('coef Value')
        ax4.set_ylabel('Sampling Frequency')
        ax4.set_title('Parameter Space Exploration')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 1D分析图保存至: {save_path}")
    
    plt.show()


def main():
    """主函数：解析命令行参数并生成可视化"""
    parser = argparse.ArgumentParser(
        description='Coef参数标定结果可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 可视化最新的coef标定结果文件
  python visualize_calibration_coef.py
  
  # 可视化指定的结果文件
  python visualize_calibration_coef.py --file calibration/results_coef/coef_calibration_results_20231014_120000.json
  
  # 生成并保存所有图表
  python visualize_calibration_coef.py --save
  
  # 只生成优化总结图
  python visualize_calibration_coef.py --summary-only
        '''
    )
    
    parser.add_argument('--file', '-f', type=str, default=None,
                       help='标定结果JSON文件路径（默认：calibration/results_coef/coef_optimization_results.json）')
    parser.add_argument('--save', '-s', action='store_true',
                       help='保存图表到文件')
    parser.add_argument('--output-dir', '-o', type=str, default='calibration/results_coef/plots',
                       help='图表输出目录（默认：calibration/results_coef/plots）')
    parser.add_argument('--summary-only', action='store_true',
                       help='只生成优化总结图（6子图）')
    parser.add_argument('--analysis-only', action='store_true',
                       help='只生成1D分析图（4子图）')
    
    args = parser.parse_args()
    
    # 确定结果文件路径
    if args.file:
        results_file = Path(args.file)
    else:
        results_file = Path('calibration/results_coef/coef_optimization_results.json')
    
    # 加载结果
    try:
        print(f"📂 加载coef标定结果: {results_file}")
        results = load_calibration_results(results_file)
        print(f"✓ 成功加载 {len(results['optimization_history'])} 次迭代的数据")
        print(f"  最优参数: coef={results['best_params']['coef']:.4f}")
        print(f"  固定参数: E={results['best_params']['E']:.4f}, ν={results['best_params']['nu']:.4f}")
        print(f"  最优目标值: {results['best_score']:.6f}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"提示：请确保已运行calibration_coef.py生成结果文件")
        return
    except Exception as e:
        print(f"❌ 加载结果失败: {e}")
        return
    
    # 准备输出路径
    output_dir = Path(args.output_dir)
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = Path(results_file).stem.replace('coef_calibration_results_', '').replace('coef_optimization_results', 'current')
        summary_path = output_dir / f"coef_summary_{timestamp}.png"
        analysis_path = output_dir / f"coef_1d_analysis_{timestamp}.png"
    else:
        summary_path = None
        analysis_path = None
    
    # 生成可视化
    print("\n📊 生成可视化图表...")
    
    if not args.analysis_only:
        print("  → Coef优化总结图（6子图）")
        create_optimization_summary(results, save_path=summary_path)
    
    if not args.summary_only:
        print("  → Coef 1D分析图（4子图）")
        create_coef_1d_analysis(results, save_path=analysis_path)
    
    if args.save:
        print(f"\n✓ 图表已保存至: {output_dir}")
    
    print("\n✅ 可视化完成！")


if __name__ == "__main__":
    main()