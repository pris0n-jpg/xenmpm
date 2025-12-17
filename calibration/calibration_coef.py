#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime
import sys
import logging
import time
import traceback
import argparse

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def add_calibration_path():
    calibration_dir = Path(__file__).parent
    if str(calibration_dir) not in sys.path:
        sys.path.insert(0, str(calibration_dir))


def add_xengym_path():
    xengym_dir = Path(__file__).parent.parent / "xengym"
    if str(xengym_dir) not in sys.path:
        sys.path.insert(0, str(xengym_dir))


add_calibration_path()
add_xengym_path()

try:
    from bayesian_demo import BayesianOptimizer
    from xengym.render.calibScene import create_calibration_scene
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)


class OptimizationLogger:
    """简化的日志记录器，只输出到控制台"""
    def __init__(self):
        self.logger = logging.getLogger("optimization")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_iteration(self, iteration: int, params: np.ndarray, score: float, is_best: bool = False):
        if is_best:
            self.logger.info(f"迭代 {iteration}: coef={params[0]:.4f}, 目标值={score:.6f} ★")
    
    def log_end(self, best_params: np.ndarray, best_score: float):
        self.logger.info(f"最优参数: coef={best_params[0]:.4f}, 最优目标值: {best_score:.6f}")


class EnhancedBayesianOptimizer(BayesianOptimizer):
    def __init__(self, bounds: List[Tuple[float, float]], n_initial: int = 5,
                 acquisition: str = 'ei', xi: float = 0.03,
                 logger: Optional[OptimizationLogger] = None,
                 convergence_tolerance: float = 1e-5, patience: int = 30,
                 param_transform_fn: Optional[Callable] = None):
        super().__init__(bounds, n_initial, acquisition, xi)
        self.logger = logger
        self.convergence_tolerance = convergence_tolerance
        self.patience = patience
        self.param_transform_fn = param_transform_fn
    
    def optimize(self, objective_function: Callable, max_evaluations: int = 50,
                 verbose: bool = True, return_history: bool = True):
        if verbose:
            print(f"🎯 开始优化 (初始样本: {self.n_initial}, 最大评估: {max_evaluations})")
        
        X_init = self._sample_initial_points()
        
        for i, x in enumerate(X_init):
            y = objective_function(x)
            self.gp.add_observation(x, y)
            self.X_history.append(x.copy())
            self.y_history.append(y)
            
            x_physical = self.param_transform_fn(x) if self.param_transform_fn else x
            if self.logger:
                self.logger.log_iteration(i+1, x_physical, y, y == min(self.y_history))
        
        best_idx = np.argmin(self.y_history)
        self.best_y_history.append(self.y_history[best_idx])
        self.best_x_history.append(self.X_history[best_idx].copy())
        
        total_iterations = max_evaluations - self.n_initial
        best_score = float('inf')
        no_improvement = 0
        
        for iteration in range(total_iterations):
            print("="*30, f"迭代 {iteration+1}/{total_iterations}", "="*30)
            y_best = min(self.y_history)
            x_next = self._optimize_acquisition(y_best, iteration, total_iterations)
            y_next = objective_function(x_next)
            
            self.gp.add_observation(x_next, y_next)
            self.X_history.append(x_next.copy())
            self.y_history.append(y_next)
            
            is_best = y_next < y_best
            if is_best:
                self.best_y_history.append(y_next)
                self.best_x_history.append(x_next.copy())
            else:
                self.best_y_history.append(y_best)
                self.best_x_history.append(self.best_x_history[-1].copy())
            
            current_best = min(self.y_history)
            if current_best < best_score - self.convergence_tolerance:
                best_score = current_best
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= self.patience:
                if verbose:
                    print(f"🏁 优化收敛 (迭代 {iteration+1})")
                break
            
            x_next_physical = self.param_transform_fn(x_next) if self.param_transform_fn else x_next
            if self.logger:
                self.logger.log_iteration(self.n_initial + iteration + 1, x_next_physical, y_next, is_best)
            
            if verbose and iteration % 10 == 0:
                print(f"评估 {len(self.y_history)}: 最佳={current_best:.6f}")
        
        best_idx = np.argmin(self.y_history)
        best_params = self.X_history[best_idx]
        best_score = self.y_history[best_idx]
        
        if self.logger:
            self.logger.log_end(best_params, best_score)
        
        optimization_history = []
        for i, (x, y) in enumerate(zip(self.X_history, self.y_history)):
            x_physical = self.param_transform_fn(x) if self.param_transform_fn else x
            optimization_history.append({'iteration': i, 'params': x_physical.tolist(), 'score': float(y)})
        
        return (best_params, best_score, optimization_history) if return_history else (best_params, best_score)


class RealDataInterface:
    def __init__(self):
        self.real_data_cache = {}
    
    def load_from_json(self, file_path: Union[str, Path]) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_from_pickle(self, file_path: Union[str, Path]) -> Dict:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def create_real_raw_data(self, E: float, nu: float, coef: float = 0.0) -> Dict:
        scene = self._create_calibration_scene()
        if scene is None:
            raise RuntimeError("无法创建标定场景")
        return scene.calibrate_with_parameters(E, nu, coef)
    
    def _create_calibration_scene(self):
        obj_path = Path("/home/czl/Downloads/workspace/xengym/calibration/obj")
        if not obj_path.exists():
            return None
        candidates = [
            "circle_r3.STL", 
            "circle_r4.STL", 
            # "circle_r5.STL",
            "r3d5.STL",
            "r4d5.STL",
            "rhombus_d6.STL",
            # "rhombus_d8.STL",
            "square_d6.STL",
            # "square_d8.STL",
            "tri_d6.STL"
            ]
        stl_files = [str(obj_path / n) for n in candidates if (obj_path / n).exists()]
        # stl_files = list(obj_path.glob("*.STL"))
        # if not stl_files:
        #     return None
        # print("Found STL files:", stl_files)
        object_files = [str(f) for f in stl_files[:]]  
        return create_calibration_scene(object_files=object_files, visible=False, sensor_visible=False)


class CoefCalibration:
    def __init__(self,
                 real_data_interface: RealDataInterface,
                 E_fixed: float,
                 nu_fixed: float,
                 coef_bounds: Tuple[float, float] = (-0.5, 0.5),
                 n_initial: int = 10,
                 n_iterations: int = 30,
                 acquisition: str = 'adaptive',
                 xi: float = 0.01,
                 enable_logging: bool = True,
                 convergence_tolerance: float = 1e-5,
                 patience: int = 20):
        
        self.real_data_interface = real_data_interface
        self.E_fixed = E_fixed
        self.nu_fixed = nu_fixed
        self.coef_bounds = coef_bounds
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.acquisition = acquisition
        self.xi = xi
        self.convergence_tolerance = convergence_tolerance
        self.patience = patience
        
        self.scene = real_data_interface._create_calibration_scene()
        if self.scene is None:
            raise RuntimeError("无法创建标定场景")
        
        self.optimization_history = []
        self.results_dir = Path("results_coef")
        self.vis_dir = Path("visualizations_coef")
        self.results_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)

        self.logger = OptimizationLogger() if enable_logging else None
        
        print(f"📋 固定参数: E={E_fixed:.4f}, nu={nu_fixed:.4f}")
        print(f"🎯 优化参数: coef ∈ [{coef_bounds[0]:.3f}, {coef_bounds[1]:.3f}]")
    
    def calculate_calibration_error(self, sim_data: Dict, real_data: Dict) -> float:
        """计算标定误差，使用与calibration_Ev copy.py相同的方法"""
        def rmse(a, b):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape:
                if a.size == b.size:
                    b = b.reshape(a.shape)
                else:
                    return np.nan
            mask = np.isfinite(a) & np.isfinite(b)
            return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if np.any(mask) else np.nan
        
        total_error, total_weight = 0.0, 0.0
        weight_marker, weight_force = 0, 1
        
        for obj_name in real_data.keys() & sim_data.keys():
            real_obj, sim_obj = real_data[obj_name], sim_data[obj_name]
            if not isinstance(real_obj, dict) or not isinstance(sim_obj, dict):
                continue
            
            for traj_name in real_obj.keys() & sim_obj.keys():
                real_traj, sim_traj = real_obj[traj_name], sim_obj[traj_name]
                if not isinstance(real_traj, dict) or not isinstance(sim_traj, dict):
                    continue
                
                for step_name in real_traj.keys() & sim_traj.keys():
                    real_step = real_traj.get(step_name, {}) or {}
                    sim_step = sim_traj.get(step_name, {}) or {}
                    
                    if 'marker_displacement' in real_step and 'marker_displacement' in sim_step:
                        err = rmse(real_step['marker_displacement'], sim_step['marker_displacement'])
                        if np.isfinite(err):
                            total_error += weight_marker * err
                            total_weight += weight_marker
                    
                    if 'force_xyz' in real_step and 'force_xyz' in sim_step:
                        err = rmse(real_step['force_xyz'][2], sim_step['force_xyz'][2])
                        if np.isfinite(err):
                            total_error += weight_force * err
                            total_weight += weight_force
        
        return total_error / total_weight if total_weight > 0 else float('inf')
    
    def objective_function(self, params: np.ndarray, real_data: Dict) -> float:
        """目标函数，仅优化coef参数，E和nu固定"""
        try:
            coef = float(params[0])
            sim_data = self.scene.calibrate_with_parameters(self.E_fixed, self.nu_fixed, coef)
            data_error = self.calculate_calibration_error(sim_data, real_data)
            return data_error
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"参数评估失败: {e}")
            return 20.0
    
    def run_calibration(self, real_data: Optional[Dict] = None) -> Dict:
        print("🎯 开始coef参数贝叶斯优化标定")
        
        if real_data is None:
            print("❌ 需要提供真实数据")
            return None
        
        def objective(params):
            return self.objective_function(params, real_data)
        
        bounds = [self.coef_bounds]
        
        optimizer = EnhancedBayesianOptimizer(
            bounds=bounds, n_initial=self.n_initial, acquisition=self.acquisition,
            xi=self.xi, logger=self.logger, convergence_tolerance=self.convergence_tolerance,
            patience=self.patience, param_transform_fn=None
        )
        
        # 确保coef=0作为初始参考点
        print("📍 首先评估 coef=0 作为基准参考...")
        coef_zero = np.array([0.0])
        score_zero = objective(coef_zero)
        optimizer.gp.add_observation(coef_zero, score_zero)
        optimizer.X_history.append(coef_zero.copy())
        optimizer.y_history.append(score_zero)
        if self.logger:
            self.logger.log_iteration(1, coef_zero, score_zero, True)
        print(f"✓ coef=0 基准得分: {score_zero:.6f}")
        
        # 调整初始采样数量（已经有一个coef=0了）
        actual_n_initial = max(1, self.n_initial - 1)
        
        best_params, best_score, optimization_history = optimizer.optimize(
            objective_function=objective,
            max_evaluations=actual_n_initial + self.n_iterations,
            verbose=True
        )
        
        # 将coef=0的评估添加到历史记录开头
        optimization_history.insert(0, {'iteration': 0, 'params': [0.0], 'score': float(score_zero)})
        
        self.optimization_history = optimization_history
        
        result = {
            'best_params': {
                'E': self.E_fixed,
                'nu': self.nu_fixed,
                'coef': round(float(best_params[0]), 4)
            },
            'best_score': float(best_score),
            'optimization_history': optimization_history,
            'timestamp': datetime.now().isoformat(),
            'n_evaluations': len(optimization_history)
        }
        
        print(f"\n🎉 优化完成! 最优参数: coef={result['best_params']['coef']:.4f}")
        print(f"   固定参数: E={self.E_fixed:.4f}, nu={self.nu_fixed:.4f}")
        return result
    
    def save_results(self, results: Dict, file_path: Union[str, Path]):
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.results_dir / file_path
        
        # 确保父目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 结果保存至: {file_path}")

    def create_optimization_summary(self, results: Dict, save_path: Optional[str] = None):
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Matplotlib 不可用")
            return
        
        history = results['optimization_history']
        scores = [h['score'] for h in history]
        coef_values = [h['params'][0] for h in history]
        
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
        ax3.text(0.1, 0.5, f'Fixed E: {self.E_fixed:.4f}', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.4, f'Fixed ν: {self.nu_fixed:.4f}', fontsize=12, transform=ax3.transAxes)
        
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


def main():
    print("🎯 coef参数贝叶斯优化标定")
    
    parser = argparse.ArgumentParser(description='Coef Parameter Calibration using Bayesian Optimization')
    parser.add_argument('--E-fixed', type=float, default=0.7966, help='固定的E值（来自先前E-nu标定）')
    parser.add_argument('--nu-fixed', type=float, default=0.3523, help='固定的nu值（来自先前E-nu标定）')
    parser.add_argument('--n-initial', type=int, default=5, help='初始采样点数')
    parser.add_argument('--n-iterations', type=int, default=50, help='优化迭代次数')
    parser.add_argument('--coef-min', type=float, default=-0.5, help='coef参数下界')
    parser.add_argument('--coef-max', type=float, default=0.5, help='coef参数上界')
    parser.add_argument('--acquisition', type=str, default='ts',
                       choices=['ei', 'ucb', 'pi', 'ts', 'adaptive'], help='采集函数类型')
    parser.add_argument('--xi', type=float, default=0.01, help='探索参数')
    parser.add_argument('--no-visualization', action='store_true', help='禁用可视化')
    parser.add_argument('--save-plots', action='store_true', help='保存图表到文件')
    
    args = parser.parse_args()
    
    real_data_interface = RealDataInterface()
    calibrator = CoefCalibration(
        real_data_interface=real_data_interface,
        E_fixed=args.E_fixed,
        nu_fixed=args.nu_fixed,
        coef_bounds=(args.coef_min, args.coef_max),
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        acquisition=args.acquisition,
        xi=args.xi,
        enable_logging=True
    )
    
    real_data = None
    calibration_dir = Path(__file__).parent
    data_sources = [
        calibration_dir / "real_data.json",
        calibration_dir / "real_data.pkl",
        calibration_dir / "data" / "real_data.json",
        calibration_dir / "data" / "real_data.pkl"
    ]
    
    for source in data_sources:
        if source.exists():
            try:
                real_data = real_data_interface.load_from_json(source) if str(source).endswith('.json') else real_data_interface.load_from_pickle(source)
                print(f"✓ 成功加载数据: {source}")
                break
            except Exception as e:
                print(f"⚠️ 加载 {source} 失败: {e}")
    
    if real_data is None:
        print("❌ 未找到真实数据文件，请提供真实标定数据")
        return
    
    print('✓ 使用真实数据进行标定')
    
    try:
        results = calibrator.run_calibration(real_data=real_data)
        
        if results is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            calibrator.save_results(results, f"coef_calibration_results_{timestamp}.json")
            calibrator.save_results(results, "coef_optimization_results.json")
            
            print(f"\n📊 标定结果:")
            print(f"   coef = {results['best_params']['coef']:.4f}")
            print(f"   E (固定) = {results['best_params']['E']:.4f}")
            print(f"   nu (固定) = {results['best_params']['nu']:.4f}")
            print(f"   误差 = {results['best_score']:.6f}")
            
            # 可视化
            if not args.no_visualization:
                print(f"\n📊 生成可视化图表...")
                
                if args.save_plots:
                    plot_dir = Path(f"coef_calibration_plots_{timestamp}")
                    plot_dir = calibrator.vis_dir / plot_dir
                    plot_dir.mkdir(exist_ok=True)
                    
                    calibrator.create_optimization_summary(
                        results, save_path=str(plot_dir / "coef_optimization_summary.png"),
                    )
                    
                    print(f"✓ 所有图表已保存至: {plot_dir}")
                else:
                    calibrator.create_optimization_summary(results)
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断优化")
    except Exception as e:
        print(f"\n❌ 标定失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()