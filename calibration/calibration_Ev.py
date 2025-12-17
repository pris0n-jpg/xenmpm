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
        
        # 只保留控制台输出
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_iteration(self, iteration: int, params: np.ndarray, score: float, is_best: bool = False):
        if is_best:
            self.logger.info(f"迭代 {iteration}: 参数={params}, 目标值={score:.6f} ★")
    
    def log_end(self, best_params: np.ndarray, best_score: float):
        self.logger.info(f"最优参数: {best_params}, 最优目标值: {best_score:.6f}")


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
    
    def create_real_raw_data(self, E_true: float = 0.1983, nu_true: float = 0.4795) -> Dict:
        scene = self._create_calibration_scene()
        if scene is None:
            raise RuntimeError("无法创建标定场景")
        return scene.calibrate_with_parameters(E_true, nu_true)
    
    def _create_calibration_scene(self):
        obj_path = Path("/home/czl/Downloads/workspace/xengym/calibration/obj")
        if not obj_path.exists():
            return None
        
        stl_files = list(obj_path.glob("*.STL"))
        if not stl_files:
            return None
        # print("Found STL files:", stl_files)
        object_files = [str(f) for f in stl_files[1:]]  
        print("object_files:", object_files)
        return create_calibration_scene(object_files=object_files, visible=False, sensor_visible=False)


class BayesianCalibration:
    def __init__(self,
                 real_data_interface: RealDataInterface,
                 E_bounds: Tuple[float, float] = (0.0500, 0.5000),
                 nu_bounds: Tuple[float, float] = (0.4000, 0.5000),
                 n_initial: int = 20,
                 n_iterations: int = 40,
                 acquisition: str = 'adaptive',
                 xi: float = 0.01,
                 enable_logging: bool = True,
                 convergence_tolerance: float = 1e-5,
                 patience: int = 30,
                 use_log_scale_E: bool = False):
        
        self.real_data_interface = real_data_interface
        self.E_bounds = E_bounds
        self.nu_bounds = nu_bounds
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.acquisition = acquisition
        self.xi = xi
        self.convergence_tolerance = convergence_tolerance
        self.patience = patience
        self.use_log_scale_E = use_log_scale_E
        
        self.scene = real_data_interface._create_calibration_scene()
        if self.scene is None:
            raise RuntimeError("无法创建标定场景")
        
        self.optimization_history = []
        self.results_dir = Path("results")
        self.vis_dir = Path("visualizations")
        self.results_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)

        self.logger = OptimizationLogger() if enable_logging else None
        self._setup_parameter_scaling()
    
    def _setup_parameter_scaling(self):
        if self.use_log_scale_E:
            self.E_log_bounds = (np.log(self.E_bounds[0]), np.log(self.E_bounds[1]))
    
    def _transform_params_to_physical(self, params_scaled: np.ndarray) -> np.ndarray:
        E_scaled, nu = params_scaled[0], params_scaled[1]
        E = np.exp(E_scaled) if self.use_log_scale_E else E_scaled
        return np.array([E, nu])
    
    def calculate_calibration_error(self, sim_data: Dict, real_data: Dict) -> float:
        def rmse(a, b):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape:
                if a.size == b.size:
                    b = b.reshape(a.shape)
                else:
                    return np.nan
            mask = np.isfinite(a) & np.isfinite(b)
            return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if np.any(mask) else np.nan
        
        def mse(a, b):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape:
                if a.size == b.size:
                    b = b.reshape(a.shape)
                else:
                    return np.nan
            mask = np.isfinite(a) & np.isfinite(b)
            return float(np.mean((a[mask] - b[mask]) ** 2)) if np.any(mask) else np.nan
        
        total_error, total_weight = 0.0, 0.0
        weight_marker, weight_force = 200, 1
        
        for obj_name in real_data.keys() & sim_data.keys():
            real_obj, sim_obj = real_data[obj_name], sim_data[obj_name]
            if not isinstance(real_obj, dict) or not isinstance(sim_obj, dict):
                continue
            
            for traj_name in real_obj.keys() & sim_obj.keys():
                real_traj, sim_traj = real_obj[traj_name], sim_obj[traj_name]
                if not isinstance(real_traj, dict) or not isinstance(sim_traj, dict):
                    continue
                
                # 收集该轨迹的所有步骤并排序
                common_steps = sorted(real_traj.keys() & sim_traj.keys())
                
                # 处理 marker_displacement（逐步比较）
                for step_name in common_steps:
                    real_step = real_traj.get(step_name, {}) or {}
                    sim_step = sim_traj.get(step_name, {}) or {}
                    
                    if 'marker_displacement' in real_step and 'marker_displacement' in sim_step:
                        err = rmse(real_step['marker_displacement'], sim_step['marker_displacement'])
                        if np.isfinite(err):
                            total_error += weight_marker * err
                            total_weight += weight_marker
                
                # 处理 force_xyz[2]（计算相邻步骤的差值）
                if len(common_steps) >= 2:
                    for i in range(len(common_steps) - 1):
                        step_curr = common_steps[i]
                        step_next = common_steps[i + 1]
                        
                        real_curr = real_traj.get(step_curr, {}) or {}
                        real_next = real_traj.get(step_next, {}) or {}
                        sim_curr = sim_traj.get(step_curr, {}) or {}
                        sim_next = sim_traj.get(step_next, {}) or {}
                        
                        # 检查是否都有 force_xyz 数据
                        if ('force_xyz' in real_curr and 'force_xyz' in real_next and
                            'force_xyz' in sim_curr and 'force_xyz' in sim_next):
                            
                            # 计算真实数据的力差值
                            real_force_curr = float(real_curr['force_xyz'][2])
                            real_force_next = float(real_next['force_xyz'][2])
                            real_force_diff = real_force_next - real_force_curr
                            
                            # 计算仿真数据的力差值
                            sim_force_curr = float(sim_curr['force_xyz'][2])
                            sim_force_next = float(sim_next['force_xyz'][2])
                            sim_force_diff = sim_force_next - sim_force_curr
                            
                            # 比较差值的误差
                            force_diff_error = abs(real_force_diff - sim_force_diff)
                            
                            if np.isfinite(force_diff_error):
                                total_error += weight_force * force_diff_error
                                total_weight += weight_force
        
        return total_error / total_weight if total_weight > 0 else float('inf')
    
    
    def objective_function(self, params: np.ndarray, real_data: Dict, use_scaled_space: bool = False) -> float:
        """
        纯数据驱动的目标函数，无先验假设，适用于未知材料参数的真实标定场景。
        
        目标函数组成：
        1. data_error: 仿真数据与观测数据的拟合误差（核心项）
        2. boundary_penalty: 软边界惩罚，仅当参数越界时生效（范围无关）
        
        无任何关于"合理"参数值的假设，完全由实验数据决定最优参数。
        """
        params_physical = self._transform_params_to_physical(params) if (use_scaled_space and self.use_log_scale_E) else params
        
        try:
            E, nu = float(params_physical[0]), float(params_physical[1])
            
            # 执行仿真并计算数据拟合误差
            sim_data = self.scene.calibrate_with_parameters(E, nu)
            data_error = self.calculate_calibration_error(sim_data, real_data)
            
            # 纯数据驱动：仅数据拟合误差 + 边界惩罚，无任何先验项
            return data_error 
            
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"参数评估失败: {e}")
            # 异常处理：返回有限惩罚值，保持优化器可继续
            return 20.0
    
    def run_calibration(self, real_data: Optional[Dict] = None,
                       E_true: Optional[float] = None, nu_true: Optional[float] = None) -> Dict:
        print("🎯 开始贝叶斯优化标定")
        
        if real_data is None:
            if E_true is None or nu_true is None:
                print("❌ 需要提供真实数据或真实参数")
                return None
            real_data = self.real_data_interface.create_real_raw_data(E_true, nu_true)
        
        if self.use_log_scale_E:
            def objective(params_scaled):
                return self.objective_function(params_scaled, real_data, use_scaled_space=True)
            bounds = [self.E_log_bounds, self.nu_bounds]
            param_transform_fn = self._transform_params_to_physical
        else:
            def objective(params):
                return self.objective_function(params, real_data, use_scaled_space=False)
            bounds = [self.E_bounds, self.nu_bounds]
            param_transform_fn = None
        
        optimizer = EnhancedBayesianOptimizer(
            bounds=bounds, n_initial=self.n_initial, acquisition=self.acquisition,
            xi=self.xi, logger=self.logger, convergence_tolerance=self.convergence_tolerance,
            patience=self.patience, param_transform_fn=param_transform_fn
        )
        
        best_params, best_score, optimization_history = optimizer.optimize(
            objective_function=objective,
            max_evaluations=self.n_initial + self.n_iterations,
            verbose=True
        )
        
        self.optimization_history = optimization_history
        best_params_physical = self._transform_params_to_physical(best_params) if self.use_log_scale_E else best_params
        
        result = {
            'best_params': {'E': round(float(best_params_physical[0]), 4), 'nu': round(float(best_params_physical[1]), 4)},
            'best_score': float(best_score),
            'optimization_history': optimization_history,
            'timestamp': datetime.now().isoformat(),
            'n_evaluations': len(optimization_history)
        }
        
        print(f"\n🎉 优化完成! 最优参数: E={result['best_params']['E']:.4f}, nu={result['best_params']['nu']:.4f}")
        return result
    
    def save_results(self, results: Dict, file_path: Union[str, Path],
                    E_true: Optional[float] = None, nu_true: Optional[float] = None):
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.results_dir / file_path
        
        if E_true is not None and nu_true is not None:
            results['true_params'] = {'E': E_true, 'nu': nu_true}
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 结果保存至: {file_path}")


    def create_optimization_summary(self, results: Dict, save_path: Optional[str] = None,
                                   E_true: Optional[float] = None, nu_true: Optional[float] = None):
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Matplotlib 不可用")
            return
        
        history = results['optimization_history']
        scores = [h['score'] for h in history]
        E_values = [h['params'][0] for h in history]
        nu_values = [h['params'][1] for h in history]
        
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
    
    def plot_gp_surface(self, optimizer: EnhancedBayesianOptimizer, save_path: Optional[str] = None):
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Matplotlib 不可用")
            return
        
        n_grid = 50
        E_range = np.linspace(self.E_bounds[0], self.E_bounds[1], n_grid)
        nu_range = np.linspace(self.nu_bounds[0], self.nu_bounds[1], n_grid)
        E_grid, nu_grid = np.meshgrid(E_range, nu_range)
        
        if self.use_log_scale_E:
            E_grid_scaled = np.log(E_grid)
            X_grid_scaled = np.column_stack([E_grid_scaled.ravel(), nu_grid.ravel()])
        else:
            X_grid_scaled = np.column_stack([E_grid.ravel(), nu_grid.ravel()])
        
        mean_pred, var_pred = optimizer.gp.predict(X_grid_scaled)
        mean_pred = mean_pred.reshape(E_grid.shape)
        std_pred = np.sqrt(var_pred).reshape(E_grid.shape)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gaussian Process Analysis', fontsize=16, fontweight='bold')
        
        # 均值预测
        contour1 = ax1.contourf(E_grid, nu_grid, mean_pred, levels=20, cmap='viridis', alpha=0.8)
        ax1.contour(E_grid, nu_grid, mean_pred, levels=20, colors='white', alpha=0.4, linewidths=0.5)
        
        if len(optimizer.X_history) > 0:
            X_array = np.array(optimizer.X_history)
            X_array_physical = np.array([self._transform_params_to_physical(x) for x in X_array]) if self.use_log_scale_E else X_array
            
            ax1.scatter(X_array_physical[:, 0], X_array_physical[:, 1], c=optimizer.y_history,
                       cmap='viridis', s=80, edgecolors='black', linewidth=1)
            
            best_idx = np.argmin(optimizer.y_history)
            best_x = X_array_physical[best_idx]
            ax1.scatter(best_x[0], best_x[1], c='red', s=200, marker='*',
                       edgecolors='white', linewidth=2, label='Best')
        
        ax1.set_xlabel('E')
        ax1.set_ylabel('ν')
        ax1.set_title('GP Mean Prediction')
        ax1.legend()
        plt.colorbar(contour1, ax=ax1)
        
        # 不确定性
        contour2 = ax2.contourf(E_grid, nu_grid, std_pred, levels=20, cmap='plasma', alpha=0.8)
        if len(optimizer.X_history) > 0:
            ax2.scatter(X_array_physical[:, 0], X_array_physical[:, 1], c='black', s=50, alpha=0.7)
        ax2.set_xlabel('E')
        ax2.set_ylabel('ν')
        ax2.set_title('GP Uncertainty')
        plt.colorbar(contour2, ax=ax2)
        
        # 3D 表面
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        surf = ax3.plot_surface(E_grid, nu_grid, mean_pred, cmap='viridis', alpha=0.7)
        
        if len(optimizer.X_history) > 0:
            ax3.plot(X_array_physical[:, 0], X_array_physical[:, 1], optimizer.y_history,
                    'r-', linewidth=2, marker='o', markersize=4)
        
        ax3.set_xlabel('E')
        ax3.set_ylabel('ν')
        ax3.set_zlabel('Objective')
        ax3.set_title('3D GP Surface')
        
        # 点数据插值曲面（右下角）- 多级回退策略确保成功
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        if len(optimizer.X_history) > 0:
            interpolation_success = False
            method_used = "Unknown"
            
            # 方法1: RBF插值（首选）
            try:
                from scipy.interpolate import Rbf
                
                print("🔄 尝试RBF插值...")
                rbf = Rbf(X_array_physical[:, 0], X_array_physical[:, 1], optimizer.y_history,
                         function='thin_plate', smooth=0)
                Z_interp = rbf(E_grid, nu_grid)
                
                # 检查结果有效性
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
                    points = np.column_stack([X_array_physical[:, 0], X_array_physical[:, 1]])
                    Z_interp = griddata(points, optimizer.y_history, (E_grid, nu_grid),
                                       method='linear', fill_value=np.nan)
                    
                    # 填充NaN值（使用nearest）
                    if np.any(np.isnan(Z_interp)):
                        Z_interp_nearest = griddata(points, optimizer.y_history, (E_grid, nu_grid),
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
                    points = np.column_stack([X_array_physical[:, 0], X_array_physical[:, 1]])
                    Z_interp = griddata(points, optimizer.y_history, (E_grid, nu_grid),
                                       method='cubic', fill_value=np.nan)
                    
                    # 填充NaN值
                    if np.any(np.isnan(Z_interp)):
                        Z_interp_nearest = griddata(points, optimizer.y_history, (E_grid, nu_grid),
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
                    points = np.column_stack([X_array_physical[:, 0], X_array_physical[:, 1]])
                    interp = CloughTocher2DInterpolator(points, optimizer.y_history)
                    Z_interp = interp(E_grid, nu_grid)
                    
                    # 处理NaN
                    if np.any(np.isnan(Z_interp)):
                        from scipy.interpolate import griddata
                        Z_interp_nearest = griddata(points, optimizer.y_history, (E_grid, nu_grid),
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
            
            # 最终回退：使用观测点散点图
            if not interpolation_success:
                print("⚠️ 所有插值方法失败，使用散点图")
                ax4.scatter(X_array_physical[:, 0], X_array_physical[:, 1], optimizer.y_history,
                           c=optimizer.y_history, cmap='viridis', s=100, edgecolors='black',
                           linewidth=1.5, depthshade=True)
                method_used = "Scatter (fallback)"
            
            # 绘制观测点（所有成功情况）
            if interpolation_success:
                ax4.scatter(X_array_physical[:, 0], X_array_physical[:, 1], optimizer.y_history,
                           c='red', s=100, edgecolors='black', linewidth=1.5,
                           depthshade=True, label='Observations', alpha=0.8)
                fig.colorbar(surf_interp, ax=ax4, shrink=0.5, aspect=5)
            
            # 标记最优点
            ax4.scatter([best_x[0]], [best_x[1]], [optimizer.y_history[best_idx]],
                       c='yellow', s=300, marker='*', edgecolors='black',
                       linewidth=2, label='Best', depthshade=True, zorder=10)
            
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
    print("🎯 贝叶斯优化材料参数标定")
    
    parser = argparse.ArgumentParser(description='Bayesian Optimization Material Parameter Calibration')
    parser.add_argument('--E-true', type=float, default=None)
    parser.add_argument('--nu-true', type=float, default=None)
    parser.add_argument('--n-initial', type=int, default=5)
    parser.add_argument('--n-iterations', type=int, default=50)
    parser.add_argument('--E-min', type=float, default=0.5000)
    parser.add_argument('--E-max', type=float, default=1.0000)
    parser.add_argument('--nu-min', type=float, default=0.3000)
    parser.add_argument('--nu-max', type=float, default=0.4000)
    parser.add_argument('--acquisition', type=str, default='ts',
                       choices=['ei', 'ucb', 'pi', 'ts', 'adaptive'])
    parser.add_argument('--xi', type=float, default=0.005)
    parser.add_argument('--no-visualization', action='store_true', help='禁用可视化')
    parser.add_argument('--save-plots', action='store_true', help='保存图表到文件')
    
    args = parser.parse_args()
    
    real_data_interface = RealDataInterface()
    calibrator = BayesianCalibration(
        real_data_interface=real_data_interface,
        E_bounds=(args.E_min, args.E_max),
        nu_bounds=(args.nu_min, args.nu_max),
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        acquisition=args.acquisition,
        xi=args.xi,
        enable_logging=True
    )
    
    real_data = None
    # 使用基于脚本位置的绝对路径
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
        E_true = args.E_true if args.E_true is not None else 0.1983
        nu_true = args.nu_true if args.nu_true is not None else 0.4795
        print(f"使用仿真真实参数: E={E_true}, nu={nu_true}")
    else:
        E_true, nu_true = args.E_true, args.nu_true
        print('使用真实数据')
    
    try:
        results = calibrator.run_calibration(real_data=real_data, E_true=E_true, nu_true=nu_true)
        
        if results is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            calibrator.save_results(results, f"calibration_results_{timestamp}.json", E_true, nu_true)
            calibrator.save_results(results, "optimization_results.json", E_true, nu_true)
            
            print(f"\n📊 标定结果:")
            print(f"   E = {results['best_params']['E']:.4f}, nu = {results['best_params']['nu']:.4f}")
            print(f"   误差 = {results['best_score']:.6f}")
            
            if E_true and nu_true:
                E_error = abs(results['best_params']['E'] - E_true)
                nu_error = abs(results['best_params']['nu'] - nu_true)
                print(f"   参数误差: E={E_error:.4f}, nu={nu_error:.4f}")
            
            # 可视化
            if not args.no_visualization:
                print(f"\n📊 生成可视化图表...")
                
                # 重建优化器以访问 GP 模型
                if calibrator.use_log_scale_E:
                    bounds = [calibrator.E_log_bounds, calibrator.nu_bounds]
                    param_transform_fn = calibrator._transform_params_to_physical
                else:
                    bounds = [calibrator.E_bounds, calibrator.nu_bounds]
                    param_transform_fn = None
                
                temp_optimizer = EnhancedBayesianOptimizer(
                    bounds=bounds, n_initial=calibrator.n_initial,
                    acquisition=calibrator.acquisition, xi=calibrator.xi,
                    logger=None, convergence_tolerance=calibrator.convergence_tolerance,
                    patience=calibrator.patience, param_transform_fn=param_transform_fn
                )
                
                # 重建优化历史
                for entry in results['optimization_history']:
                    params_physical = np.array(entry['params'])
                    if calibrator.use_log_scale_E:
                        params_scaled = np.array([np.log(params_physical[0]), params_physical[1]])
                    else:
                        params_scaled = params_physical
                    
                    score = entry['score']
                    temp_optimizer.gp.add_observation(params_scaled, score)
                    temp_optimizer.X_history.append(params_scaled)
                    temp_optimizer.y_history.append(score)
                
                if args.save_plots:
                    plot_dir = Path(f"calibration_plots_{timestamp}")
                    plot_dir = calibrator.vis_dir / plot_dir
                    plot_dir.mkdir(exist_ok=True)
                    
                    calibrator.create_optimization_summary(
                        results, save_path=str(plot_dir / "optimization_summary.png"),
                        E_true=E_true, nu_true=nu_true
                    )
                    
                    calibrator.plot_gp_surface(
                        temp_optimizer, save_path=str(plot_dir / "gp_surface.png")
                    )
                    
                    print(f"✓ 所有图表已保存至: {plot_dir}")
                else:
                    calibrator.create_optimization_summary(results, E_true=E_true, nu_true=nu_true)
                    calibrator.plot_gp_surface(temp_optimizer)
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断优化")
    except Exception as e:
        print(f"\n❌ 标定失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()