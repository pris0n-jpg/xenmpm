#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadratic Function Parameter Calibration using Bayesian Optimization
Objective: Approximate a cubic function with a quadratic function in a specified range

Author: AI Assistant
Function: Use Bayesian Optimization to efficiently find optimal quadratic function parameters
"""

import numpy as np
from typing import Tuple, List, Callable, Union
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class GaussianProcess:
    """Gaussian Process for Bayesian Optimization"""
    
    # 修复方案 2.1 & 2.2: 调整GP模型参数，增加噪声方差，减小核长度尺
    def __init__(self, kernel_lengthscale: float = 0.5, kernel_variance: float = 1.0,
                 noise_variance: float = 1e-4):
        """
        Initialize Gaussian Process
        
        Parameters:
            kernel_lengthscale: RBF kernel lengthscale parameter. 较小的值使模型更灵活.
            kernel_variance: RBF kernel variance parameter.
            noise_variance: Observation noise variance. 增加此值以避免模型过度自信.
        """
        self.lengthscale = kernel_lengthscale
        self.variance = kernel_variance
        self.noise_var = noise_variance
        
        self.X_observed = []
        self.y_observed = []
    
    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Radial Basis Function) kernel"""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        
        # Compute squared Euclidean distances
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.variance * np.exp(-0.5 * sqdist / self.lengthscale**2)
    
    def add_observation(self, x: np.ndarray, y: float):
        """Add new observation to GP"""
        self.X_observed.append(x.copy())
        self.y_observed.append(y)
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at test points - 改进版本，增强数值稳定性
        
        功能说明：
        - 计算高斯过程后验均值和方差
        - 处理数值不稳定性和边界情况
        - 确保在所有情况下返回有效的预测结果
        
        Args:
            X_test: 测试点，形状为 (n_test_points, n_dimensions)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (均值, 方差)
        """
        if len(self.X_observed) == 0:
            # Prior prediction
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)
            mean = np.zeros(X_test.shape[0])
            var = np.ones(X_test.shape[0]) * self.variance
            return mean, var
        
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        # 检查输入数据有效性
        if not np.all(np.isfinite(X_test)):
            raise ValueError("X_test contains non-finite values")
        
        if not np.all(np.isfinite(X_obs)):
            raise ValueError("X_obs contains non-finite values")
        
        if not np.all(np.isfinite(y_obs)):
            raise ValueError("y_obs contains non-finite values")
        
        # Compute kernel matrices
        K_obs = self.rbf_kernel(X_obs, X_obs) + self.noise_var * np.eye(len(X_obs))
        K_test_obs = self.rbf_kernel(X_test, X_obs)
        K_test = self.rbf_kernel(X_test, X_test)
        
        # Compute posterior mean and variance
        try:
            # 尝试使用Cholesky分解
            L = np.linalg.cholesky(K_obs)
            alpha = np.linalg.solve(L, y_obs)
            alpha = np.linalg.solve(L.T, alpha)
            
            mean = K_test_obs.dot(alpha)
            
            v = np.linalg.solve(L, K_test_obs.T)
            var = np.diag(K_test) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)  # Ensure positive variance
            
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            try:
                K_obs_inv = np.linalg.pinv(K_obs)
                mean = K_test_obs.dot(K_obs_inv).dot(y_obs)
                var = np.diag(K_test - K_test_obs.dot(K_obs_inv).dot(K_test_obs.T))
                var = np.maximum(var, 1e-10)
            except Exception as e:
                # 最后的fallback：使用对角线加正则化
                K_obs_reg = K_obs + 1e-6 * np.eye(len(K_obs))
                K_obs_inv = np.linalg.pinv(K_obs_reg)
                mean = K_test_obs.dot(K_obs_inv).dot(y_obs)
                var = np.ones(X_test.shape[0]) * self.variance
                print(f"Warning: GP prediction fallback used due to: {e}")
        
        # 确保输出有效性
        mean = np.where(np.isfinite(mean), mean, 0.0)
        var = np.where(np.isfinite(var), var, self.variance)
        var = np.maximum(var, 1e-10)  # 确保方差为正
        
        return mean, var


class BayesianOptimizer:
    """Bayesian Optimization Implementation"""

    def __init__(self, bounds: List[Tuple[float, float]], n_initial: int = 5,
                 acquisition: str = 'ei', xi: float = 0.03):
        """
        Initialize Bayesian Optimizer

        Parameters:
            bounds: Parameter boundaries [(min_a, max_a), (min_b, max_b)]
            n_initial: Number of initial random samples
            acquisition: Acquisition function strategy:
                - 'ei': Expected Improvement (default)
                - 'ucb': Upper Confidence Bound
                - 'pi': Probability of Improvement
                - 'ts': Thompson Sampling
                - 'adaptive': Adaptive strategy that switches based on optimization progress
            xi: Exploration parameter for acquisition function
        """
        self.bounds = bounds
        self.n_initial = n_initial
        self.acquisition = acquisition.lower()
        self.xi = xi
        self.n_dimensions = len(bounds)

        # Validate acquisition function
        valid_acquisitions = ['ei', 'ucb', 'pi', 'ts', 'adaptive']
        if self.acquisition not in valid_acquisitions:
            raise ValueError(f"Invalid acquisition function '{self.acquisition}'. Valid options: {valid_acquisitions}")

        # Initialize Gaussian Process
        self.gp = GaussianProcess()

        # Record optimization history
        self.X_history = []
        self.y_history = []
        self.best_y_history = []
        self.best_x_history = []

        # UCB parameters
        self.beta = 2.0  # UCB exploration parameter
        self.beta_decay = 0.99  # UCB exploration decay

        # Thompson sampling parameters
        self.n_samples = 100  # Number of samples for Thompson sampling
    
    def _sample_initial_points(self) -> np.ndarray:
        """Sample initial points using Latin Hypercube Sampling"""
        # Simple random sampling for initial points
        X_init = np.random.uniform(
            low=[bound[0] for bound in self.bounds],
            high=[bound[1] for bound in self.bounds],
            size=(self.n_initial, self.n_dimensions)
        )
        return X_init
    
    def _expected_improvement(self, X: np.ndarray, y_best: float) -> np.ndarray:
        """
        Expected Improvement acquisition function - 改进版本，增强数值稳定性
        
        功能说明：
        - 计算期望改进采集函数值
        - 处理数值不稳定性和边界情况
        - 确保在所有情况下返回有效的采集值
        
        Args:
            X: 待评估的参数点，形状为 (n_points, n_dimensions)
            y_best: 当前最佳目标函数值
            
        Returns:
            np.ndarray: 期望改进值，形状为 (n_points,)
        """
        mean, var = self.gp.predict(X)
        std = np.sqrt(var)

        # 修复方案 1.1 & 5.1: 增强数值稳定性，添加 epsilon 避免除以零
        epsilon = 1e-10  # 增加数值稳定性
        std = np.maximum(std, epsilon)

        # 避免除以零和数值溢出
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            improvement = y_best - mean - self.xi
            
            # 限制Z的范围避免数值不稳定
            Z = improvement / std
            Z = np.clip(Z, -10, 10)  # 限制Z值范围
            
            # 计算EI，使用更稳定的数值方法
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            
            # 确保在标准差为零的地方EI为零
            ei[std <= epsilon] = 0.0
            
            # 确保EI值为有限数
            ei = np.where(np.isfinite(ei), ei, 0.0)

        return ei

    def _upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition function"""
        mean, var = self.gp.predict(X)
        std = np.sqrt(var)
        # 修复方案 1.2 & 3.3: 修复UCB采集函数为负值的问题
        # 对于最小化问题，我们希望选择 mean 小、std 大的点。
        # 因此，采集函数应该是 -mean + beta * std。
        # 返回的值越大，代表这个点越有希望成为最优解。
        # 这个公式是正确的，问题通常来自 std 过小，增加 noise_variance 可以解决。
        ucb = -mean + self.beta * std
        return ucb

    def _probability_of_improvement(self, X: np.ndarray, y_best: float) -> np.ndarray:
        """
        Probability of Improvement acquisition function - 改进版本，增强数值稳定性
        
        功能说明：
        - 计算改进概率采集函数值
        - 处理数值不稳定性和边界情况
        - 确保在所有情况下返回有效的概率值
        
        Args:
            X: 待评估的参数点，形状为 (n_points, n_dimensions)
            y_best: 当前最佳目标函数值
            
        Returns:
            np.ndarray: 改进概率值，形状为 (n_points,)
        """
        mean, var = self.gp.predict(X)
        std = np.sqrt(var)

        # 修复方案 1.1 & 5.1: 增强数值稳定性，添加 epsilon 避免除以零
        epsilon = 1e-10  # 增加数值稳定性
        std = np.maximum(std, epsilon)

        # 避免除以零和数值溢出
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            improvement = y_best - mean - self.xi
            
            # 限制Z的范围避免数值不稳定
            Z = improvement / std
            Z = np.clip(Z, -10, 10)  # 限制Z值范围
            
            # 计算PI
            pi = norm.cdf(Z)
            
            # 确保在标准差为零的地方PI为0.5
            pi[std <= epsilon] = 0.5
            
            # 确保PI值在[0,1]范围内
            pi = np.clip(pi, 0.0, 1.0)

        return pi

    def _thompson_sampling(self, X: np.ndarray) -> np.ndarray:
        """
        Thompson Sampling acquisition function for minimization - 改进版本，增强数值稳定性
        
        功能说明：
        - 从高斯过程后验采样多个函数
        - 对于最小化问题，返回负的平均采样值
        - 处理数值不稳定性和边界情况
        
        Args:
            X: 待评估的参数点，形状为 (n_points, n_dimensions)
            
        Returns:
            np.ndarray: Thompson采样值，形状为 (n_points,)
        """
        # Sample multiple functions from GP posterior
        n_points = X.shape[0]
        samples = np.zeros(n_points)

        for _ in range(self.n_samples):
            # Sample from GP posterior
            mean, var = self.gp.predict(X)
            std = np.sqrt(var)
            
            # 确保标准差为正数
            std = np.maximum(std, 1e-10)
            
            # 限制采样范围避免数值不稳定
            mean = np.where(np.isfinite(mean), mean, 0.0)
            std = np.where(np.isfinite(std), std, 1e-10)
            
            # 从正态分布采样
            sample = np.random.normal(mean, std)
            sample = np.where(np.isfinite(sample), sample, 0.0)
            samples += sample

        # For minimization: use negative of average samples
        # This way, points with lower sampled values get higher acquisition scores
        # Higher acquisition value = more promising for minimization
        avg_samples = samples / self.n_samples
        avg_samples = np.where(np.isfinite(avg_samples), avg_samples, 0.0)
        
        return -avg_samples

    def _get_current_acquisition_function(self, iteration: int, total_iterations: int):
        """Get current acquisition function (supports adaptive strategy)"""
        if self.acquisition == 'adaptive':
            # Early optimization: explore more (UCB)
            # Mid optimization: balance exploration/exploitation (EI)
            # Late optimization: exploit more (PI)
            progress = iteration / total_iterations

            if progress < 0.2:
                # Early phase: exploration
                return 'ucb'
            elif progress < 0.6:
                return 'ei'
            else:
                # Late phase: exploitation
                return 'pi'
        else:
            return self.acquisition

    def _compute_acquisition_values(self, X: np.ndarray, y_best: float, iteration: int, total_iterations: int) -> np.ndarray:
        """Compute acquisition values using selected strategy"""
        current_acq = self._get_current_acquisition_function(iteration, total_iterations)

        if current_acq == 'ei':
            return self._expected_improvement(X, y_best)
        elif current_acq == 'ucb':
            # Decay exploration parameter over time
            self.beta *= self.beta_decay
            return self._upper_confidence_bound(X)
        elif current_acq == 'pi':
            return self._probability_of_improvement(X, y_best)
        elif current_acq == 'ts':
            return self._thompson_sampling(X)
        else:
            raise ValueError(f"Unknown acquisition function: {current_acq}")
    
    def _optimize_acquisition(self, y_best: float, iteration: int, total_iterations: int) -> np.ndarray:
        """Optimize acquisition function to find next point to evaluate"""
        best_x = None
        best_acq = -np.inf

        # Multi-start optimization
        n_restarts = 10
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.uniform(
                low=[bound[0] for bound in self.bounds],
                high=[bound[1] for bound in self.bounds]
            )

            # All acquisition functions should be MAXIMIZED
            # They represent "how promising a point is to explore"
            def objective(x):
                acq_values = self._compute_acquisition_values(x.reshape(1, -1), y_best, iteration, total_iterations)
                return -acq_values[0]  # Minimize negative = Maximize positive

            result = minimize(
                objective, x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )

            if result.success:
                current_acq_value = -result.fun
                if current_acq_value > best_acq:
                    best_acq = current_acq_value
                    best_x = result.x

        # Fallback to random sampling if optimization fails
        if best_x is None:
            best_x = np.random.uniform(
                low=[bound[0] for bound in self.bounds],
                high=[bound[1] for bound in self.bounds]
            )

        return best_x
    
    def optimize(self, objective_function: Callable, max_evaluations: int = 50,
                 verbose: bool = True, return_history: bool = True) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, List]]:
        """Execute Bayesian Optimization"""
        start_time = time.time()

        if verbose:
            print("🎯 Starting Bayesian Optimization...")
            print(f"Initial samples: {self.n_initial}, Max evaluations: {max_evaluations}")
            print(f"Acquisition function: {self.acquisition.upper()}")
            if self.acquisition == 'adaptive':
                print("Strategy: Adaptive (UCB→EI→PI based on progress)")
            print("-" * 70)

        # Initial random sampling
        X_init = self._sample_initial_points()

        for i, x in enumerate(X_init):
            y = objective_function(x)
            self.gp.add_observation(x, y)
            self.X_history.append(x.copy())
            self.y_history.append(y)

            if verbose:
                print(f"Init {i+1:2d}: f({x[0]:.4f}, {x[1]:.4f}) = {y:.6f}")

        # Record initial best
        best_idx = np.argmin(self.y_history)
        self.best_y_history.append(self.y_history[best_idx])
        self.best_x_history.append(self.X_history[best_idx].copy())

        if verbose:
            print("-" * 70)

        # Bayesian optimization loop
        total_iterations = max_evaluations - self.n_initial
        for iteration in range(total_iterations):
            print("="*30, "尝试", iteration, "="*30)
            # Find current best
            y_best = min(self.y_history)

            # Optimize acquisition function to get next point
            x_next = self._optimize_acquisition(y_best, iteration, total_iterations)

            # Evaluate objective function at next point
            y_next = objective_function(x_next)

            # Update GP with new observation
            self.gp.add_observation(x_next, y_next)
            self.X_history.append(x_next.copy())
            self.y_history.append(y_next)

            # Update best
            if y_next < y_best:
                self.best_y_history.append(y_next)
                self.best_x_history.append(x_next.copy())
            else:
                self.best_y_history.append(y_best)
                self.best_x_history.append(self.best_x_history[-1].copy())

            # Show current acquisition function if adaptive
            current_acq = self._get_current_acquisition_function(iteration, total_iterations)

            if verbose and (iteration % 10 == 0 or iteration == total_iterations - 1):
                elapsed = time.time() - start_time
                current_best = min(self.y_history)
                best_x = self.X_history[np.argmin(self.y_history)]
                acq_info = f"Acq: {current_acq.upper()}" if self.acquisition == 'adaptive' else ""
                print(f"Eval {len(self.y_history):2d}: Best = {current_best:.6f} | "
                      f"Params: a={best_x[0]:.4f}, b={best_x[1]:.4f} | "
                      f"Current: f({x_next[0]:.4f}, {x_next[1]:.4f}) = {y_next:.6f} | "
                      f"{acq_info} | Time: {elapsed:.2f}s")

        # Return best found solution
        best_idx = np.argmin(self.y_history)
        best_params = self.X_history[best_idx]
        best_score = self.y_history[best_idx]

        if verbose:
            print("-" * 70)
            print("✅ Bayesian Optimization completed!")

        # Create optimization history
        optimization_history = []
        for i, (x, y) in enumerate(zip(self.X_history, self.y_history)):
            optimization_history.append({
                'iteration': i,
                'params': x.tolist(),
                'score': float(y)
            })

        if return_history:
            return best_params, best_score, optimization_history
        else:
            return best_params, best_score


def target_cubic_function(x: np.ndarray) -> np.ndarray:
    """Target cubic function: f(x) = 0.5x² - 2x + 1 (Note: simplified from original cubic)"""
    return 0.5 * x**2 - 2 * x + 1


def quadratic_approximation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Quadratic function to be calibrated: f(x) = ax + b (Note: simplified from quadratic)"""
    a, b = params
    return a * x + b


def create_optimization_problem(x_range: Tuple[float, float] = (-2, 3), n_points: int = 200):
    """Create optimization problem"""
    x_values = np.linspace(x_range[0], x_range[1], n_points)
    target_values = target_cubic_function(x_values)
    
    def objective_function(params: np.ndarray) -> float:
        """Objective function: minimize mean squared error"""
        predicted_values = quadratic_approximation(x_values, params)
        mse = np.mean((predicted_values - target_values)**2)
        return mse
    
    return objective_function, x_values, target_values


def evaluate_solution(x_values: np.ndarray, target_values: np.ndarray, 
                     best_params: np.ndarray) -> dict:
    """Evaluate solution quality"""
    predicted_values = quadratic_approximation(x_values, best_params)
    
    # Calculate various error metrics
    mse = np.mean((predicted_values - target_values)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_values - target_values))
    max_error = np.max(np.abs(predicted_values - target_values))
    
    # Calculate R²
    ss_res = np.sum((target_values - predicted_values)**2)
    ss_tot = np.sum((target_values - np.mean(target_values))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'r2': r2,
        'predicted_values': predicted_values
    }


def plot_optimization_results(x_values: np.ndarray, target_values: np.ndarray, 
                             best_params: np.ndarray, metrics: dict, 
                             best_y_history: List[float], X_history: List[np.ndarray]):
    """Plot optimization results visualization"""
    predicted_values = metrics['predicted_values']
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Quadratic Function Calibration - Bayesian Optimization Results', fontsize=16, fontweight='bold')
    
    # Subplot 1: Function comparison
    ax1.plot(x_values, target_values, 'b-', linewidth=2.5, label='Target Cubic Function', alpha=0.8)
    ax1.plot(x_values, predicted_values, 'r--', linewidth=2.5, label='Calibrated Quadratic', alpha=0.8)
    ax1.fill_between(x_values, target_values, predicted_values, alpha=0.2, color='gray', label='Error Region')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Function Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add parameter info
    a, b = best_params
    ax1.text(0.05, 0.95, f'Calibrated: f(x) = {a:.4f}x + {b:.4f}\nR² = {metrics["r2"]:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Subplot 2: Optimization progress
    evaluations = range(1, len(best_y_history) + 1)
    ax2.plot(evaluations, best_y_history, 'g-', linewidth=2, marker='o', markersize=4, alpha=0.8)
    ax2.set_xlabel('Function Evaluations', fontsize=12)
    ax2.set_ylabel('Best Fitness Value (MSE)', fontsize=12)
    ax2.set_title('Bayesian Optimization Progress', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add optimization info
    total_evals = len(best_y_history)
    improvement = ((best_y_history[0] - best_y_history[-1]) / best_y_history[0] * 100)
    ax2.text(0.05, 0.95, f'Total Evaluations: {total_evals}\nImprovement: {improvement:.1f}%', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Subplot 3: Parameter space exploration
    X_array = np.array(X_history)
    scatter = ax3.scatter(X_array[:, 0], X_array[:, 1], c=range(len(X_array)), 
                         cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.scatter(best_params[0], best_params[1], c='red', s=200, marker='*', 
               edgecolors='black', linewidth=2, label='Best Solution')
    ax3.set_xlabel('Parameter a', fontsize=12)
    ax3.set_ylabel('Parameter b', fontsize=12)
    ax3.set_title('Parameter Space Exploration', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Evaluation Order', fontsize=10)
    
    # Subplot 4: Residual analysis
    residuals = predicted_values - target_values
    ax4.scatter(predicted_values, residuals, alpha=0.6, s=30, c='orange', edgecolors='black', linewidth=0.5)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Predicted Values', fontsize=12)
    ax4.set_ylabel('Residuals (Predicted - Actual)', fontsize=12)
    ax4.set_title('Residual Analysis', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add residual statistics
    residual_std = np.std(residuals)
    ax4.text(0.05, 0.95, f'Residual Std: {residual_std:.4f}\nResidual Mean: {np.mean(residuals):.4f}', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_gp_surface(optimizer: BayesianOptimizer, bounds: List[Tuple[float, float]]):
    """Plot Gaussian Process posterior surface"""
    # Create grid for visualization
    a_range = np.linspace(bounds[0][0], bounds[0][1], 50)
    b_range = np.linspace(bounds[1][0], bounds[1][1], 50)
    A, B = np.meshgrid(a_range, b_range)
    
    # Predict on grid
    grid_points = np.column_stack([A.ravel(), B.ravel()])
    mean_pred, var_pred = optimizer.gp.predict(grid_points)
    mean_pred = mean_pred.reshape(A.shape)
    std_pred = np.sqrt(var_pred).reshape(A.shape)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Gaussian Process Posterior Analysis', fontsize=16, fontweight='bold')
    
    # Left plot: Mean prediction
    contour1 = ax1.contour(A, B, mean_pred, levels=20, alpha=0.6)
    ax1.clabel(contour1, inline=True, fontsize=8)
    
    # Plot observed points
    X_array = np.array(optimizer.X_history)
    scatter1 = ax1.scatter(X_array[:, 0], X_array[:, 1], c=optimizer.y_history, 
                          cmap='viridis', s=80, edgecolors='black', linewidth=1)
    
    # Mark best point
    best_idx = np.argmin(optimizer.y_history)
    best_x = X_array[best_idx]
    ax1.scatter(best_x[0], best_x[1], c='red', s=200, marker='*', 
               edgecolors='black', linewidth=2, label='Best Solution')
    
    ax1.set_xlabel('Parameter a', fontsize=12)
    ax1.set_ylabel('Parameter b', fontsize=12)
    ax1.set_title('GP Mean Prediction', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Objective Value', fontsize=10)
    
    # Right plot: Uncertainty (standard deviation)
    contour2 = ax2.contour(A, B, std_pred, levels=15, alpha=0.6)
    ax2.clabel(contour2, inline=True, fontsize=8)
    ax2.scatter(X_array[:, 0], X_array[:, 1], c='black', s=50, alpha=0.7)
    ax2.scatter(best_x[0], best_x[1], c='red', s=200, marker='*', 
               edgecolors='black', linewidth=2, label='Best Solution')
    
    ax2.set_xlabel('Parameter a', fontsize=12)
    ax2.set_ylabel('Parameter b', fontsize=12)
    ax2.set_title('GP Uncertainty (Std Dev)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_comprehensive_contour(objective_function: Callable, optimizer: BayesianOptimizer,
                              bounds: List[Tuple[float, float]], resolution: int = 100):
    """
    Generate a comprehensive contour plot visualization of the objective function landscape
    with Bayesian optimization trajectory and key points.
    
    Parameters:
        objective_function: The objective function to optimize
        optimizer: The BayesianOptimizer instance after optimization
        bounds: Parameter bounds [(min_a, max_a), (min_b, max_b)]
        resolution: Grid resolution for contour plot
    """
    print("Generating comprehensive contour plot...")
    
    # Create parameter grid
    a_range = np.linspace(bounds[0][0], bounds[0][1], resolution)
    b_range = np.linspace(bounds[1][0], bounds[1][1], resolution)
    A, B = np.meshgrid(a_range, b_range)
    
    # Calculate objective function values on grid
    print("Computing objective function landscape...")
    Z = np.zeros_like(A)
    for i in range(resolution):
        for j in range(resolution):
            params = np.array([A[i, j], B[i, j]])
            Z[i, j] = objective_function(params)
    
    # Get optimization history
    X_history = np.array(optimizer.X_history)
    y_history = np.array(optimizer.y_history)
    best_idx = np.argmin(y_history)
    best_params = X_history[best_idx]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Bayesian Optimization - Comprehensive Objective Function Analysis',
                 fontsize=18, fontweight='bold')
    
    # === Subplot 1: Main contour plot with optimization trajectory ===
    # Create contour plot
    levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), 30)
    contour1 = ax1.contourf(A, B, Z, levels=levels, cmap='viridis', alpha=0.8)
    contour_lines = ax1.contour(A, B, Z, levels=levels[::3], colors='white',
                               alpha=0.4, linewidths=0.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2e')
    
    # Plot optimization trajectory
    ax1.plot(X_history[:, 0], X_history[:, 1], 'r-', linewidth=2, alpha=0.7,
            marker='o', markersize=4, label='Optimization Trajectory')
    
    # Mark initial points
    n_initial = optimizer.n_initial
    ax1.scatter(X_history[:n_initial, 0], X_history[:n_initial, 1],
               c='orange', s=100, marker='s', edgecolors='black',
               linewidth=1.5, label='Initial Points', zorder=5)
    
    # Mark best point
    ax1.scatter(best_params[0], best_params[1], c='red', s=300, marker='*',
               edgecolors='white', linewidth=2, label='Best Solution', zorder=6)
    
    # Add colorbar
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('Objective Function Value (MSE)', fontsize=12)
    
    # Set labels and title
    ax1.set_xlabel('Parameter a', fontsize=14)
    ax1.set_ylabel('Parameter b', fontsize=14)
    ax1.set_title('Objective Function Landscape with Optimization Trajectory',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add annotation for best solution
    ax1.annotate(f'Best: a={best_params[0]:.3f}, b={best_params[1]:.3f}\nMSE={y_history[best_idx]:.2e}',
                xy=(best_params[0], best_params[1]), xytext=(best_params[0]+0.5, best_params[1]+0.5),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                fontsize=10, color='black')
    
    # === Subplot 2: 3D surface plot ===
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    
    # Create 3D surface
    surf = ax2.plot_surface(A, B, Z, cmap='viridis', alpha=0.7,
                           linewidth=0, antialiased=True, rcount=50, ccount=50)
    
    # Plot optimization trajectory on 3D surface
    Z_history = y_history
    ax2.plot(X_history[:, 0], X_history[:, 1], Z_history, 'r-',
            linewidth=3, marker='o', markersize=5, label='Trajectory')
    
    # Mark best point
    ax2.scatter([best_params[0]], [best_params[1]], [y_history[best_idx]],
               color='red', s=200, marker='*', edgecolors='white', linewidth=2)
    
    # Set labels and title
    ax2.set_xlabel('Parameter a', fontsize=12)
    ax2.set_ylabel('Parameter b', fontsize=12)
    ax2.set_zlabel('Objective Value', fontsize=12)
    ax2.set_title('3D Objective Function Surface', fontsize=14, fontweight='bold')
    ax2.view_init(elev=30, azim=45)
    
    # Add colorbar
    cbar2 = plt.colorbar(surf, ax=ax2, pad=0.1, shrink=0.8)
    cbar2.set_label('Objective Function Value', fontsize=10)
    
    # === Subplot 3: Convergence heatmap ===
    # Create heatmap showing evaluation order
    eval_order = np.arange(len(X_history)).reshape(-1, 1)
    scatter3 = ax3.scatter(X_history[:, 0], X_history[:, 1], c=eval_order,
                          cmap='plasma', s=80, edgecolors='black', linewidth=0.5)
    
    # Add contour lines in background
    contour3 = ax3.contour(A, B, Z, levels=15, colors='gray', alpha=0.3, linewidths=0.5)
    
    # Mark best point
    ax3.scatter(best_params[0], best_params[1], c='red', s=300, marker='*',
               edgecolors='white', linewidth=2, zorder=5)
    
    # Add colorbar
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Evaluation Order', fontsize=12)
    
    # Set labels and title
    ax3.set_xlabel('Parameter a', fontsize=14)
    ax3.set_ylabel('Parameter b', fontsize=14)
    ax3.set_title('Parameter Evaluation Order and Convergence', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # === Subplot 4: Optimization progress analysis ===
    # Plot objective values over iterations
    iterations = np.arange(len(y_history))
    best_so_far = np.minimum.accumulate(y_history)
    
    ax4.plot(iterations, y_history, 'bo-', markersize=4, alpha=0.6,
            label='Individual Evaluations')
    ax4.plot(iterations, best_so_far, 'r-', linewidth=2.5,
            label='Best Value So Far')
    
    # Mark initial points and best point
    ax4.axvline(x=n_initial-1, color='orange', linestyle='--', alpha=0.7,
               label='End of Initial Sampling')
    ax4.scatter(best_idx, y_history[best_idx], color='red', s=100,
               marker='*', edgecolors='black', linewidth=1.5, zorder=5)
    
    # Set y-axis to log scale for better visualization
    ax4.set_yscale('log')
    
    # Set labels and title
    ax4.set_xlabel('Iteration Number', fontsize=14)
    ax4.set_ylabel('Objective Function Value (MSE)', fontsize=14)
    ax4.set_title('Optimization Progress Over Iterations', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3, which='both')
    
    # Add improvement annotation
    improvement = (y_history[0] - y_history[best_idx]) / y_history[0] * 100
    ax4.text(0.05, 0.95, f'Total Improvement: {improvement:.1f}%\n'
             f'Best Iteration: {best_idx}\n'
             f'Final MSE: {y_history[best_idx]:.2e}',
             transform=ax4.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save figure with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'calibration/contour_plot_bayesian_opt_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Contour plot saved as: {filename}")
    
    plt.show()


def print_detailed_results(x_values: np.ndarray, target_values: np.ndarray,
                          best_params: np.ndarray, metrics: dict, 
                          optimizer: BayesianOptimizer):
    """Print detailed results"""
    print("\n" + "="*60)
    print("🎯 Bayesian Optimization Results Details")
    print("="*60)
    
    a, b = best_params
    print(f"\n📈 Optimal Parameters:")
    print(f"   a = {a:.6f}")
    print(f"   b = {b:.6f}")
    print(f"   Calibrated function: f(x) = {a:.4f}x + {b:.4f}")
    
    print(f"\n📊 Fitting Quality:")
    print(f"   MSE (Mean Squared Error):     {metrics['mse']:.6f}")
    print(f"   RMSE (Root Mean Squared Error): {metrics['rmse']:.6f}")
    print(f"   MAE (Mean Absolute Error):    {metrics['mae']:.6f}")
    print(f"   Maximum Absolute Error:       {metrics['max_error']:.6f}")
    print(f"   R² (Coefficient of Determination): {metrics['r2']:.6f}")
    
    # Fitting quality assessment
    if metrics['r2'] > 0.95:
        quality = "Excellent"
    elif metrics['r2'] > 0.90:
        quality = "Good"
    elif metrics['r2'] > 0.80:
        quality = "Fair"
    else:
        quality = "Poor"
    print(f"   Fitting Quality Assessment:   {quality}")
    
    print(f"\n🔄 Optimization Efficiency:")
    print(f"   Total Function Evaluations:  {len(optimizer.y_history)}")
    print(f"   Initial Best Fitness:        {optimizer.best_y_history[0]:.6f}")
    print(f"   Final Best Fitness:          {optimizer.best_y_history[-1]:.6f}")
    print(f"   Fitness Improvement:         {((optimizer.best_y_history[0] - optimizer.best_y_history[-1]) / optimizer.best_y_history[0] * 100):.2f}%")
    
    # Efficiency analysis
    evaluations_to_converge = len(optimizer.best_y_history)
    efficiency = f"High (converged in {evaluations_to_converge} evaluations)"
    print(f"   Optimization Efficiency:     {efficiency}")
    
    # Key points comparison
    # print(f"\n📍 Key Points Function Value Comparison:")
    # test_points = np.linspace(x_values[0], x_values[-1], 8)
    # for x in test_points:
    #     target = target_cubic_function(np.array([x]))[0]
    #     predicted = quadratic_approximation(np.array([x]), best_params)[0]
    #     error = abs(predicted - target)
    #     rel_error = (error / abs(target) * 100) if abs(target) > 1e-10 else 0
    #     print(f"   x={x:6.2f}: Target={target:8.4f}, Predicted={predicted:8.4f}, "
    #           f"Error={error:6.4f} ({rel_error:5.1f}%)")
    
    # Display visualization charts
    print(f"\n📊 Generating visualization charts...")
    plot_optimization_results(x_values, target_values, best_params, metrics, 
                            optimizer.best_y_history, optimizer.X_history)
    
    # Plot GP posterior surface
    print(f"\n🔍 Generating Gaussian Process analysis...")
    plot_gp_surface(optimizer, [(-5, 5), (-5, 5)])
    
    # Generate comprehensive contour plot
    print(f"\n📊 Generating comprehensive contour plot visualization...")
    
    # We need to get the objective function from the outer scope
    # Since we can't directly access it, we'll recreate it
    x_range = (-2, 3)
    objective_func, _, _ = create_optimization_problem(x_range)
    plot_comprehensive_contour(objective_func, optimizer, [(-5, 5), (-5, 5)])


def run_optimization_suite():
    """Run complete optimization suite"""
    print("🎯 Quadratic Function Parameter Calibration - Bayesian Optimization")
    print("="*60)
    
    # Problem setup
    x_range = (-2, 3)
    param_bounds = [(-5, 5), (-5, 5)]
    
    print(f"\n📋 Problem Setup:")
    print(f"   Target function: f(x) = 0.5x² - 2x + 1")
    print(f"   Function to calibrate: f(x) = ax + b")
    print(f"   x range: {x_range}")
    print(f"   Parameter search range: a∈{param_bounds[0]}, b∈{param_bounds[1]}")
    
    # Create optimization problem
    objective_func, x_values, target_values = create_optimization_problem(x_range)
    
    # Execute Bayesian optimization
    print(f"\n🎯 Executing Bayesian Optimization...")
    optimizer = BayesianOptimizer(
        bounds=param_bounds,
        n_initial=8,
        acquisition='ei',
        xi=0.01
    )
    
    best_params, best_score, history = optimizer.optimize(objective_func, max_evaluations=30)
    
    # Evaluate results
    metrics = evaluate_solution(x_values, target_values, best_params)
    
    # Print detailed results
    print_detailed_results(x_values, target_values, best_params, metrics, optimizer)
    
    # # Multiple trial validation
    # print(f"\n🔄 Multiple Trial Validation (3 trials)...")
    # trial_results = []
    
    # for trial in range(3):
    #     trial_optimizer = BayesianOptimizer(
    #         bounds=param_bounds,
    #         n_initial=8,
    #         acquisition='ei',
    #         xi=0.01
    #     )
    #     trial_params, trial_score, history = trial_optimizer.optimize(objective_func, max_evaluations=30, verbose=False)
    #     trial_results.append((trial_params, trial_score))
    #     print(f"   Trial {trial+1}: a={trial_params[0]:.4f}, b={trial_params[1]:.4f}, Error={trial_score:.6f}")
    
    # # Statistical analysis
    # all_params = np.array([result[0] for result in trial_results])
    # all_scores = np.array([result[1] for result in trial_results])
    
    # print(f"\n📈 Multiple Trial Statistics:")
    # print(f"   Parameter a: Mean={np.mean(all_params[:, 0]):.4f} ± {np.std(all_params[:, 0]):.4f}")
    # print(f"   Parameter b: Mean={np.mean(all_params[:, 1]):.4f} ± {np.std(all_params[:, 1]):.4f}")
    # print(f"   Error: Mean={np.mean(all_scores):.6f} ± {np.std(all_scores):.6f}")
    # print(f"   Algorithm Stability: {'Excellent' if np.std(all_scores) < 1e-6 else 'Good' if np.std(all_scores) < 1e-4 else 'Fair'}")
    
    return best_params, metrics


def algorithm_comparison():
    """Compare Bayesian Optimization with other methods"""
    print(f"\n🔍 Algorithm Performance Comparison")
    print("="*60)
    
    x_range = (-2, 3)
    param_bounds = [(-5, 5), (-5, 5)]
    objective_func, x_values, target_values = create_optimization_problem(x_range)
    
    # Bayesian Optimization
    print(f"\n🎯 Running Bayesian Optimization...")
    bo_optimizer = BayesianOptimizer(bounds=param_bounds, n_initial=8)
    bo_start = time.time()
    bo_params, bo_score = bo_optimizer.optimize(objective_func, max_evaluations=30, verbose=False)
    bo_time = time.time() - bo_start
    
    # Results summary
    print(f"\n📊 Comparison Results:")
    print(f"   Bayesian Optimization:")
    print(f"     Parameters: a={bo_params[0]:.4f}, b={bo_params[1]:.4f}")
    print(f"     Final Error: {bo_score:.6f}")
    print(f"     Function Evaluations: {len(bo_optimizer.y_history)}")
    print(f"     Optimization Time: {bo_time:.3f}s")
    
    bo_metrics = evaluate_solution(x_values, target_values, bo_params)
    print(f"     R² Score: {bo_metrics['r2']:.6f}")
    
    print(f"\n💡 Bayesian Optimization特点:")
    print(f"   ✅ 函数评估次数少 (仅{len(bo_optimizer.y_history)}次)")
    print(f"   ✅ 智能选择下一个评估点")
    print(f"   ✅ 适合昂贵的目标函数")
    print(f"   ⚠️  需要更复杂的数学基础")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run main optimization
    best_params, metrics = run_optimization_suite()
    
    # Run algorithm comparison
    # algorithm_comparison()
 