#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadratic Function Parameter Calibration using Bayesian Optimization
Objective: Approximate a cubic function with a quadratic function in a specified range

Author: AI Assistant
Function: Use Bayesian Optimization to efficiently find optimal quadratic function parameters
"""

import numpy as np
from typing import Tuple, List, Callable
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class GaussianProcess:
    """Gaussian Process for Bayesian Optimization"""
    
    def __init__(self, kernel_lengthscale: float = 1.0, kernel_variance: float = 1.0, 
                 noise_variance: float = 1e-6):
        """
        Initialize Gaussian Process
        
        Parameters:
            kernel_lengthscale: RBF kernel lengthscale parameter
            kernel_variance: RBF kernel variance parameter
            noise_variance: Observation noise variance
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
        """Predict mean and variance at test points"""
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
        
        # Compute kernel matrices
        K_obs = self.rbf_kernel(X_obs, X_obs) + self.noise_var * np.eye(len(X_obs))
        K_test_obs = self.rbf_kernel(X_test, X_obs)
        K_test = self.rbf_kernel(X_test, X_test)
        
        # Compute posterior mean and variance
        try:
            L = np.linalg.cholesky(K_obs)
            alpha = np.linalg.solve(L, y_obs)
            alpha = np.linalg.solve(L.T, alpha)
            
            mean = K_test_obs.dot(alpha)
            
            v = np.linalg.solve(L, K_test_obs.T)
            var = np.diag(K_test) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)  # Ensure positive variance
            
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky fails
            K_obs_inv = np.linalg.pinv(K_obs)
            mean = K_test_obs.dot(K_obs_inv).dot(y_obs)
            var = np.diag(K_test - K_test_obs.dot(K_obs_inv).dot(K_test_obs.T))
            var = np.maximum(var, 1e-10)
        
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
            acquisition: Acquisition function ('ei' for Expected Improvement)
            xi: Exploration parameter for acquisition function
        """
        self.bounds = bounds
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.xi = xi
        self.n_dimensions = len(bounds)
        
        # Initialize Gaussian Process
        self.gp = GaussianProcess()
        
        # Record optimization history
        self.X_history = []
        self.y_history = []
        self.best_y_history = []
        self.best_x_history = []
    
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
        """Expected Improvement acquisition function"""
        mean, var = self.gp.predict(X)
        std = np.sqrt(var)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            improvement = y_best - mean - self.xi
            Z = improvement / std
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0] = 0
        
        return ei
    
    def _optimize_acquisition(self, y_best: float) -> np.ndarray:
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
            
            # Optimize acquisition function (maximize EI = minimize -EI)
            def negative_ei(x):
                return -self._expected_improvement(x.reshape(1, -1), y_best)[0]
            
            result = minimize(
                negative_ei, x0, 
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.success and -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x
        
        # Fallback to random sampling if optimization fails
        if best_x is None:
            best_x = np.random.uniform(
                low=[bound[0] for bound in self.bounds],
                high=[bound[1] for bound in self.bounds]
            )
        
        return best_x
    
    def optimize(self, objective_function: Callable, max_evaluations: int = 50, 
                 verbose: bool = True) -> Tuple[np.ndarray, float]:
        """Execute Bayesian Optimization"""
        start_time = time.time()
        
        if verbose:
            print("🎯 Starting Bayesian Optimization...")
            print(f"Initial samples: {self.n_initial}, Max evaluations: {max_evaluations}")
            print(f"Acquisition function: {self.acquisition.upper()}")
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
        for iteration in range(max_evaluations - self.n_initial):
            print("="*30, "尝试", iteration, "="*30)
            # Find current best
            y_best = min(self.y_history)
            
            # Optimize acquisition function to get next point
            x_next = self._optimize_acquisition(y_best)
            
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
            
            if verbose and (iteration % 10 == 0 or iteration == max_evaluations - self.n_initial - 1):
                elapsed = time.time() - start_time
                current_best = min(self.y_history)
                best_x = self.X_history[np.argmin(self.y_history)]
                print(f"Eval {len(self.y_history):2d}: Best = {current_best:.6f} | "
                      f"Params: a={best_x[0]:.4f}, b={best_x[1]:.4f} | "
                      f"Current: f({x_next[0]:.4f}, {x_next[1]:.4f}) = {y_next:.6f} | "
                      f"Time: {elapsed:.2f}s")
        
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
        
        return best_params, best_score, optimization_history


def target_cubic_function(x: np.ndarray) -> np.ndarray:
    """Target cubic function: f(x) = 0.5x³ - 2x² + 3x + 1"""
    return 0.5 * x**2 - 2 * x + 1


def quadratic_approximation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Quadratic function to be calibrated: f(x) = ax² + bx + 1"""
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
    ax1.text(0.05, 0.95, f'Calibrated: f(x) = {a:.4f}x² + {b:.4f}x + 1\nR² = {metrics["r2"]:.4f}', 
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
    print(f"   Calibrated function: f(x) = {a:.4f}x² + {b:.4f}x + 1")
    
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
    print(f"\n📍 Key Points Function Value Comparison:")
    test_points = np.linspace(x_values[0], x_values[-1], 8)
    for x in test_points:
        target = target_cubic_function(np.array([x]))[0]
        predicted = quadratic_approximation(np.array([x]), best_params)[0]
        error = abs(predicted - target)
        rel_error = (error / abs(target) * 100) if abs(target) > 1e-10 else 0
        print(f"   x={x:6.2f}: Target={target:8.4f}, Predicted={predicted:8.4f}, "
              f"Error={error:6.4f} ({rel_error:5.1f}%)")
    
    # Display visualization charts
    print(f"\n📊 Generating visualization charts...")
    plot_optimization_results(x_values, target_values, best_params, metrics, 
                            optimizer.best_y_history, optimizer.X_history)
    
    # Plot GP posterior surface
    print(f"\n🔍 Generating Gaussian Process analysis...")
    plot_gp_surface(optimizer, [(-5, 5), (-5, 5)])


def run_optimization_suite():
    """Run complete optimization suite"""
    print("🎯 Quadratic Function Parameter Calibration - Bayesian Optimization")
    print("="*60)
    
    # Problem setup
    x_range = (-2, 3)
    param_bounds = [(-5, 5), (-5, 5)]
    
    print(f"\n📋 Problem Setup:")
    print(f"   Target function: f(x) = 0.5x³ - 2x² + 3x + 1")
    print(f"   Function to calibrate: f(x) = ax² + bx + 1")
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
    
    best_params, best_score = optimizer.optimize(objective_func, max_evaluations=30)
    
    # Evaluate results
    metrics = evaluate_solution(x_values, target_values, best_params)
    
    # Print detailed results
    print_detailed_results(x_values, target_values, best_params, metrics, optimizer)
    
    # Multiple trial validation
    print(f"\n🔄 Multiple Trial Validation (3 trials)...")
    trial_results = []
    
    for trial in range(3):
        trial_optimizer = BayesianOptimizer(
            bounds=param_bounds,
            n_initial=8,
            acquisition='ei',
            xi=0.01
        )
        trial_params, trial_score = trial_optimizer.optimize(objective_func, max_evaluations=30, verbose=False)
        trial_results.append((trial_params, trial_score))
        print(f"   Trial {trial+1}: a={trial_params[0]:.4f}, b={trial_params[1]:.4f}, Error={trial_score:.6f}")
    
    # Statistical analysis
    all_params = np.array([result[0] for result in trial_results])
    all_scores = np.array([result[1] for result in trial_results])
    
    print(f"\n📈 Multiple Trial Statistics:")
    print(f"   Parameter a: Mean={np.mean(all_params[:, 0]):.4f} ± {np.std(all_params[:, 0]):.4f}")
    print(f"   Parameter b: Mean={np.mean(all_params[:, 1]):.4f} ± {np.std(all_params[:, 1]):.4f}")
    print(f"   Error: Mean={np.mean(all_scores):.6f} ± {np.std(all_scores):.6f}")
    print(f"   Algorithm Stability: {'Excellent' if np.std(all_scores) < 1e-6 else 'Good' if np.std(all_scores) < 1e-4 else 'Fair'}")
    
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
    algorithm_comparison()
    
    print(f"\n🎉 Script execution completed!")
    print(f"💡 Tip: Bayesian Optimization is most effective for expensive objective functions")
    print(f"⚙️  Tip: Adjust n_initial and max_evaluations based on your computational budget")
    print(f"🔬 Tip: The GP posterior provides uncertainty estimates for better decision making") 