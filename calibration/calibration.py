#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯优化材料参数标定脚本
独立运行的材料参数标定工具，支持真实数据导入和仿真数据对比
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime
import sys
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Matplotlib not available, visualization features will be disabled")

# 添加必要的路径
def add_calibration_path():
    """添加calibration目录到Python路径"""
    calibration_dir = Path(__file__).parent  # 当前目录就是calibration
    if str(calibration_dir) not in sys.path:
        sys.path.insert(0, str(calibration_dir))

def add_xengym_path():
    """添加xengym目录到Python路径"""
    xengym_dir = Path(__file__).parent.parent / "xengym"
    if str(xengym_dir) not in sys.path:
        sys.path.insert(0, str(xengym_dir))

# 添加路径
add_calibration_path()
add_xengym_path()

try:
    from bayesian_demo import BayesianOptimizer, GaussianProcess
    from fem_processor import process_gel_data
    from xengym.render.calibScene import create_calibration_scene
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在正确的目录中运行此脚本")
    sys.exit(1)


class RealDataInterface:
    """真实数据导入接口"""
    
    def __init__(self):
        self.real_data_cache = {}
    
    def load_from_json(self, file_path: Union[str, Path]) -> Dict:
        """从JSON文件加载真实数据"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"真实数据文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ 从JSON文件加载真实数据: {file_path}")
            return data
        except Exception as e:
            raise ValueError(f"JSON文件解析失败: {e}")
    
    def load_from_pickle(self, file_path: Union[str, Path]) -> Dict:
        """从Pickle文件加载真实数据"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"真实数据文件不存在: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ 从Pickle文件加载真实数据: {file_path}")
            return data
        except Exception as e:
            raise ValueError(f"Pickle文件解析失败: {e}")
    
    def load_from_directory(self, dir_path: Union[str, Path], pattern: str = "*.json") -> Dict:
        """从目录加载多个真实数据文件"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {dir_path}")
        
        combined_data = {}
        for file_path in dir_path.glob(pattern):
            try:
                if file_path.suffix.lower() == '.json':
                    data = self.load_from_json(file_path)
                elif file_path.suffix.lower() == '.pkl':
                    data = self.load_from_pickle(file_path)
                else:
                    continue
                
                # 合并数据
                for obj_name, obj_data in data.items():
                    if obj_name in combined_data:
                        combined_data[obj_name].update(obj_data)
                    else:
                        combined_data[obj_name] = obj_data
                        
            except Exception as e:
                print(f"⚠️ 跳过文件 {file_path}: {e}")
        
        print(f"✓ 从目录加载了 {len(combined_data)} 个物体的数据")
        return combined_data
    
    def create_real_raw_data(self, E_true: float = 0.1983, nu_true: float = 0.4795, coef_true: float = 0.200) -> Dict:
        """创建仿真真实数据（用于测试）"""
        print(f"🎯 创建仿真真实数据 (E={E_true}, nu={nu_true}, coef={coef_true})")
        
        # 创建标定场景
        scene = self._create_calibration_scene()
        if scene is None:
            raise RuntimeError("无法创建标定场景")
        
        # 使用真实参数生成数据
        try:
            real_data = scene.calibrate_with_parameters(E_true, nu_true, coef_true)
            print(f"✓ 仿真真实数据创建完成")
            return real_data
        except Exception as e:
            raise RuntimeError(f"创建仿真真实数据失败: {e}")
    
    def _create_calibration_scene(self):
        """创建标定场景"""
        # 尝试多个可能的路径
        possible_paths = [
            # Path("../xengym/assets/obj"),
            # Path(__file__).parent.parent / "xengym" / "assets" / "obj",
            # Path("/home/czl/Downloads/workspace/xengym/xengym/assets/obj"),
            Path("/home/czl/Downloads/workspace/xengym/calibration/obj")
        ]
        
        object_files = []
        for path in possible_paths:
            if path.exists():
                stl_files = list(path.glob("*.STL"))
                # stl_files = list(path.glob("*.obj"))
                if stl_files:
                    object_files = [str(f) for f in stl_files[:4]]  
                    break
        
        if not object_files:
            print("❌ 无法找到STL文件")
            return None
        
        try:
            scene = create_calibration_scene(
                object_files=object_files,
                visible=False,
                sensor_visible=False
            )
            print(f"✓ 标定场景创建完成，使用 {len(object_files)} 个物体")
            return scene
        except Exception as e:
            print(f"❌ 标定场景创建失败: {e}")
            return None


class BayesianCalibration:
    """贝叶斯优化标定器"""
    
    def __init__(self,
                 real_data_interface: RealDataInterface,
                 E_bounds: Tuple[float, float] = (0.1000, 0.3000),
                 nu_bounds: Tuple[float, float] = (0.4000, 0.5000),
                 coef_bounds: Tuple[float, float] = (0.0000, 1.0000),
                 n_initial: int = 15,  # 增加初始样本数，适应3D参数空间
                 n_iterations: int = 30,  # 增加迭代次数
                 acquisition: str = 'adaptive',  # 添加采集函数选择
                 xi: float = 0.01):  # 添加探索参数
        
        self.real_data_interface = real_data_interface
        self.E_bounds = E_bounds
        self.nu_bounds = nu_bounds
        self.coef_bounds = coef_bounds
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.acquisition = acquisition
        self.xi = xi
        
        # 创建标定场景
        self.scene = real_data_interface._create_calibration_scene()
        if self.scene is None:
            raise RuntimeError("无法创建标定场景")
        
        # 优化历史
        self.optimization_history = []
    
    def calculate_calibration_error(self, sim_data: Dict, real_data: Dict) -> float:
        """计算标定误差
        - 依据 traj.json 的层级结构对齐: object → trajectory → step
        - 分别计算 marker/force/depth 的 RMSE
        - 健壮处理缺失键、NaN、形状不一致
        - 以加权和归一化返回综合误差
        """
        def rmse(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a)
            b = np.asarray(b)
            # 对齐形状：仅当元素总数相等时允许 reshape
            if a.shape != b.shape:
                if a.size == b.size:
                    b = b.reshape(a.shape)
                else:
                    # 形状完全不匹配，放弃该项评价
                    return np.nan
            mask = np.isfinite(a) & np.isfinite(b)
            if not np.any(mask):
                return np.nan
            diff = a[mask] - b[mask]
            return float(np.sqrt(np.mean(diff ** 2)))
        
        total_error = 0.0
        total_weight = 0.0
        
        # 权重：根据项目实际关注度可调整
        weight_marker = 100   # 标记位移场
        weight_force = 10/3    # 三维力
        
        # 遍历两侧共同包含的对象/轨迹/步
        for obj_name in real_data.keys() & sim_data.keys():
            real_obj = real_data[obj_name]
            sim_obj = sim_data[obj_name]
            if not isinstance(real_obj, dict) or not isinstance(sim_obj, dict):
                continue
            for traj_name in real_obj.keys() & sim_obj.keys():
                real_traj = real_obj[traj_name]
                sim_traj = sim_obj[traj_name]
                if not isinstance(real_traj, dict) or not isinstance(sim_traj, dict):
                    continue
                for step_name in real_traj.keys() & sim_traj.keys():
                    real_step = real_traj.get(step_name, {}) or {}
                    sim_step = sim_traj.get(step_name, {}) or {}
                    if not isinstance(real_step, dict) or not isinstance(sim_step, dict):
                        continue
                    
                    # 1) Marker 位移误差 (HxWx2)
                    if 'marker_displacement' in real_step and 'marker_displacement' in sim_step:
                        marker_err = rmse(real_step['marker_displacement'], sim_step['marker_displacement'])
                        if np.isfinite(marker_err):
                            total_error += weight_marker * marker_err
                            total_weight += weight_marker
                    
                    # 2) 力误差 (3,)
                    if 'force_xyz' in real_step and 'force_xyz' in sim_step:
                        force_err = rmse(real_step['force_xyz'], sim_step['force_xyz'])
                        if np.isfinite(force_err):
                            total_error += weight_force * force_err
                            total_weight += weight_force
        
        if total_weight <= 0:
            # 无可比较项时返回一个大值，避免误导优化
            return float('inf')
        return total_error / total_weight
    
    def objective_function(self, params: np.ndarray, real_data: Dict) -> float:
        """目标函数 - 使用改进的误差计算方法"""
        E, nu, coef = params
        
        try:
            # 使用材料参数进行标定
            sim_data = self.scene.calibrate_with_parameters(E, nu, coef)
            
            # 计算综合误差
            error = self.calculate_calibration_error(sim_data, real_data)
            print(f"   参数 E={E:.4f}, nu={nu:.4f}, coef={coef:.3f}, 综合误差={error:.6f}")

            return error
            
        except Exception as e:
            print(f"   参数 E={E:.4f}, nu={nu:.4f}, coef={coef:.3f} 评估失败: {e}")
            return float('inf')
    
    def run_calibration(self, 
                       real_data: Optional[Dict] = None,
                       E_true: Optional[float] = None,
                       nu_true: Optional[float] = None,
                       coef_true: Optional[float] = None) -> Dict:
        """运行贝叶斯优化标定"""
        
        print("🎯 开始贝叶斯优化标定")
        print("=" * 60)
        
        # 处理真实数据
        if real_data is None:
            if E_true is None or nu_true is None:
                print("❌ 需要提供真实数据或真实参数")
                return None
            
            # 使用仿真真实数据
            real_data = self.real_data_interface.create_real_raw_data(E_true, nu_true, coef_true)
        
        print(f"✓ 真实数据包含 {len(real_data)} 个物体")
        
        # 创建目标函数
        def objective(params):
            return self.objective_function(params, real_data)
        
        # 设置参数边界
        bounds = [
            self.E_bounds,  # E边界
            self.nu_bounds,   # nu边界
            self.coef_bounds  # 系数边界
        ]
        
        # 创建贝叶斯优化器
        optimizer = BayesianOptimizer(
            bounds=bounds,
            n_initial=self.n_initial,
            acquisition=self.acquisition,
            xi=self.xi
        )
        
        # 运行优化
        print(f"\n📋 优化设置:")
        print(f"   E范围: {self.E_bounds} (精度: 4位小数)")
        print(f"   nu范围: {self.nu_bounds} (精度: 4位小数)")
        print(f"   coef范围: {self.coef_bounds} (精度: 3位小数)")
        print(f"   初始样本: {self.n_initial}")
        print(f"   迭代次数: {self.n_iterations}")
        print(f"   采集函数: {self.acquisition}")
        print(f"   探索参数 xi: {self.xi}")
        
        print(f"\n🔄 开始优化...")
        best_params, best_score, optimization_history = optimizer.optimize(
            objective_function=objective,
            max_evaluations=self.n_initial + self.n_iterations,
            verbose=True
        )
        
        # 保存优化历史
        self.optimization_history = optimization_history
        
        # 创建结果
        result = {
            'best_params': {
                'E': round(float(best_params[0]), 4),
                'nu': round(float(best_params[1]), 4),
                'coef': round(float(best_params[2]), 3)
            },
            'best_score': float(best_score),
            'optimization_history': optimization_history,
            'timestamp': datetime.now().isoformat(),
            'n_evaluations': len(optimization_history)
        }
        
        print(f"\n🎉 优化完成!")
        print(f"   最优参数: E={result['best_params']['E']:.4f}, nu={result['best_params']['nu']:.4f}, coef={result['best_params']['coef']:.3f}")
        print(f"   最小误差: {result['best_score']:.6f}")
        print(f"   评估次数: {result['n_evaluations']}")
        
        return result
    
    def save_results(self, results: Dict, file_path: Union[str, Path], E_true: Optional[float] = None, nu_true: Optional[float] = None, coef_true: Optional[float] = None):
        """保存标定结果"""
        file_path = "results"/Path(file_path)
        
        # 添加真实值到结果中（如果提供）
        if E_true is not None and nu_true is not None:
            results['true_params'] = {
                'E': E_true,
                'nu': nu_true,
                'coef': coef_true
            }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ 结果保存至: {file_path}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def load_results(self, file_path: Union[str, Path]) -> Dict:
        """加载标定结果"""
        file_path = "results"/Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"结果文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"✓ 结果加载自: {file_path}")
            return results
        except Exception as e:
            raise ValueError(f"加载结果失败: {e}")
    
    def plot_optimization_history(self, results: Dict, save_path: Optional[str] = None, 
                                  E_true: Optional[float] = None, nu_true: Optional[float] = None, 
                                  coef_true: Optional[float] = None):
        """Plot optimization history with comprehensive visualization for 3 parameters"""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Matplotlib not available, skipping visualization")
            return
        
        history = results['optimization_history']
        iterations = range(len(history))
        scores = [h['score'] for h in history]
        E_values = [h['params'][0] for h in history]
        nu_values = [h['params'][1] for h in history]
        coef_values = [h['params'][2] for h in history]
        
        # 从结果中读取真实值（如果存在）
        if 'true_params' in results and E_true is None and nu_true is None:
            E_true = results['true_params'].get('E')
            nu_true = results['true_params'].get('nu')
            coef_true = results['true_params'].get('coef')
        
        # Create figure with subplots - 使用3x3布局来容纳更多图表
        fig = plt.figure(figsize=(18, 15))
        fig.suptitle('Bayesian Optimization Calibration Visualization (3 Parameters)', fontsize=16, fontweight='bold')
        
        # 1. Optimization progress
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(iterations, scores, 'o-', linewidth=2, markersize=4)
        plt.xlabel('Iteration', fontsize=10)
        plt.ylabel('Objective Function Value', fontsize=10)
        plt.title('Optimization Progress', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 2. Parameter convergence
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(iterations, E_values, 'o-', label='Young\'s Modulus E', linewidth=2, markersize=3)
        plt.plot(iterations, nu_values, 's-', label='Poisson\'s Ratio ν', linewidth=2, markersize=3)
        plt.plot(iterations, coef_values, 'd-', label='Nonlinear Coefficient', linewidth=2, markersize=3)
        
        # 在参数收敛图中添加真实值
        if E_true is not None:
            plt.axhline(y=E_true, color='red', linestyle='--', linewidth=2, 
                       label=f'True E={E_true:.4f}')
        if nu_true is not None:
            plt.axhline(y=nu_true, color='blue', linestyle='--', linewidth=2, 
                       label=f'True ν={nu_true:.4f}')
        if coef_true is not None:
            plt.axhline(y=coef_true, color='green', linestyle='--', linewidth=2, 
                       label=f'True coef={coef_true:.3f}')
            
        plt.xlabel('Iteration', fontsize=10)
        plt.ylabel('Parameter Value', fontsize=10)
        plt.title('Parameter Convergence', fontsize=12, fontweight='bold')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 3. 3D Parameter space exploration
        ax3 = plt.subplot(3, 3, 3, projection='3d')
        scatter = ax3.scatter(E_values, nu_values, coef_values, c=scores, cmap='viridis', s=30, alpha=0.7)
        
        # Mark best point
        best_idx = np.argmin(scores)
        ax3.scatter(E_values[best_idx], nu_values[best_idx], coef_values[best_idx], 
                   c='red', s=200, marker='*', edgecolors='black', linewidth=2, label='Best Parameters')
        
        # 在3D参数空间中添加真实值
        if E_true is not None and nu_true is not None and coef_true is not None:
            ax3.scatter(E_true, nu_true, coef_true, c='blue', s=150, marker='o', 
                       label='True Values', edgecolors='black', linewidths=1.5)
            
        ax3.set_xlabel('Young\'s Modulus E', fontsize=10)
        ax3.set_ylabel('Poisson\'s Ratio ν', fontsize=10)
        ax3.set_zlabel('Nonlinear Coefficient', fontsize=10)
        ax3.set_title('3D Parameter Space', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        
        # 4. E vs nu 2D projection
        ax4 = plt.subplot(3, 3, 4)
        scatter4 = plt.scatter(E_values, nu_values, c=scores, cmap='viridis', s=40, alpha=0.7)
        plt.colorbar(scatter4, label='Score', shrink=0.8)
        plt.scatter(E_values[best_idx], nu_values[best_idx], c='red', s=200, 
                   marker='*', edgecolors='black', linewidth=2, label='Best')
        if E_true is not None and nu_true is not None:
            plt.scatter(E_true, nu_true, c='blue', s=150, marker='o', 
                       label='True', edgecolors='black', linewidths=1.5)
        plt.xlabel('Young\'s Modulus E', fontsize=10)
        plt.ylabel('Poisson\'s Ratio ν', fontsize=10)
        plt.title('E vs ν Projection', fontsize=12, fontweight='bold')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 5. E vs coef 2D projection
        ax5 = plt.subplot(3, 3, 5)
        scatter5 = plt.scatter(E_values, coef_values, c=scores, cmap='viridis', s=40, alpha=0.7)
        plt.colorbar(scatter5, label='Score', shrink=0.8)
        plt.scatter(E_values[best_idx], coef_values[best_idx], c='red', s=200, 
                   marker='*', edgecolors='black', linewidth=2, label='Best')
        if E_true is not None and coef_true is not None:
            plt.scatter(E_true, coef_true, c='blue', s=150, marker='o', 
                       label='True', edgecolors='black', linewidths=1.5)
        plt.xlabel('Young\'s Modulus E', fontsize=10)
        plt.ylabel('Nonlinear Coefficient', fontsize=10)
        plt.title('E vs Coef Projection', fontsize=12, fontweight='bold')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 6. nu vs coef 2D projection
        ax6 = plt.subplot(3, 3, 6)
        scatter6 = plt.scatter(nu_values, coef_values, c=scores, cmap='viridis', s=40, alpha=0.7)
        plt.colorbar(scatter6, label='Score', shrink=0.8)
        plt.scatter(nu_values[best_idx], coef_values[best_idx], c='red', s=200, 
                   marker='*', edgecolors='black', linewidth=2, label='Best')
        if nu_true is not None and coef_true is not None:
            plt.scatter(nu_true, coef_true, c='blue', s=150, marker='o', 
                       label='True', edgecolors='black', linewidths=1.5)
        plt.xlabel('Poisson\'s Ratio ν', fontsize=10)
        plt.ylabel('Nonlinear Coefficient', fontsize=10)
        plt.title('ν vs Coef Projection', fontsize=12, fontweight='bold')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 7. Cumulative best score
        ax7 = plt.subplot(3, 3, 7)
        best_scores = [min(scores[:i+1]) for i in range(len(scores))]
        plt.plot(iterations, best_scores, 'o-', color='green', linewidth=2, markersize=4)
        plt.xlabel('Iteration', fontsize=10)
        plt.ylabel('Best Score Achieved', fontsize=10)
        plt.title('Cumulative Best Score', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 8. Parameter distribution (分别显示)
        ax8 = plt.subplot(3, 3, 8)
        n_bins = min(15, len(E_values)//3)  # 动态调整bins数量
        plt.hist(E_values, bins=n_bins, alpha=0.7, label='E', density=True, color='red')
        plt.hist(nu_values, bins=n_bins, alpha=0.7, label='ν', density=True, color='blue')
        plt.hist(coef_values, bins=n_bins, alpha=0.7, label='coef', density=True, color='green')
        
        # 在参数分布中添加真实值
        if E_true is not None:
            plt.axvline(x=E_true, color='red', linestyle='--', linewidth=2, 
                       label=f'True E={E_true:.4f}')
        if nu_true is not None:
            plt.axvline(x=nu_true, color='blue', linestyle='--', linewidth=2, 
                       label=f'True ν={nu_true:.4f}')
        if coef_true is not None:
            plt.axvline(x=coef_true, color='green', linestyle='--', linewidth=2, 
                       label=f'True coef={coef_true:.3f}')
            
        plt.xlabel('Parameter Value', fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.title('Parameter Distribution', fontsize=12, fontweight='bold')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 9. 参数相关性分析
        ax9 = plt.subplot(3, 3, 9)
        # 计算参数间的相关系数
        import pandas as pd
        param_df = pd.DataFrame({
            'E': E_values,
            'ν': nu_values, 
            'coef': coef_values,
            'score': scores
        })
        correlation_matrix = param_df.corr()
        
        # 绘制相关性热图
        im = ax9.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # 添加文本标注
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
            print(f"✓ Optimization history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_real_time_optimization(self, real_data: Dict, 
                                   E_true: Optional[float] = None, 
                                   nu_true: Optional[float] = None,
                                   coef_true: Optional[float] = None,
                                   update_interval: int = 5):
        """Real-time optimization visualization for 3 parameters"""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Matplotlib not available, skipping real-time visualization")
            return
        
        # Setup the figure with 2x3 layout for 3 parameters
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Real-time Bayesian Optimization (3 Parameters)', fontsize=16, fontweight='bold')
        
        ax1 = plt.subplot(2, 3, 1)  # Optimization progress
        ax2 = plt.subplot(2, 3, 2)  # Parameter convergence
        ax3 = plt.subplot(2, 3, 3, projection='3d')  # 3D parameter space
        ax4 = plt.subplot(2, 3, 4)  # Best score progress
        ax5 = plt.subplot(2, 3, 5)  # E vs nu projection
        ax6 = plt.subplot(2, 3, 6)  # Parameter distributions
        
        # Initialize data storage
        self.real_time_iterations = []
        self.real_time_scores = []
        self.real_time_E_values = []
        self.real_time_nu_values = []
        self.real_time_coef_values = []
        self.real_time_best_scores = []
        
        # Wrap the objective function to capture data
        def tracked_objective(params):
            score = self.objective_function(params, real_data)
            
            # Store data
            self.real_time_iterations.append(len(self.real_time_iterations))
            self.real_time_scores.append(score)
            self.real_time_E_values.append(params[0])
            self.real_time_nu_values.append(params[1])
            self.real_time_coef_values.append(params[2])
            self.real_time_best_scores.append(min(self.real_time_scores))
            
            # Update plots every update_interval iterations
            if len(self.real_time_iterations) % update_interval == 0:
                self._update_real_time_plots_3d(ax1, ax2, ax3, ax4, ax5, ax6, E_true, nu_true, coef_true)
                plt.pause(0.1)
            
            return score
        
        # Create objective function for optimization
        def objective(params):
            return tracked_objective(params)
        
        # Setup and run optimization
        bounds = [self.E_bounds, self.nu_bounds, self.coef_bounds]
        optimizer = BayesianOptimizer(
            bounds=bounds,
            n_initial=self.n_initial,
            acquisition=self.acquisition,  # 使用一致的采集函数
            xi=self.xi
        )
        
        print("🎯 Starting real-time optimization visualization...")
        print(f"Update interval: {update_interval} iterations")
        
        # Run optimization
        best_params, best_score, optimization_history = optimizer.optimize(
            objective_function=objective,
            max_evaluations=self.n_initial + self.n_iterations,
            verbose=True
        )
        
        # Final plot update
        self._update_real_time_plots_3d(ax1, ax2, ax3, ax4, ax5, ax6, E_true, nu_true, coef_true)
        plt.show()
        
        return best_params, best_score, optimization_history
    
    def _update_real_time_plots_3d(self, ax1, ax2, ax3, ax4, ax5, ax6, E_true, nu_true, coef_true):
        """Update real-time plots for 3 parameters"""
        # Clear axes
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.clear()
        
        # 1. Optimization progress
        ax1.plot(self.real_time_iterations, self.real_time_scores, 'o-', linewidth=2, markersize=4)
        ax1.set_xlabel('Iteration', fontsize=10)
        ax1.set_ylabel('Objective Function Value', fontsize=10)
        ax1.set_title('Optimization Progress', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Parameter convergence
        ax2.plot(self.real_time_iterations, self.real_time_E_values, 'o-', label='E', linewidth=2, markersize=3)
        ax2.plot(self.real_time_iterations, self.real_time_nu_values, 's-', label='ν', linewidth=2, markersize=3)
        ax2.plot(self.real_time_iterations, self.real_time_coef_values, 'd-', label='coef', linewidth=2, markersize=3)
        if E_true is not None:
            ax2.axhline(y=E_true, color='red', linestyle='--', label='True E')
        if nu_true is not None:
            ax2.axhline(y=nu_true, color='blue', linestyle='--', label='True ν')
        if coef_true is not None:
            ax2.axhline(y=coef_true, color='green', linestyle='--', label='True coef')
        ax2.set_xlabel('Iteration', fontsize=10)
        ax2.set_ylabel('Parameter Value', fontsize=10)
        ax2.set_title('Parameter Convergence', fontsize=12)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. 3D Parameter space
        ax3.scatter(self.real_time_E_values, self.real_time_nu_values, self.real_time_coef_values,
                   c=self.real_time_scores, cmap='viridis', s=30, alpha=0.7)
        if E_true is not None and nu_true is not None and coef_true is not None:
            ax3.scatter(E_true, nu_true, coef_true, c='blue', s=150, marker='o', label='True Values')
        ax3.set_xlabel('E', fontsize=10)
        ax3.set_ylabel('ν', fontsize=10)
        ax3.set_zlabel('coef', fontsize=10)
        ax3.set_title('3D Parameter Space', fontsize=12)
        if E_true is not None and nu_true is not None and coef_true is not None:
            ax3.legend(fontsize=8)
        
        # 4. Best score progress
        ax4.plot(self.real_time_iterations, self.real_time_best_scores, 'o-', color='green', linewidth=2, markersize=4)
        ax4.set_xlabel('Iteration', fontsize=10)
        ax4.set_ylabel('Best Score', fontsize=10)
        ax4.set_title('Best Score Progress', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # 5. E vs nu projection
        ax5.scatter(self.real_time_E_values, self.real_time_nu_values, 
                   c=self.real_time_scores, cmap='viridis', s=40, alpha=0.7)
        if E_true is not None and nu_true is not None:
            ax5.scatter(E_true, nu_true, c='blue', s=150, marker='o', label='True')
        ax5.set_xlabel('E', fontsize=10)
        ax5.set_ylabel('ν', fontsize=10)
        ax5.set_title('E vs ν Projection', fontsize=12)
        if E_true is not None and nu_true is not None:
            ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Parameter distributions
        if len(self.real_time_E_values) > 5:  # Only plot when we have enough data
            n_bins = min(10, len(self.real_time_E_values)//2)
            ax6.hist(self.real_time_E_values, bins=n_bins, alpha=0.7, label='E', density=True, color='red')
            ax6.hist(self.real_time_nu_values, bins=n_bins, alpha=0.7, label='ν', density=True, color='blue')
            ax6.hist(self.real_time_coef_values, bins=n_bins, alpha=0.7, label='coef', density=True, color='green')
            if E_true is not None:
                ax6.axvline(x=E_true, color='red', linestyle='--', label='True E')
            if nu_true is not None:
                ax6.axvline(x=nu_true, color='blue', linestyle='--', label='True ν')
            if coef_true is not None:
                ax6.axvline(x=coef_true, color='green', linestyle='--', label='True coef')
            ax6.set_xlabel('Parameter Value', fontsize=10)
            ax6.set_ylabel('Density', fontsize=10)
            ax6.set_title('Parameter Distribution', fontsize=12)
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def create_optimization_summary(self, results: Dict, save_path: Optional[str] = None, 
                                   E_true: Optional[float] = None, nu_true: Optional[float] = None, 
                                   coef_true: Optional[float] = None):
        """Create comprehensive optimization summary for 3 parameters"""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Matplotlib not available, skipping summary creation")
            return
        
        history = results['optimization_history']
        scores = [h['score'] for h in history]
        E_values = [h['params'][0] for h in history]
        nu_values = [h['params'][1] for h in history]
        coef_values = [h['params'][2] for h in history]
        
        # Create summary figure with 2x3 layout
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Bayesian Optimization Summary (3 Parameters)', fontsize=16, fontweight='bold')
        
        # 1. Final optimization progress
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(range(len(scores)), scores, 'o-', linewidth=2, markersize=4)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Objective Function Value', fontsize=12)
        ax1.set_title('Complete Optimization Progress', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 3D parameter space with best solution
        ax2 = plt.subplot(2, 3, 2, projection='3d')
        scatter = ax2.scatter(E_values, nu_values, coef_values, c=scores, cmap='viridis', s=40, alpha=0.7)
        ax2.scatter(results['best_params']['E'], results['best_params']['nu'], results['best_params']['coef'],
                   c='red', s=200, marker='*', label='Best Solution', edgecolors='black', linewidths=1.5)
        
        # Add true values if provided
        if E_true is not None and nu_true is not None and coef_true is not None:
            ax2.scatter(E_true, nu_true, coef_true, c='blue', s=200, marker='o', 
                       label='True Values', edgecolors='black', linewidths=1.5)
        
        ax2.set_xlabel('Young\'s Modulus E', fontsize=12)
        ax2.set_ylabel('Poisson\'s Ratio ν', fontsize=12)
        ax2.set_zlabel('Nonlinear Coefficient', fontsize=12)
        ax2.set_title('3D Parameter Space', fontsize=14, fontweight='bold')
        ax2.legend()
        
        # 3. Statistics text
        ax3 = plt.subplot(2, 3, 3)
        ax3.text(0.1, 0.9, f'Optimization Statistics:', fontsize=14, fontweight='bold', 
                transform=ax3.transAxes)
        ax3.text(0.1, 0.8, f'Total Evaluations: {len(scores)}', fontsize=12, 
                transform=ax3.transAxes)
        ax3.text(0.1, 0.7, f'Best Score: {results["best_score"]:.6f}', fontsize=12, 
                transform=ax3.transAxes)
        ax3.text(0.1, 0.6, f'Best E: {results["best_params"]["E"]:.4f}', fontsize=12, 
                transform=ax3.transAxes)
        ax3.text(0.1, 0.5, f'Best ν: {results["best_params"]["nu"]:.4f}', fontsize=12, 
                transform=ax3.transAxes)
        ax3.text(0.1, 0.4, f'Best coef: {results["best_params"]["coef"]:.3f}', fontsize=12, 
                transform=ax3.transAxes)
        
        # Add true value comparison if available
        y_pos = 0.3
        if E_true is not None and nu_true is not None and coef_true is not None:
            ax3.text(0.1, y_pos, f'True E: {E_true:.4f}', fontsize=12, color='blue',
                    transform=ax3.transAxes)
            y_pos -= 0.05
            ax3.text(0.1, y_pos, f'True ν: {nu_true:.4f}', fontsize=12, color='blue',
                    transform=ax3.transAxes)
            y_pos -= 0.05
            ax3.text(0.1, y_pos, f'True coef: {coef_true:.3f}', fontsize=12, color='blue',
                    transform=ax3.transAxes)
            y_pos -= 0.05
            
            E_error = abs(results["best_params"]["E"] - E_true)
            nu_error = abs(results["best_params"]["nu"] - nu_true)
            coef_error = abs(results["best_params"]["coef"] - coef_true)
            ax3.text(0.1, y_pos, f'E Error: {E_error:.4f} ({100*E_error/E_true:.1f}%)', 
                    fontsize=12, color='darkgreen', transform=ax3.transAxes)
            y_pos -= 0.05
            ax3.text(0.1, y_pos, f'ν Error: {nu_error:.4f} ({100*nu_error/nu_true:.1f}%)', 
                    fontsize=12, color='darkgreen', transform=ax3.transAxes)
            y_pos -= 0.05
            ax3.text(0.1, y_pos, f'coef Error: {coef_error:.3f} ({100*coef_error/coef_true:.1f}%)', 
                    fontsize=12, color='darkgreen', transform=ax3.transAxes)
            y_pos -= 0.05
        
        ax3.text(0.1, y_pos, f'Score Std: {np.std(scores):.6f}', fontsize=12, 
                transform=ax3.transAxes)
        y_pos -= 0.05
        ax3.text(0.1, y_pos, f'Improvement: {scores[0] - results["best_score"]:.6f}', fontsize=12, 
                transform=ax3.transAxes)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # 4. Convergence analysis
        ax4 = plt.subplot(2, 3, 4)
        convergence_scores = [min(scores[:i+1]) for i in range(len(scores))]
        ax4.plot(range(len(convergence_scores)), convergence_scores, 'o-', color='green', linewidth=2, markersize=4)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Best Score Achieved', fontsize=12)
        ax4.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Parameter evolution over time
        ax5 = plt.subplot(2, 3, 5)
        iterations = range(len(E_values))
        ax5.plot(iterations, E_values, 'o-', label='E', linewidth=2, markersize=3)
        ax5.plot(iterations, nu_values, 's-', label='ν', linewidth=2, markersize=3)
        ax5.plot(iterations, coef_values, 'd-', label='coef', linewidth=2, markersize=3)
        
        # Add true values as horizontal lines
        if E_true is not None:
            ax5.axhline(y=E_true, color='red', linestyle='--', alpha=0.7, label='True E')
        if nu_true is not None:
            ax5.axhline(y=nu_true, color='blue', linestyle='--', alpha=0.7, label='True ν')
        if coef_true is not None:
            ax5.axhline(y=coef_true, color='green', linestyle='--', alpha=0.7, label='True coef')
        
        ax5.set_xlabel('Iteration', fontsize=12)
        ax5.set_ylabel('Parameter Value', fontsize=12)
        ax5.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # 6. Parameter correlation heatmap
        ax6 = plt.subplot(2, 3, 6)
        import pandas as pd
        param_df = pd.DataFrame({
            'E': E_values,
            'ν': nu_values,
            'coef': coef_values,
            'score': scores
        })
        correlation_matrix = param_df.corr()
        
        im = ax6.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values as text
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax6.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax6.set_xticks(range(len(correlation_matrix.columns)))
        ax6.set_yticks(range(len(correlation_matrix.columns)))
        ax6.set_xticklabels(correlation_matrix.columns, fontsize=12)
        ax6.set_yticklabels(correlation_matrix.columns, fontsize=12)
        ax6.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Optimization summary saved to: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    print("🎯 贝叶斯优化材料参数标定")
    print("=" * 60)
    
    # 检查命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='Bayesian Optimization Material Parameter Calibration')
    parser.add_argument('--real-time', action='store_true', help='Enable real-time visualization')
    parser.add_argument('--no-visualization', action='store_true', help='Disable all visualization')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    parser.add_argument('--E-true', type=float, default=None, help='True Young\'s Modulus value')
    parser.add_argument('--nu-true', type=float, default=None, help='True Poisson\'s Ratio value')
    parser.add_argument('--coef-true', type=float, default=None, help='True nonlinear coefficient value')
    parser.add_argument('--n-initial', type=int, default=10, help='Number of initial samples')
    parser.add_argument('--n-iterations', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--E-min', type=float, default=0.1000, help='Minimum E value')
    parser.add_argument('--E-max', type=float, default=0.3000, help='Maximum E value')
    parser.add_argument('--nu-min', type=float, default=0.4500, help='Minimum nu value')
    parser.add_argument('--nu-max', type=float, default=0.4900, help='Maximum nu value')
    parser.add_argument('--coef-min', type=float, default=0.0, help='Minimum nonlinear coefficient value')
    parser.add_argument('--coef-max', type=float, default=0.500, help='Maximum nonlinear coefficient value')
    parser.add_argument('--acquisition', type=str, default='adaptive',
                       choices=['ei', 'ucb', 'pi', 'ts', 'adaptive'],
                       help='Acquisition function for Bayesian optimization')
    parser.add_argument('--xi', type=float, default=0.01,
                       help='Exploration parameter for acquisition function')
    
    args = parser.parse_args()
    
    # 创建真实数据接口
    real_data_interface = RealDataInterface()
    
    # 创建贝叶斯优化器
    calibrator = BayesianCalibration(
        real_data_interface=real_data_interface,
        E_bounds=(args.E_min, args.E_max),
        nu_bounds=(args.nu_min, args.nu_max),
        coef_bounds=(args.coef_min, args.coef_max),
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        acquisition=args.acquisition,  # 使用命令行指定的采集函数
        xi=args.xi  # 使用命令行指定的探索参数
    )
    
    # 检查是否有真实数据
    print("\n📋 检查真实数据...")
    
    # 尝试加载真实数据
    real_data = None
    data_sources = [
        "real_data.json",
        "real_data.pkl", 
        "data/real_data.json",
        "data/real_data.pkl"
    ]
    
    for source in data_sources:
        if Path(source).exists():
            try:
                if source.endswith('.json'):
                    real_data = real_data_interface.load_from_json(source)
                else:
                    real_data = real_data_interface.load_from_pickle(source)
                break
            except Exception as e:
                print(f"⚠️ 加载 {source} 失败: {e}")
    
    if real_data is None:
        print("ℹ️ 未找到真实数据，使用仿真数据进行演示")
        print("   要使用真实数据，请将数据保存为 real_data.json 或 real_data.pkl")
        
        # 使用命令行参数或默认真实参数
        E_true = args.E_true if args.E_true is not None else 0.1983
        nu_true = args.nu_true if args.nu_true is not None else 0.4795
        coef_true = args.coef_true if args.coef_true is not None else 0.200
        
        print(f"   使用仿真真实参数: E={E_true}, nu={nu_true}")
    else:
        E_true = args.E_true
        nu_true = args.nu_true
        coef_true = args.coef_true
        print("✓ 找到真实数据")
    
    # 显示配置信息
    print(f"\n📋 优化配置:")
    print(f"   E范围: [{args.E_min}, {args.E_max}]")
    print(f"   nu范围: [{args.nu_min}, {args.nu_max}]")
    print(f"   coef范围: [{args.coef_min}, {args.coef_max}]")
    print(f"   初始样本: {args.n_initial}")
    print(f"   迭代次数: {args.n_iterations}")
    print(f"   采集函数: {args.acquisition}")
    print(f"   探索参数 xi: {args.xi}")
    print(f"   实时可视化: {'是' if args.real_time else '否'}")
    print(f"   保存图表: {'是' if args.save_plots else '否'}")
    
    # 运行标定
    try:
        if args.real_time:
            print("\n🎯 启动实时可视化优化...")
            results = calibrator.plot_real_time_optimization(
                real_data=real_data,
                E_true=E_true,
                nu_true=nu_true,
                coef_true=coef_true,
                update_interval=3
            )
        else:
            results = calibrator.run_calibration(
                real_data=real_data,
                E_true=E_true,
                nu_true=nu_true,
                coef_true=coef_true
            )
        
        if results is not None:
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"calibration_results_{timestamp}.json"
            calibrator.save_results(results, result_file, E_true, nu_true, coef_true)
            
            # 显示最终结果
            print(f"\n📊 标定结果总结:")
            print(f"   优化后的材料参数:")
            print(f"     杨氏模量 E = {results['best_params']['E']:.4f}")
            print(f"     泊松比 nu = {results['best_params']['nu']:.4f}")
            print(f"     非线性系数 coef = {results['best_params']['coef']:.4f}")
            print(f"   最终误差 = {results['best_score']:.6f}")
            print(f"   评估次数 = {results['n_evaluations']}")
            
            if E_true is not None and nu_true is not None:
                print(f"\n📈 参数对比:")
                print(f"   真实参数: E={E_true:.4f}, nu={nu_true:.4f}, coef={coef_true:.4f}")
                print(f"   优化参数: E={results['best_params']['E']:.4f}, nu={results['best_params']['nu']:.4f}, coef={results['best_params']['coef']:.4f}")
                
                E_error = abs(results['best_params']['E'] - E_true)
                nu_error = abs(results['best_params']['nu'] - nu_true)
                coef_error = abs(results['best_params']['coef'] - coef_true)
                print(f"   参数误差: E={E_error:.4f}, nu={nu_error:.4f}, coef={coef_error:.4f}")
            
            print(f"\n💡 使用建议:")
            print(f"   1. 结果已保存至 {result_file}")
            print(f"   2. 要提高精度，可以增加 n_iterations 参数")
            print(f"   3. 要使用真实数据，请准备 real_data.json 或 real_data.pkl 文件")
            print(f"   4. 使用 --real-time 参数启用实时可视化")
            print(f"   5. 使用 --save-plots 参数保存图表到文件")
            
            # 可视化处理
            if not args.no_visualization and not args.real_time:
                print(f"\n📊 生成可视化图表...")
                
                if args.save_plots:
                    # 保存图表到文件
                    plot_dir = Path(f"calibration_plots_{timestamp}")
                    plot_dir.mkdir(exist_ok=True)
                    
                    calibrator.plot_optimization_history(
                        results, 
                        save_path=str(plot_dir / "optimization_history.png"),
                        E_true=E_true,
                        nu_true=nu_true,
                        coef_true=coef_true
                    )
                    calibrator.create_optimization_summary(
                        results, 
                        save_path=str(plot_dir / "optimization_summary.png"),
                        E_true=E_true,
                        nu_true=nu_true,
                        coef_true=coef_true
                    )
                    print(f"✓ 所有图表已保存至: {plot_dir}")
                else:
                    # 显示图表
                    calibrator.plot_optimization_history(results, E_true=E_true, nu_true=nu_true, coef_true=coef_true)
                    calibrator.create_optimization_summary(results, E_true=E_true, nu_true=nu_true, coef_true=coef_true)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断优化过程")
    except Exception as e:
        print(f"\n❌ 标定过程失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()