#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real Calibration Data Fitting Tool
从多条真实采集数据中拟合出最终的力曲线和marker位移，用于标定
生成 real_fit_data.pkl，格式与 calibration.py 兼容
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# 添加项目路径
PROJ_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJ_DIR))


class RealDataFitter:
    def __init__(self, data_path: str, output_path: str = None):
        """
        初始化真实数据拟合器
        
        Parameters:
        - data_path: real_calibration_data.pkl 路径
        - output_path: 输出文件路径，默认为 real_fit_data.pkl
        """
        self.data_path = Path(data_path)
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.data_path.parent / "real_fit_data.pkl"
        
        print(f"📂 加载数据: {self.data_path}")
        self.data = self.load_data(self.data_path)
        self.fitted_data = {}
        
    def load_data(self, file_path: Path) -> Dict:
        """加载 pickle 数据文件"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def save_data(self, data: Dict, file_path: Path):
        """保存数据到 pickle 文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 已保存: {file_path}")
    
    def extract_trajectory_runs(self, obj_name: str, base_traj: str) -> List[Tuple[str, Dict]]:
        """
        提取某个物体的某条轨迹的所有 run 数据
        
        Returns:
        - List of (run_name, trajectory_data) tuples
        """
        runs = []
        for traj_key in self.data[obj_name].keys():
            if traj_key.startswith(base_traj + "_run") or traj_key == base_traj:
                runs.append((traj_key, self.data[obj_name][traj_key]))
        return runs
    
    def polynomial_force_model(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        二次多项式力模型（完整形式）
        F(x) = a*x² + b*x + c
        """
        return a * x**2 + b * x + c
    
    def fit_force_curve(self, all_steps: List[np.ndarray], all_forces: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
        """
        从多条 run 的力数据中拟合出最终的力曲线
        
        Parameters:
        - all_steps: 多条 run 的 step 数组列表
        - all_forces: 多条 run 的 force 数组列表
        
        Returns:
        - fitted_steps: 拟合后的 step 序列
        - fitted_forces: 拟合后的 force 值
        - fit_params: 拟合参数字典
        - run_weights: 每条run的权重数组（用于marker加权）
        """
        # 合并所有数据点
        all_x = np.concatenate(all_steps)
        all_y = np.concatenate(all_forces)
        
        # 使用加权：根据数据点密度调整权重（出现频率高的step权重更大）
        unique_steps, counts = np.unique(all_x, return_counts=True)
        weights_map = dict(zip(unique_steps, np.sqrt(counts)))  # sqrt 使权重差异不要太大
        weights = np.array([weights_map[x] for x in all_x])
        
        try:
            # 拟合完整的二次多项式
            popt, pcov = curve_fit(
                self.polynomial_force_model,
                all_x, all_y,
                p0=[0.01, 0.1, 0.0],  # 初始猜测 [a, b, c]
                sigma=1/weights,  # 权重
                absolute_sigma=False
            )
            
            a, b, c = popt
            print(f"    拟合参数: a={a:.6f}, b={b:.6f}, c={c:.6f}")
            print(f"    拟合公式: F(x) = {a:.6f}*x² + {b:.6f}*x + {c:.6f}")
            print(f"    Y轴截距: F(0) = {c:.6f}")
            
            # 生成拟合后的序列（使用所有出现过的 step）
            fitted_steps = np.sort(unique_steps)
            fitted_forces = self.polynomial_force_model(fitted_steps, a, b, c)
            
            print(f"    力值范围: [{fitted_forces.min():.4f}, {fitted_forces.max():.4f}] N")
            
            # 计算拟合质量指标
            y_pred = self.polynomial_force_model(all_x, a, b, c)
            residuals = all_y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((all_y - np.mean(all_y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"    R² = {r_squared:.4f}")
            
            # 计算每条run的拟合误差，用于marker加权
            run_weights = []
            for steps, forces in zip(all_steps, all_forces):
                # 计算该run的预测值
                y_pred_run = self.polynomial_force_model(steps, a, b, c)
                # 计算该run的均方根误差 (RMSE)
                rmse = np.sqrt(np.mean((forces - y_pred_run)**2))
                # 权重与误差成反比（误差小权重大）
                # 使用倒数并加小量避免除零
                weight = 1.0 / (rmse + 1e-6)
                run_weights.append(weight)
            
            run_weights = np.array(run_weights)
            # 归一化权重
            run_weights = run_weights / run_weights.sum()
            
            print(f"    Run权重分布: {[f'{w:.3f}' for w in run_weights]}")
            
            fit_params = {
                'model': 'polynomial_full',
                'coefficients': {'a': float(a), 'b': float(b), 'c': float(c)},
                'r_squared': float(r_squared),
                'n_samples': len(all_x),
                'n_runs': len(all_steps),
                'run_weights': run_weights.tolist()
            }
            
            return fitted_steps, fitted_forces, fit_params, run_weights
            
        except Exception as e:
            print(f"    ⚠️  拟合失败，使用加权平均: {e}")
            return self.weighted_average_force(all_steps, all_forces)
    
    def weighted_average_force(self, all_steps: List[np.ndarray], all_forces: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        使用加权平均作为备选方案
        """
        # 找出所有唯一的 step
        unique_steps = np.unique(np.concatenate(all_steps))
        fitted_forces = []
        
        for step in unique_steps:
            forces_at_step = []
            for steps, forces in zip(all_steps, all_forces):
                idx = np.where(steps == step)[0]
                if len(idx) > 0:
                    forces_at_step.append(forces[idx[0]])
            
            # 加权平均（权重相同）
            fitted_forces.append(np.mean(forces_at_step))
        
        fitted_forces = np.array(fitted_forces)
        
        fit_params = {
            'model': 'weighted_average',
            'n_samples': sum(len(s) for s in all_steps),
            'n_runs': len(all_steps)
        }
        
        # 对于备选方案，使用等权重
        run_weights = np.ones(len(all_steps)) / len(all_steps)
        
        return unique_steps, fitted_forces, fit_params, run_weights
    
    def fit_marker_displacement(self, all_steps: List[np.ndarray], all_markers: List[np.ndarray],
                                run_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据力拟合质量对 marker 位移做加权平均
        
        Parameters:
        - all_steps: 多条 run 的 step 数组列表
        - all_markers: 多条 run 的 marker 数组列表 (每个元素shape: [n_steps, 20, 11, 2])
        - run_weights: 每条run的权重（基于力拟合质量）
        
        Returns:
        - fitted_steps: 拟合后的 step 序列
        - fitted_markers: 拟合后的 marker 位移 (n_steps, 20, 11, 2)
        """
        # 找出所有唯一的 step
        unique_steps = np.unique(np.concatenate(all_steps))
        
        # 获取 marker 形状
        marker_shape = all_markers[0].shape[1:]  # (20, 11, 2)
        fitted_markers = []
        
        for step in unique_steps:
            markers_at_step = []
            weights_at_step = []
            
            for i, (steps, markers) in enumerate(zip(all_steps, all_markers)):
                idx = np.where(steps == step)[0]
                if len(idx) > 0:
                    markers_at_step.append(markers[idx[0]])
                    # 使用基于力拟合质量的权重
                    weights_at_step.append(run_weights[i])
            
            if len(markers_at_step) > 0:
                # 加权平均
                weights_at_step = np.array(weights_at_step)
                weights_at_step /= weights_at_step.sum()
                
                weighted_marker = np.zeros(marker_shape)
                for marker, weight in zip(markers_at_step, weights_at_step):
                    weighted_marker += marker * weight
                
                fitted_markers.append(weighted_marker)
        
        fitted_markers = np.array(fitted_markers)
        
        print(f"    Marker 加权平均完成: shape={fitted_markers.shape}")
        print(f"    使用权重: {[f'{w:.3f}' for w in run_weights]}")
        
        return unique_steps, fitted_markers
    
    def fit_all_trajectories(self):
        """
        对所有物体的所有轨迹进行拟合
        """
        print("\n" + "="*60)
        print("开始拟合真实标定数据")
        print("="*60)
        
        for obj_name in self.data.keys():
            print(f"\n📦 处理物体: {obj_name}")
            
            # 提取所有基础轨迹名称
            base_trajs = set()
            for traj_key in self.data[obj_name].keys():
                if "_run" in traj_key:
                    base_name = traj_key.split("_run")[0]
                else:
                    base_name = traj_key
                base_trajs.add(base_name)
            
            self.fitted_data[obj_name] = {}
            
            for base_traj in sorted(base_trajs):
                print(f"\n  📊 拟合轨迹: {base_traj}")
                
                # 获取所有 run
                runs = self.extract_trajectory_runs(obj_name, base_traj)
                print(f"    找到 {len(runs)} 条 run")
                
                if len(runs) == 0:
                    continue
                
                # 提取所有 run 的数据
                all_steps = []
                all_forces = []
                all_markers = []
                
                for run_name, traj_data in runs:
                    steps = []
                    forces = []
                    markers = []
                    
                    for step_key in sorted(traj_data.keys()):
                        step_data = traj_data[step_key]
                        if 'force_xyz' in step_data and 'marker_displacement' in step_data:
                            step_num = int(step_key.split('_')[-1])
                            steps.append(step_num)
                            forces.append(float(step_data['force_xyz'][2]))  # Z方向力
                            markers.append(step_data['marker_displacement'])
                    
                    if len(steps) > 0:
                        all_steps.append(np.array(steps))
                        all_forces.append(np.array(forces))
                        all_markers.append(np.array(markers))
                
                if len(all_steps) == 0:
                    print("    ⚠️  没有有效数据，跳过")
                    continue
                
                # 拟合力曲线（同时计算每条run的权重）
                print(f"    拟合力曲线...")
                fitted_steps, fitted_forces, fit_params, run_weights = self.fit_force_curve(all_steps, all_forces)
                
                # 使用力拟合权重对 marker 位移做加权平均
                print(f"    加权平均 marker 位移...")
                _, fitted_markers = self.fit_marker_displacement(all_steps, all_markers, run_weights)
                
                # 构建拟合后的轨迹数据（与 calibration.py 格式兼容）
                fitted_traj_data = {}
                for i, step_num in enumerate(fitted_steps):
                    step_key = f"step_{step_num:03d}"
                    fitted_traj_data[step_key] = {
                        'marker_displacement': fitted_markers[i].astype(np.float32),
                        'force_xyz': np.array([0.0, 0.0, fitted_forces[i]], dtype=np.float32),  # 只有Z方向
                        'metadata': {
                            'trajectory': base_traj,
                            'step_index': int(step_num),
                            'fitted': True,
                            'fit_params': fit_params
                        },
                        'depth_field': None
                    }
                
                # 使用基础轨迹名称（不带 _runX）
                self.fitted_data[obj_name][base_traj] = fitted_traj_data
                
                print(f"    ✓ 完成: {len(fitted_steps)} 个 step")
        
        print("\n" + "="*60)
        print("拟合完成")
        print("="*60)
    
    def save_fitted_data(self):
        """保存拟合后的数据"""
        self.save_data(self.fitted_data, self.output_path)
        
        # 打印摘要
        print(f"\n📋 拟合数据摘要:")
        for obj_name, trajs in self.fitted_data.items():
            print(f"  {obj_name}: {len(trajs)} 条轨迹")
            for traj_name, traj_data in trajs.items():
                print(f"    - {traj_name}: {len(traj_data)} steps")
    
    def run(self):
        """运行完整的拟合流程"""
        self.fit_all_trajectories()
        self.save_fitted_data()
        print(f"\n✅ 拟合数据已保存到: {self.output_path}")


def main():
    """主函数"""
    print("🎯 Real Calibration Data Fitting Tool")
    print("="*60)
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    
    # 输入输出路径
    input_path = script_dir / "real_calibration_data.pkl"
    output_path = script_dir / "real_fit_data.pkl"
    
    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"❌ 错误: {input_path} 不存在")
        print(f"   请确保文件存在于: {script_dir}")
        sys.exit(1)
    
    print(f"✓ 输入文件: {input_path}")
    print(f"✓ 输出文件: {output_path}")
    
    try:
        fitter = RealDataFitter(str(input_path), str(output_path))
        fitter.run()
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()