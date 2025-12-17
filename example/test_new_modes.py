#!/usr/bin/env python3
"""
测试三种触觉传感器数据采集模式的参数设置和文件夹结构
"""

import numpy as np
from pathlib import Path

# 模拟参数设置
PROJ_DIR = Path("../xengym")
DATA_PATH = PROJ_DIR / "data" / "obj"
item_name = "circle_r4"

# 传感器参数
y_range = (-0.007, 0.007)  # 传感器Y方向范围，约14mm
z_range = (-0.014, 0.0014)  # 传感器Z方向范围，约15.4mm

# 新增参数
slide_range_y = (-0.003, 0.003)  # 平移范围，保守一些避免脱离
slide_range_z = (-0.005, 0.005)  # 平移范围
max_rotation_angle = 30  # 最大旋转角度（度）
contact_force_threshold = 0.1  # 接触力阈值

def test_folder_structure():
    """测试文件夹结构"""
    print("=== 测试文件夹结构 ===")
    
    # 三种模式的文件夹
    modes = ['press_only', 'press_slide', 'press_rotate']
    sensors = ['sensor_0', 'sensor_1']
    
    for sensor in sensors:
        for mode in modes:
            folder_path = DATA_PATH / item_name / sensor / mode
            print(f"模式文件夹: {folder_path}")
            
            # 轨迹文件夹示例
            for trj_num in range(3):
                trj_folder = folder_path / f'trj_{trj_num}'
                print(f"  轨迹文件夹: {trj_folder}")

def test_parameters():
    """测试参数合理性"""
    print("\n=== 测试参数合理性 ===")
    
    print(f"传感器尺寸范围:")
    print(f"  Y方向: {y_range[0]*1000:.1f}mm 到 {y_range[1]*1000:.1f}mm (总宽度: {(y_range[1]-y_range[0])*1000:.1f}mm)")
    print(f"  Z方向: {z_range[0]*1000:.1f}mm 到 {z_range[1]*1000:.1f}mm (总高度: {(z_range[1]-z_range[0])*1000:.1f}mm)")
    
    print(f"\n平移参数:")
    print(f"  Y方向平移范围: ±{slide_range_y[1]*1000:.1f}mm")
    print(f"  Z方向平移范围: ±{slide_range_z[1]*1000:.1f}mm")
    print(f"  平移范围占传感器尺寸比例: Y={slide_range_y[1]/(y_range[1]-y_range[0])*200:.1f}%, Z={slide_range_z[1]/(z_range[1]-z_range[0])*200:.1f}%")
    
    print(f"\n旋转参数:")
    print(f"  最大旋转角度: ±{max_rotation_angle}°")
    print(f"  接触力检测阈值: {contact_force_threshold}")

def test_motion_sequence():
    """测试运动序列"""
    print("\n=== 测试运动序列 ===")
    
    print("1. 纯按压模式:")
    print("   - 随机位置接触")
    print("   - 执行多级深度按压")
    print("   - 保存每个深度的数据")
    
    print("\n2. 按压+平移模式:")
    print("   - 随机位置接触")
    print("   - 按压到固定深度(2mm)")
    print("   - 生成随机平移向量")
    slide_y = np.random.uniform(*slide_range_y)
    slide_z = np.random.uniform(*slide_range_z)
    print(f"   - 示例平移: Y={slide_y*1000:.2f}mm, Z={slide_z*1000:.2f}mm")
    print("   - 分10步执行平移")
    print("   - 每步检测接触状态")
    
    print("\n3. 按压+旋转模式:")
    print("   - 随机位置接触（作为旋转中心）")
    print("   - 按压到固定深度(2mm)")
    max_rotation = np.random.uniform(10, max_rotation_angle)
    rotation_direction = np.random.choice([-1, 1])
    total_rotation = max_rotation * rotation_direction
    print(f"   - 示例旋转: {total_rotation:.1f}°")
    print("   - 分15步执行旋转")
    print("   - 每步检测接触状态")

def main():
    print("触觉传感器三种数据采集模式测试")
    print("="*50)
    
    test_folder_structure()
    test_parameters()
    test_motion_sequence()
    
    print("\n=== 数据保存格式 ===")
    print("每帧数据包含:")
    print("- real_rectify: 真实传感器矫正图像")
    print("- diff_real: 真实传感器差分图像") 
    print("- depth_real: 真实传感器深度")
    print("- marker_of_real: 真实传感器原始marker") 
    print("- marker_real: 真实传感器插值marker")
    print("- force_real: 真实传感器力数据")
    print("- res_force_real: 真实传感器合力")
    print("- rectify_sim: 仿真矫正图像")
    print("- rectify_pure_sim: 仿真纯净图像")
    print("- diff_sim: 仿真差分图像")
    print("- marker_sim: 仿真marker坐标")
    print("- raw_depth_sim: 仿真原始深度")
    print("- vis_depth_sim: 仿真可视化深度")
    print("- force_sim: 仿真力数据")
    print("- vertex: FEM顶点坐标")

if __name__ == "__main__":
    main() 