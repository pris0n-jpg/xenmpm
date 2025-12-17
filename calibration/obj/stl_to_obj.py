#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STL转OBJ转换工具，并将坐标原点移至物体质心
"""

import os
import sys
import numpy as np
import trimesh
import json
import shutil
from pathlib import Path


def convert_stl_to_obj(stl_file, output_dir=None, center_to_centroid=True, simplify_ratio=None):
    """
    将STL文件转换为OBJ文件，并将坐标原点移至物体质心
    
    Args:
        stl_file: STL文件路径
        output_dir: 输出目录，如果为None则与STL文件同目录
        center_to_centroid: 是否将原点移至质心
        simplify_ratio: 网格简化比例 (0.0-1.0)，None表示不简化
        
    Returns:
        输出的OBJ文件路径
    """
    stl_path = Path(stl_file)
    
    # 如果输出目录未指定，则使用STL文件所在目录
    if output_dir is None:
        output_dir = stl_path.parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # 加载STL文件
    print(f"正在加载STL文件: {stl_path}")
    try:
        mesh = trimesh.load(str(stl_path))
    except Exception as e:
        print(f"加载STL文件失败: {e}")
        return None
    
    # 打印原始网格信息
    print(f"原始网格: {len(mesh.faces)} 个面, {len(mesh.vertices)} 个顶点")
    
    # 如果需要简化网格
    if simplify_ratio is not None:
        try:
            # 确保简化比例在有效范围内
            simplify_ratio = max(0.01, min(1.0, simplify_ratio))
            target_faces = int(len(mesh.faces) * simplify_ratio)
            
            # 确保最少保留10个面
            target_faces = max(10, target_faces)
            
            print(f"正在简化网格到 {target_faces} 个面 (原始的 {simplify_ratio:.1%})...")
            
            # 执行网格简化
            mesh = mesh.simplify_quadratic_decimation(target_faces)
            
            print(f"简化后网格: {len(mesh.faces)} 个面, {len(mesh.vertices)} 个顶点")
        except Exception as e:
            print(f"网格简化失败: {e}")
            print("将使用原始网格继续处理")
    
    # 计算质心
    centroid = mesh.centroid
    print(f"物体质心: {centroid}")
    
    # 如果需要将原点移至质心，则进行平移
    if center_to_centroid:
        # 创建平移矩阵
        translation = np.eye(4)
        translation[:3, 3] = -centroid
        
        # 应用平移变换
        mesh.apply_transform(translation)
        print("已将原点移至物体质心")
    
    # 设置输出文件路径
    filename_parts = []
    filename_parts.append(stl_path.stem)
    # if center_to_centroid:
    #     filename_parts.append("centered")
    if simplify_ratio is not None:
        filename_parts.append(f"simplified_{int(simplify_ratio*100)}")
    
    obj_filename = "_".join(filename_parts) + ".obj"
    obj_path = output_dir / obj_filename
    
    # 导出为OBJ格式
    mesh.export(str(obj_path), file_type='obj')
    print(f"成功导出OBJ文件: {obj_path}")
    
    return obj_path


def process_directory(directory, output_dir=None, center_to_centroid=True, 
                     copy_to_assets=False, simplify_ratio=None):
    """
    处理目录中的所有STL文件
    
    Args:
        directory: 输入目录
        output_dir: 输出目录
        center_to_centroid: 是否将原点移至质心
        copy_to_assets: 是否复制到xengym资源目录
        simplify_ratio: 网格简化比例
    """
    directory = Path(directory)
    
    # 如果输出目录未指定，则使用输入目录
    if output_dir is None:
        output_dir = directory
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有STL文件
    stl_files = list(directory.glob("*.STL")) + list(directory.glob("*.stl"))
    
    if not stl_files:
        print(f"目录 {directory} 中没有找到STL文件")
        return
    
    print(f"在目录 {directory} 中找到 {len(stl_files)} 个STL文件")
    
    # 处理所有STL文件
    obj_files = []
    for stl_file in stl_files:
        print(f"\n处理文件: {stl_file.name}")
        obj_path = convert_stl_to_obj(stl_file, output_dir, center_to_centroid, simplify_ratio)
        if obj_path is not None:
            obj_files.append(obj_path)
    
    # 如果需要复制到xengym资源目录
    if copy_to_assets and obj_files:
        copy_to_xengym_assets(obj_files)


def copy_to_xengym_assets(obj_files):
    """
    将OBJ文件复制到xengym资源目录
    
    Args:
        obj_files: OBJ文件路径列表
    """
    # 确定xengym资源目录路径
    # 从当前脚本路径推导
    script_path = Path(__file__)
    # 假设当前脚本位于 calibration/obj/ 目录下
    xengym_assets_dir = "/home/czl/Downloads/workspace/xengym/calibration/obj"
    
    if not xengym_assets_dir.exists():
        print(f"警告: xengym资源目录不存在: {xengym_assets_dir}")
        # 尝试其他可能的路径
        possible_paths = [
            # Path("/home/czl/Downloads/workspace/xengym/xengym/assets/obj"),
            # Path("./xengym/assets/obj"),
            # Path("../xengym/assets/obj"),
            # Path("../../xengym/assets/obj"),
            Path("/home/czl/Downloads/workspace/xengym/calibration/obj")
        ]
        
        for path in possible_paths:
            if path.exists():
                xengym_assets_dir = path
                print(f"找到xengym资源目录: {xengym_assets_dir}")
                break
        else:
            print("无法找到xengym资源目录，将不会复制文件")
            return
    
    # 创建资源目录(如果不存在)
    os.makedirs(xengym_assets_dir, exist_ok=True)
    
    # 复制所有OBJ文件
    for obj_file in obj_files:
        dest_file = xengym_assets_dir / obj_file.name
        shutil.copy2(obj_file, dest_file)
        print(f"已复制 {obj_file.name} 到 {xengym_assets_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="STL转OBJ工具，并将坐标原点移至物体质心")
    parser.add_argument("-i", "--input", help="输入STL文件或目录，默认为当前目录",default="/home/czl/Downloads/workspace/xengym/calibration//obj")
    parser.add_argument("-o", "--output", help="输出目录，默认与输入相同")
    parser.add_argument("--no-center", action="store_true", help="不将原点移至质心")
    parser.add_argument("--copy-to-assets", action="store_true", help="复制转换后的OBJ文件到xengym资源目录")
    parser.add_argument("--simplify", type=float, help="网格简化比例 (0.0-1.0)，例如0.5表示保留50%的面")
    
    args = parser.parse_args()
    
    # 如果没有指定输入，则默认使用当前目录
    if args.input is None:
        input_path = Path(".")
        print(f"未指定输入路径，将使用当前目录: {input_path.absolute()}")
    else:
        input_path = Path(args.input)
    
    center_to_centroid = not args.no_center
    simplify_ratio = args.simplify
    
    # 检查简化比例是否合法
    if simplify_ratio is not None:
        if simplify_ratio <= 0 or simplify_ratio > 1.0:
            print(f"无效的简化比例: {simplify_ratio}，应在0.0-1.0之间")
            simplify_ratio = None
        else:
            print(f"将使用简化比例: {simplify_ratio:.1%}")
    
    if input_path.is_dir():
        # 如果输入是目录，则处理目录中的所有STL文件
        process_directory(input_path, args.output, center_to_centroid, args.copy_to_assets, simplify_ratio)
    elif input_path.is_file():
        # 如果输入是文件，则处理单个STL文件
        obj_path = convert_stl_to_obj(input_path, args.output, center_to_centroid, simplify_ratio)
        
        # 如果需要复制到xengym资源目录
        if args.copy_to_assets and obj_path is not None:
            copy_to_xengym_assets([obj_path])
    else:
        print(f"输入路径 {input_path} 不存在")
        sys.exit(1)
    
    print("\n所有转换任务完成") 