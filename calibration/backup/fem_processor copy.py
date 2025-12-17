# -*- coding: utf-8 -*-
"""
FEM处理器 - 支持动态材料参数和智能缓存
分离存储几何信息和材料参数，优化批量计算性能
"""

import numpy as np
import scipy.sparse as sp
import os
from tqdm import tqdm
from pathlib import Path
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# 定义face_normal函数替代ezgl.items.MeshData中的函数
def face_normal(v1, v2, v3):
    """计算三角形面片的法向量"""
    a = v2 - v1
    b = v3 - v1
    normal = np.cross(a, b)
    length = np.linalg.norm(normal)
    if length > 0:
        return normal / length
    return normal

# Set up base directory
BASE_DIR = Path(__file__).resolve().parent
program_path = os.path.abspath(__file__)
program_dir = os.path.dirname(program_path)
os.chdir(program_dir)

from Function import *

class FEMProcessor:
    """
    FEM处理器 - 支持动态材料参数
    """
    def __init__(self, Node, Ele, Top, Bot, E=0.1983, nu=0.4795, cache_dir=None, use_cache=True):
        self.node = Node    # 结点坐标
        self.Ele = Ele      # 单元结点编号
        self.E = E          # 杨氏模量
        self.nu = nu        # 泊松比
        self.NN = len(self.node) # 结点总数
        self.NE = len(self.Ele)  # 单元总数
        
        self.Top = Top  # 顶层结点
        self.Bot = Bot  # 底层结点
        self.use_cache = use_cache  # 是否使用缓存
        
        # 设置缓存目录
        self.cache_dir = Path(cache_dir) if cache_dir else BASE_DIR / "cache"
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # 检查几何缓存
        self.geometry_loaded_from_cache = False
        if self.use_cache:
            self.geometry_loaded_from_cache = self._check_and_load_geometry_cache()
        
        # 如果没有缓存，计算几何信息
        if not self.geometry_loaded_from_cache:
            print("🔄 计算几何信息...")
            self._compute_geometry()
            # 保存到缓存
            if self.use_cache:
                self._save_geometry_cache()
        
        self.save_geometry_data()
        # 计算刚度矩阵
        self.update_material_properties(E, nu)
    
    def _check_and_load_geometry_cache(self):
        """检查并加载几何缓存"""
        geometry_hash = self._compute_geometry_hash()
        geometry_cache_file = self.cache_dir / f"geometry_{geometry_hash}.pkl"
        
        print(f"🔍 检查几何缓存: {geometry_cache_file.name}")
        
        if not geometry_cache_file.exists():
            print("❌ 未找到匹配的几何缓存")
            return False
        
        try:
            with open(geometry_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.element_geometry = cache_data['element_geometry']
                self.assembly_matrices = cache_data['assembly_matrices']
                self.fixed_elements = cache_data['fixed_elements']
            
            print(f"✅ 几何信息从缓存加载: {len(self.element_geometry)} 个单元")
            return True
            
        except Exception as e:
            print(f"⚠ 缓存文件损坏，将重新计算: {e}")
            geometry_cache_file.unlink(missing_ok=True)
            return False
    
    def _save_geometry_cache(self):
        """保存几何信息到缓存"""
        geometry_hash = self._compute_geometry_hash()
        geometry_cache_file = self.cache_dir / f"geometry_{geometry_hash}.pkl"
        
        cache_data = {
            'element_geometry': self.element_geometry,
            'assembly_matrices': self.assembly_matrices,
            'fixed_elements': self.fixed_elements,
            'metadata': {'cache_version': '1.0'}
        }
        
        try:
            with open(geometry_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ 几何信息已保存到缓存: {geometry_cache_file.name}")
        except Exception as e:
            print(f"⚠ 缓存保存失败: {e}")
    
    def _has_exact_geometry_cache(self):
        """检查是否有完全匹配的几何缓存"""
        geometry_hash = self._compute_geometry_hash()
        geometry_cache_file = self.cache_dir / f"geometry_{geometry_hash}.pkl"
        return geometry_cache_file.exists()
    
    def _ensure_geometry_ready(self):
        """确保几何信息已准备好"""
        if (not hasattr(self, 'element_geometry') or self.element_geometry is None):
            print("🔄 重新计算几何信息...")
            self._compute_geometry()
    

    
    def _compute_geometry_hash(self):
        """计算几何数据的哈希值用于缓存"""
        geometry_data = np.concatenate([
            self.node.flatten(),
            self.Ele.flatten(),
            np.array(self.Top),
            np.array(self.Bot)
        ])
        return hashlib.md5(geometry_data.tobytes()).hexdigest()[:16]
    
    def _compute_geometry(self):
        """计算几何相关信息"""
        self.fixed_elements = self._check_and_fix_elements()
        self.element_geometry = []
        self.assembly_matrices = []
        
        for i in tqdm(range(self.NE), desc="计算几何"):
            if i in self.fixed_elements['problem_elements']:
                continue
                
            # 获取单元节点坐标
            coords = np.zeros((8, 3))
            for j in range(8):
                n = int(self.Ele[i][j]-1)
                coords[j] = self.node[n]
            
            # 预计算几何信息和装配矩阵
            element_geom = self._precompute_element_geometry(coords)
            self.element_geometry.append(element_geom)
            self.assembly_matrices.append(get_A(self.NN, self.Ele[i]))
    
    def _precompute_element_geometry(self, coords):
        """预计算单元的几何信息（仅刚度矩阵相关）"""
        xo = coords[:, 0]
        yo = coords[:, 1]
        zo = coords[:, 2]
        
        # 高斯积分点
        n = 3
        P = np.array([-0.7745966692414834, 0, 0.7745966692414834])
        H = np.array([5/9, 8/9, 5/9])
        
        # 预计算所有积分点的几何信息
        integration_points = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    det_J, Nx, Ny, Nz = jaco(P[i], P[j], P[k], xo, yo, zo)
                    weight = H[i] * H[j] * H[k] * det_J
                    
                    # 计算B矩阵（用于刚度矩阵）
                    B = get_B(Nx, Ny, Nz)
                    
                    integration_points.append({
                        'weight': weight,
                        'B': B
                    })
        
        return integration_points
    
    def _check_and_fix_elements(self):
        """检查并修复单元节点顺序"""
        problem_elements = []
        
        # 并行检查所有单元
        def check_element(i):
            coords = np.zeros((8, 3))
            for j in range(8):
                n = int(self.Ele[i][j]-1)
                coords[j] = self.node[n]
            
            # 计算雅可比行列式
            NA = np.zeros(8)
            NB = np.zeros(8)
            NC = np.zeros(8)
            FNA(0, 0, NA)
            FNB(0, 0, NB)
            FNC(0, 0, NC)
            
            J = np.zeros((3, 3))
            for k in range(8):
                J[0, 0] += NA[k] * coords[k, 0]
                J[0, 1] += NA[k] * coords[k, 1]
                J[0, 2] += NA[k] * coords[k, 2]
                J[1, 0] += NB[k] * coords[k, 0]
                J[1, 1] += NB[k] * coords[k, 1]
                J[1, 2] += NB[k] * coords[k, 2]
                J[2, 0] += NC[k] * coords[k, 0]
                J[2, 1] += NC[k] * coords[k, 1]
                J[2, 2] += NC[k] * coords[k, 2]
            
            return i, np.linalg.det(J) <= 0
        
        # 多线程检查
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(executor.map(lambda i: check_element(i), range(self.NE)))
        
        for i, has_problem in results:
            if has_problem:
                problem_elements.append(i)
                # 修复单元
                fixed, self.Ele[i] = self.check_and_fix_element(i)
        
        if problem_elements:
            print(f'修复了 {len(problem_elements)} 个单元')
        
        return {'problem_elements': problem_elements, 'fixed_count': len(problem_elements)}

    def update_material_properties(self, E=None, nu=None):
        """更新材料属性并重新计算刚度矩阵"""
        if E is not None:
            self.E = E
        if nu is not None:
            self.nu = nu
            
        print(f"更新材料参数: E={self.E}, nu={self.nu}")
        
        # 重新计算本构矩阵
        self.D = eld(self.E, self.nu)
        
        # 只计算刚度矩阵
        self.K = self.get_K_optimized()
        self.Kf = self.Fix_matrix(self.K)  # 带底部约束的总刚度矩阵
        self.save_data()
    
    def get_K_optimized(self):
        """优化的刚度矩阵计算"""
        self._ensure_geometry_ready()
        
        # 如果几何信息不可用，使用传统方法
        if not hasattr(self, 'element_geometry') or self.element_geometry is None:
            return self._get_K_traditional()
        
        K = sp.sparse.lil_matrix((3*self.NN, 3*self.NN))
        valid_elements = [i for i in range(self.NE) 
                         if i not in self.fixed_elements['problem_elements']]
        
        for idx, i in enumerate(tqdm(valid_elements, desc="构建刚度矩阵")):
            element_geom = self.element_geometry[idx]
            A = self.assembly_matrices[idx]
            
            # 计算单元刚度矩阵
            Ke = np.zeros((24, 24))
            for point in element_geom:
                Ke += point['weight'] * (point['B'].T @ self.D @ point['B'])
            
            # 组装到总矩阵
            self._assemble_matrix_optimized(K, Ke, A)
        
        return K.tocsr()
    
    def _get_K_traditional(self):
        """传统方法计算刚度矩阵"""
        K = None
        for i in tqdm(range(self.NE), desc="传统方法构建刚度矩阵"):
            coords = np.zeros((8, 3))
            for j in range(8):
                n = int(self.Ele[i][j]-1)
                coords[j] = self.node[n]
                
            Ke = get_Ke(coords[:, 0], coords[:, 1], coords[:, 2], self.D, 3)
            A = get_A(self.NN, self.Ele[i])
            
            if i == 0:
                K = A.T @ sp.sparse.csr_matrix(Ke) @ A
            else:
                K = Assemble_K(K, Ke, A)
                
        return K
    
    def _assemble_matrix_optimized(self, K_global, Ke, A):
        """优化的刚度矩阵装配"""
        # 直接装配到全局矩阵
        A_csr = A.tocsr()
        
        # 使用稀疏矩阵乘法
        K_contrib = A_csr.T @ sp.sparse.csr_matrix(Ke) @ A_csr
        
        K_global += K_contrib

    """
    寻找顶部底部结点
    """
    def find_Top(self):
        """顶层结点"""
        MaxZ = max(self.node[:,2])
        return [i+1 for i in range(self.NN) if self.node[i][2]==MaxZ]
    
    def find_Bot(self):
        """底层结点"""
        MinZ = min(self.node[:,2])
        return [i+1 for i in range(self.NN) if self.node[i][2]==MinZ]
    

    

    
    """
    计算两个矩阵（同时）- 保留原方法
    """
    def check_and_fix_element(self, ele_idx):
        """检查并修复单元节点顺序以确保雅可比行列式为正值"""
        # 获取单元节点坐标
        coords = np.zeros((8, 3))
        for j in range(8):
            n = int(self.Ele[ele_idx][j]-1)
            coords[j] = self.node[n]
        
        # 使用原始函数库中的形函数导数计算
        NA = np.zeros(8)
        NB = np.zeros(8)
        NC = np.zeros(8)
        FNA(0, 0, NA)  # 在原点处计算形函数导数
        FNB(0, 0, NB)
        FNC(0, 0, NC)
        
        # 计算雅可比矩阵
        J = np.zeros((3, 3))
        for k in range(8):
            J[0, 0] += NA[k] * coords[k, 0]
            J[0, 1] += NA[k] * coords[k, 1]
            J[0, 2] += NA[k] * coords[k, 2]
            J[1, 0] += NB[k] * coords[k, 0]
            J[1, 1] += NB[k] * coords[k, 1]
            J[1, 2] += NB[k] * coords[k, 2]
            J[2, 0] += NC[k] * coords[k, 0]
            J[2, 1] += NC[k] * coords[k, 1]
            J[2, 2] += NC[k] * coords[k, 2]
        
        det_J = np.linalg.det(J)
        
        if det_J > 0:
            return True, self.Ele[ele_idx]
        
        # 如果雅可比行列式为负值，交换底面和顶面节点
        fixed_ele = self.Ele[ele_idx].copy()
        fixed_ele[0:4], fixed_ele[4:8] = self.Ele[ele_idx][4:8].copy(), self.Ele[ele_idx][0:4].copy()
        
        # 检查修复后的节点顺序
        for j in range(8):
            n = int(fixed_ele[j]-1)
            coords[j] = self.node[n]
        
        # 重新计算形函数导数
        NA = np.zeros(8)
        NB = np.zeros(8)
        NC = np.zeros(8)
        FNA(0, 0, NA)
        FNB(0, 0, NB)
        FNC(0, 0, NC)
        
        J = np.zeros((3, 3))
        for k in range(8):
            J[0, 0] += NA[k] * coords[k, 0]
            J[0, 1] += NA[k] * coords[k, 1]
            J[0, 2] += NA[k] * coords[k, 2]
            J[1, 0] += NB[k] * coords[k, 0]
            J[1, 1] += NB[k] * coords[k, 1]
            J[1, 2] += NB[k] * coords[k, 2]
            J[2, 0] += NC[k] * coords[k, 0]
            J[2, 1] += NC[k] * coords[k, 1]
            J[2, 2] += NC[k] * coords[k, 2]
        
        det_J = np.linalg.det(J)
        
        if det_J > 0:
            print(f"  单元 {ele_idx} 节点顺序已修复")
            return True, fixed_ele
        
        print(f"  警告: 无法修复单元 {ele_idx} 的节点顺序")
        return False, self.Ele[ele_idx]
    

    
    def Fix_matrix(self, matrix):
        """添加底部结点位移约束"""
        m = matrix.copy().tocsr()
        for i in self.Bot:
            for j in range(3):
                n = 3 * (i - 1) + j
                m.data[m.indptr[n]:m.indptr[n+1]] = 0
                m[n, n] = 1
        return m.tocsr()
    
    def save_geometry_data(self, name=None):
        """保存几何信息"""
        if name is None:
            name = "data"
        
        # 检查文件是否已存在
        output_path = BASE_DIR / f"data/geometry_{name}.npz"
        if output_path.exists():
            print(f"✓ 几何文件已存在，跳过保存: {output_path}")
            return output_path
        
        geometry_data = {
            'node': np.array(self.node, np.float32),
            'elements': np.array(self.Ele, np.uint32) - 1,
            'top_nodes': np.array(self.Top, np.uint32) - 1,
            'bot_nodes': np.array(self.Bot, np.uint32) - 1,
            'mesh_shape': np.array([35, 20], np.int32),
            'metadata': {'nodes_count': self.NN, 'elements_count': self.NE, 'version': '2.0'}
        }
        
        # 处理top mesh数据
        top_indices = []
        top_vert_indices = []
        for ele in geometry_data["elements"]:
            quad_ind = [node_id for node_id in ele if node_id in geometry_data["top_nodes"]]
            if len(quad_ind) == 4:
                norm0 = face_normal(self.node[quad_ind[0]], self.node[quad_ind[1]], self.node[quad_ind[2]])
                if norm0[2] > 0:
                    top_indices.append(quad_ind)
                    top_vert_indices.append([np.where(geometry_data["top_nodes"] == node_id)[0][0] for node_id in quad_ind])
                else:
                    top_indices.append(quad_ind[::-1])
                    top_vert_indices.append([np.where(geometry_data["top_nodes"] == node_id)[0][0] for node_id in quad_ind[::-1]])
        
        geometry_data.update({
            "top_indices": np.array(top_indices, np.uint32),
            "top_vert_indices": np.array(top_vert_indices, np.uint32)
        })
        
        output_path = BASE_DIR / f"data/geometry_{name}.npz"
        np.savez(output_path, **geometry_data)
        print(f"✓ 几何信息已保存: {output_path}")
        return output_path
    
    def save_material_data(self, name=None, E=None, nu=None):
        """保存材料参数和刚度矩阵"""
        name = name or "data"
        E = E or self.E
        nu = nu or self.nu
            
        material_data = {
            'E': E, 'nu': nu,
            'KF_data': self.Kf.data,
            'KF_indices': self.Kf.indices,
            'KF_indptr': self.Kf.indptr,
            'KF_shape': self.Kf.shape,
            'metadata': {'E': E, 'nu': nu, 'version': '2.0'}
        }
        
        output_path = BASE_DIR / f"data/material_{name}.npz"
        np.savez(output_path, **material_data)
        print(f"✓ 材料参数已保存")
        return output_path

    
    @staticmethod
    def load_geometry_data(name="data"):
        """加载几何信息"""
        geometry_file = BASE_DIR / f"data/geometry_{name}.npz"
        if not geometry_file.exists():
            raise FileNotFoundError(f"几何文件不存在: {geometry_file}")
            
        with np.load(geometry_file, allow_pickle=True) as data:
            return {k: data[k] for k in ['node', 'elements', 'top_nodes', 'bot_nodes', 'mesh_shape', 'top_indices', 'top_vert_indices']}
    
    @staticmethod
    def load_material_data(name="data"):
        """加载材料参数和刚度矩阵"""
        files = list((BASE_DIR / "data").glob(f"material_{name}.npz"))
        if not files:
            raise FileNotFoundError(f"未找到材料文件")
        material_file = files[0]
            
        if not material_file.exists():
            raise FileNotFoundError(f"材料文件不存在: {material_file}")
            
        with np.load(material_file, allow_pickle=True) as data:
            return {k: data[k] for k in ['E', 'nu', 'KF_data', 'KF_indices', 'KF_indptr', 'KF_shape']}
    
    def save_data(self, name="data"):
        """保存完整数据"""
        self.save_geometry_data(name)
        self.save_material_data(name)

    def get_data(self, name="data"):
        """获取完整数据"""
        geometry_data = FEMProcessor.load_geometry_data(name)
        material_data = FEMProcessor.load_material_data(name)
        return {**geometry_data, **material_data}
    

def process_gel_data(name, dir_name=None, E=0.1983, nu=0.4795, use_cache=True, cache_dir=None):
    """处理gel数据"""
    # 读取数据文件
    file_path = f'data/{dir_name}/{name}.txt' if dir_name else f'data/{name}.txt'
    with open(file_path) as f:
        lines = f.readlines()
    
    NN, NE = map(int, lines[0].split())
    
    # 构建Node,Ele
    Node = np.zeros((NN, 3))
    Ele = np.zeros((NE, 8))
    
    for i in range(NN):
        l = lines[i+2].replace(' ', '').split(',')
        Node[i] = [float(l[j+1]) for j in range(3)]
    
    for i in range(NE):
        l = lines[i+NN+3].replace(' ', '').split(',')
        Ele[i] = [int(l[j+1]) for j in range(8)]
    
    Node *= 10  # 坐标调整
    
    # 构建Top,Bot
    top_end = NN // 3
    bot_st = NN * 2 // 3
    Top = list(range(1, top_end + 1))
    Bot = list(range(bot_st, NN + 1))
    
    # 创建处理器
    processor = FEMProcessor(Node, Ele, Top, Bot, E=E, nu=nu, 
                           cache_dir=cache_dir, use_cache=use_cache)
    
    return processor



if __name__ == '__main__':
    # 演示用法
    name = 'g1-ws'
    print("=== FEM处理器演示 ===")
    processor = process_gel_data(name, E=0.20, nu=0.45)
    processor.update_material_properties(E=0.1983, nu=0.4795)
    fem_data = processor.get_data()
    print(f"fem_data.keys: {fem_data.keys()}")