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

def shape_functions_and_derivatives(p: np.ndarray):
    """
    计算八节点六面体单元的形函数及其对局部坐标的导数。
    """
    from Function import NODE_LOCAL_COORDS
    c = NODE_LOCAL_COORDS.T
    val = 1 + p[:, :, np.newaxis] * c[np.newaxis, :, :]
    N = 0.125 * np.prod(val, axis=1)
    dNdabc = np.zeros((p.shape[0], 3, 8))
    dNdabc[:, 0, :] = 0.125 * c[0, :] * val[:, 1, :] * val[:, 2, :]
    dNdabc[:, 1, :] = 0.125 * c[1, :] * val[:, 0, :] * val[:, 2, :]
    dNdabc[:, 2, :] = 0.125 * c[2, :] * val[:, 0, :] * val[:, 1, :]
    return N, dNdabc

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
        self.Ele = Ele      # 单元结点编号 (1-based index)
        self.E = E          # 杨氏模量
        self.nu = nu        # 泊松比
        self.NN = len(self.node) # 结点总数
        self.NE = len(self.Ele)  # 单元总数
        
        self.Top = Top  # 顶层结点
        self.Bot = Bot  # 底层结点
        self.use_cache = use_cache  # 是否使用缓存
        
        # 0-based 索引，在内部使用，确保为整数类型
        self.Ele_zero_based = (self.Ele - 1).astype(np.int32)
        
        # 设置缓存目录
        self.cache_dir = Path(cache_dir) if cache_dir else BASE_DIR / "cache"
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # 检查或计算几何信息
        if not self.use_cache or not self._check_and_load_geometry_cache():
            print("🔄 计算几何信息...")
            self._compute_geometry()
            if self.use_cache:
                self._save_geometry_cache()
        
        self.save_geometry_data()
        # 计算刚度矩阵
        self.update_material_properties(E, nu)
    
    def _compute_geometry_hash(self):
        """计算几何数据的哈希值用于缓存"""
        geometry_data = np.concatenate([
            self.node.flatten(),
            self.Ele.flatten(),
            np.array(self.Top),
            np.array(self.Bot)
        ])
        return hashlib.md5(geometry_data.tobytes()).hexdigest()[:16]

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
                # 验证缓存版本和内容
                if cache_data.get('metadata', {}).get('cache_version') != '3.0-fastpath':
                    print("⚠ 缓存版本不匹配，将重新计算。")
                    return False

                self.iK = cache_data['iK']
                self.jK = cache_data['jK']
                self.fixed_elements_mask = cache_data['fixed_elements_mask']
                self.Ele_zero_based = cache_data['Ele_zero_based'].astype(np.int32)
                self.Ele = self.Ele_zero_based + 1 # 同步1-based
                self.is_regular_grid = cache_data['is_regular_grid']
                if self.is_regular_grid:
                    self.template_coords = cache_data['template_coords']
                    print(f"✅ 加载规则网格缓存，将使用快车道模式")
                else:
                    print(f"ℹ️ 加载不规则网格缓存，将使用通用向量化模式")
            
            print(f"✅ 几何信息从缓存加载: {self.NE} 个单元")
            return True
            
        except (Exception, KeyError) as e:
            print(f"⚠ 缓存文件损坏或格式错误，将重新计算: {e}")
            geometry_cache_file.unlink(missing_ok=True)
            return False
    
    def _save_geometry_cache(self):
        """保存几何信息到缓存"""
        geometry_hash = self._compute_geometry_hash()
        geometry_cache_file = self.cache_dir / f"geometry_{geometry_hash}.pkl"
        
        cache_data = {
            'iK': self.iK,
            'jK': self.jK,
            'fixed_elements_mask': self.fixed_elements_mask,
            'Ele_zero_based': self.Ele_zero_based,
            'is_regular_grid': self.is_regular_grid,
            'metadata': {'cache_version': '3.0-fastpath'}
        }
        if self.is_regular_grid:
            cache_data['template_coords'] = self.template_coords

        try:
            with open(geometry_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ 几何信息已保存到缓存: {geometry_cache_file.name}")
        except Exception as e:
            print(f"⚠ 缓存保存失败: {e}")
    
    def _compute_geometry(self):
        """一次性计算所有几何相关信息"""
        # 1. 修复单元节点顺序，并获取有问题的单元掩码
        self.fixed_elements_mask = self._check_and_fix_all_elements()
        
        # 2. 准备所有单元的节点坐标
        all_element_coords = self.node[self.Ele_zero_based]
        
        # 3. 检查是否为规则网格
        dims = np.max(all_element_coords, axis=1) - np.min(all_element_coords, axis=1)
        valid_dims = dims[~self.fixed_elements_mask]
        
        # 更宽松的规则网格检测
        if valid_dims.shape[0] > 0:
            # 计算每个单元的体积作为规则性指标
            volumes = np.prod(valid_dims, axis=1)
            mean_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            
            # 如果体积变化很小，认为是规则网格
            self.is_regular_grid = (volume_std / mean_volume) < 0.1 if mean_volume > 1e-12 else False
            
            if not self.is_regular_grid:
                # 备用检测：检查单元尺寸的相对标准差
                dim_means = np.mean(valid_dims, axis=0)
                dim_stds = np.std(valid_dims, axis=0)
                relative_stds = dim_stds / (dim_means + 1e-12)
                self.is_regular_grid = np.all(relative_stds < 0.1)
        else:
            self.is_regular_grid = False

        if self.is_regular_grid:
            print("✅ Detected regular grid. Using fast path for Ke computation.")
            first_valid_idx = np.where(~self.fixed_elements_mask)[0]
            if first_valid_idx.size > 0:
                 self.template_coords = all_element_coords[first_valid_idx[0]]
            else:
                 self.is_regular_grid = False
                 print("⚠️ All elements have invalid Jacobians, falling back to general path.")
        else:
            if valid_dims.shape[0] > 0:
                print(f"ℹ️ Using general path for Ke computation on irregular grid. "
                      f"Volume std/mean: {volume_std/mean_volume:.4f}")
            else:
                print("ℹ️ Using general path for Ke computation on irregular grid. Volume std/mean: N/A")

        # 4. 计算用于稀疏矩阵组装的全局索引 iK, jK
        dofs_per_element = 24
        sII, sI = np.triu_indices(dofs_per_element)
        
        ele_dof = np.repeat(self.Ele_zero_based, 3, axis=1) * 3 + \
                  np.tile(np.arange(3), (self.NE, 8))
        
        iK_all = ele_dof[:, sI]
        jK_all = ele_dof[:, sII]
        
        self.iK = np.maximum(iK_all, jK_all).flatten()
        self.jK = np.minimum(iK_all, jK_all).flatten()
        
    def _check_and_fix_all_elements(self):
        """一次性向量化检查并修复所有单元"""
        print("检查并修复单元...")
        all_coords = self.node[self.Ele_zero_based]
        
        p_center = np.array([[0.0, 0.0, 0.0]])
        _, dNdabc = shape_functions_and_derivatives(p_center)
        
        J = np.einsum('gik,ekj->egij', dNdabc, all_coords)
        det_J = np.linalg.det(J).flatten()
        
        problem_mask = det_J <= 0
        problem_indices = np.where(problem_mask)[0]
        
        if problem_indices.size > 0:
            print(f"发现 {len(problem_indices)} 个雅可比行列式为负的单元，正在尝试修复...")
            
            original_eles = self.Ele_zero_based[problem_mask].copy()
            fixed_eles = original_eles[:, [4, 5, 6, 7, 0, 1, 2, 3]]
            
            fixed_coords = self.node[fixed_eles]
            J_fixed = np.einsum('gik,ekj->egij', dNdabc, fixed_coords)
            det_J_fixed = np.linalg.det(J_fixed).flatten()
            
            still_problem_mask = det_J_fixed <= 0
            if np.any(still_problem_mask):
                still_problem_indices = problem_indices[still_problem_mask]
                print(f"警告: 无法修复 {len(still_problem_indices)} 个单元: {still_problem_indices}")

            successfully_fixed_mask = ~still_problem_mask
            self.Ele_zero_based[problem_indices[successfully_fixed_mask]] = fixed_eles[successfully_fixed_mask]
            self.Ele = self.Ele_zero_based + 1

        return problem_mask
    
    def update_material_properties(self, E=None, nu=None):
        """更新材料属性并重新计算刚度矩阵"""
        if E is not None: self.E = E
        if nu is not None: self.nu = nu
            
        print(f"更新材料参数: E={self.E}, nu={self.nu}")
        
        self.D = eld(self.E, self.nu)
        
        self.K = self.get_K_optimized()
        self.Kf = self.Fix_matrix(self.K)
        self.save_material_data()
    
    def get_K_optimized(self):
        """优化的刚度矩阵计算"""
        print("构建刚度矩阵...")
        
        if self.is_regular_grid:
            print("🚀 Using fast path for regular grid...")
            sK_template = get_all_Ke_contributions(self.template_coords[np.newaxis, ...], self.D)[0]
            sK_full = np.tile(sK_template, (self.NE, 1))
            sK_full[self.fixed_elements_mask] = 0.0
        else:
            print("🐢 Using general vectorized path for irregular grid...")
            all_element_coords = self.node[self.Ele_zero_based]
            valid_coords = all_element_coords[~self.fixed_elements_mask]
            
            if valid_coords.shape[0] > 0:
                sK_all_valid = get_all_Ke_contributions(valid_coords, self.D)
                sK_full = np.zeros((self.NE, sK_all_valid.shape[1]))
                sK_full[~self.fixed_elements_mask] = sK_all_valid
            else:
                sK_full = np.zeros((self.NE, 36))
        
        sK = sK_full.flatten()
        
        K = sp.sparse.coo_matrix((sK, (self.iK, self.jK)),
                           shape=(3*self.NN, 3*self.NN)).tocsr()
        
        K = K + K.T - sp.sparse.diags(K.diagonal())
        return K

    def find_Top(self):
        """顶层结点"""
        MaxZ = np.max(self.node[:,2])
        return np.where(np.abs(self.node[:,2] - MaxZ) < 1e-6)[0] + 1
    
    def find_Bot(self):
        """底层结点"""
        MinZ = np.min(self.node[:,2])
        return np.where(np.abs(self.node[:,2] - MinZ) < 1e-6)[0] + 1
    
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
        
        output_path = BASE_DIR / f"data/geometry_{name}.npz"
        if output_path.exists():
            print(f"✓ 几何文件已存在，跳过保存: {output_path}")
            return output_path
        
        geometry_data = {
            'node': np.array(self.node, np.float32),
            'elements': np.array(self.Ele_zero_based, np.uint32),
            'top_nodes': np.array(self.Top, np.uint32) - 1,
            'bot_nodes': np.array(self.Bot, np.uint32) - 1,
            'mesh_shape': np.array([35, 20], np.int32), # This might need to be dynamic
            'metadata': {'nodes_count': self.NN, 'elements_count': self.NE, 'version': '3.0'}
        }
        
        top_nodes_set = set(geometry_data["top_nodes"])
        top_indices = []
        top_vert_indices_map = {node_id: i for i, node_id in enumerate(geometry_data["top_nodes"])}

        for ele in geometry_data["elements"]:
            quad_ind = [node_id for node_id in ele if node_id in top_nodes_set]
            if len(quad_ind) == 4:
                p0 = self.node[quad_ind[0]]
                p1 = self.node[quad_ind[1]]
                p2 = self.node[quad_ind[2]]
                norm_z = np.cross(p1 - p0, p2 - p0)[2]
                
                if norm_z < 0:
                    quad_ind = quad_ind[::-1]
                
                top_indices.append(quad_ind)
        
        top_indices_arr = np.array(top_indices, np.uint32)
        top_vert_indices = np.vectorize(top_vert_indices_map.get)(top_indices_arr)

        geometry_data.update({
            "top_indices": top_indices_arr,
            "top_vert_indices": np.array(top_vert_indices, np.uint32)
        })
        
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
    import time
    name = 'g1-ws'
    print("=== FEM处理器演示 (高性能版) ===")
    
    start_time = time.time()
    # Force cache to be disabled for the first run to see computation time
    print("\n--- 首次运行 (无缓存) ---")
    processor_fresh = process_gel_data(name, E=0.20, nu=0.45, use_cache=False)
    end_time = time.time()
    print(f"首次计算总耗时: {end_time - start_time:.4f} 秒")

    start_time = time.time()
    print("\n--- 第二次运行 (有缓存) ---")
    processor_cached = process_gel_data(name, E=0.20, nu=0.45, use_cache=True)
    end_time = time.time()
    print(f"缓存加载总耗时: {end_time - start_time:.4f} 秒")

    start_time = time.time()
    print("\n--- 更新材料参数 ---")
    processor_cached.update_material_properties(E=0.1983, nu=0.4795)
    end_time = time.time()
    print(f"更新材料和重算刚度矩阵耗时: {end_time - start_time:.4f} 秒")
    
    fem_data = processor_cached.get_data()
    print(f"\nfem_data.keys: {fem_data.keys()}")
    # print(f"fem_data['KF_data']: {fem_data['KF_data']}")