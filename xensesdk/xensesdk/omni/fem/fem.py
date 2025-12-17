"""
Author: Jin Liu
Date: 2024/12/13
"""
import cv2
import numpy as np
from pathlib import Path
from xensesdk.xenseInfer.markerConfig import MarkerConfig
from xensesdk.ezgl.experimental import compute_normals
from xensesdk.ezgl.functions import GridInterpolator
from xensesdk import PROJ_DIR

from .simpleCSR import SimpleCSR
from xensesdk.utils.decorator import deprecated
from xensesdk.xenseInterface.configManager import ConfigManager
from xensesdk.omni.transforms import Rectify2
from xensesdk.ezgl.experimental.GLEllipseItem import GLEllipseItem

class FemSolver:

    def __init__(self, Mf: SimpleCSR, Kf: SimpleCSR):
        """
        有限元求解器
        已知 K, P, 求解 V, 满足 K * V = P, 刚度矩阵K, 位移V, 外部力载荷 P

        Parameters:
        - Mf : _type_, 带底部约束质量矩阵
        - Kf : _type_, 带底部约束刚度矩阵
        """
        self.Mf = Mf        # 带底部约束质量矩阵M
        self.Kf_init = Kf        # 带底部约束刚度矩阵K
        self.Kf = None
        self.node_num = Kf.shape[0] // 3

        self.alpha = 0.01       # 瑞利阻尼 α: 0.005-0.02
        self.beta = 0.06        # 瑞利阻尼 β: 0.02-0.1

    def solve_force(self, constrain_idx, node_disp, update_Kf=False):
        """
        已知位移约束, 求解外部力载荷

        Parameters:
        - constrain_idx : List[int], 被约束的节点编号
        - node_disp : array-like of (n_nodes, 3), 所有节点的位移初始估计, 其中被约束的位置为约束位移
        - update_Kf : bool, optional, default: False, 是否更新约束矩阵

        Returns:
        - F : array-like of (n_nodes, 3), 外部力载荷
        """
        node_disp = node_disp.reshape(-1, 3)
        # 构造约束
        P = np.zeros_like(node_disp)
        P[constrain_idx] = node_disp[constrain_idx]
        P = P.flatten()

        if update_Kf:
            self.setup_Kf(constrain_idx)

        # displacement, info = sp.linalg.cg(K, P, x0=x0.flatten(), tol=0.01, maxiter=100)
        displacement, _ = SimpleCSR.solve(self.Kf, P, x0=node_disp.flatten())
        F = self.Kf_init @ displacement

        return F.reshape(-1, 3)

    def setup_Kf(self, constrain_idx):
        """
        施加约束

        Parameters:
        - K : 刚度矩阵
        - constrain_idx : array-like of (n, ), 约束节点编号
        """
        self.Kf = self.Kf_init.copy()

        if len(constrain_idx) == 0:
            return

        # 把连续的行归并
        n = len(constrain_idx)
        start = 0
        for i in range(1, n+1):
            if i==n or constrain_idx[i]-constrain_idx[i-1]>1:
                self.Kf.set_row_zero(constrain_idx[start]*3, constrain_idx[i-1]*3+3)
                start = i

        idx = np.repeat(3 * constrain_idx[:, None], repeats=3, axis=1) + np.array([0, 1, 2])
        idx = idx.flatten()
        self.Kf.set_diag_one(idx)


class FemModel:

    def __init__(
        self,
        fem_path: Path,
        grid_init,
        grid_coord_size,
        mesh_shape: tuple[int, int, int] = (3, 35, 20),
        mm_per_pix_uv: tuple[float, float] = (19.4/400, 30.8/700),
    ):
        """
        有限元模型

        Parameters:
        - fem_path : Path, fem 模型文件路径
        - grid_init : np.ndarray, shape: (n_row, n_col, 2), 初始 grid 坐标, unit: pixel
        - grid_coord_size: tuple[int, int], grid 坐标系的尺寸, (width, height), unit: pixel
        - mesh_shape : tuple[int, int, int], 网格节点形状: (mesh_n_layer, mesh_n_row, mesh_n_col)
        - mm_per_pix_uv : tuple[float, float], 深度图像素宽度 u, 高度 v, 到实际距离的转换系数, 为了避免和3d空间的歧义, 图像坐标用 UV 表示
        - u_axis : tuple[float, float, float], optional, default: (0, 1, 0), 单位向量, 表示图像 u 轴对应在 3D 空间的方向, 若为 None, u轴在空间中为曲线
        - v_axis : tuple[float, float, float], optional, default: None, 单位向量, 表示图像 v 轴对应在 3D 空间的方向, 若为 None, v轴在空间中为曲线
        """
        self.mesh_shape = mesh_shape  # 网格节点形状: (mesh_n_layer, mesh_n_row, mesh_n_col)
        self.mm_per_pix_uv = mm_per_pix_uv
        # 有限元求解器
        self.solver_force_func = None

        # 读取有限元模型
        node_xyz, c3d8_elements, top_global_idx, top_global_faces, top_local_faces = self.read_data(fem_path)
        self.node_xyz = node_xyz   # (n_nodes, 3)  unit: mm
        self.c3d8_elements = c3d8_elements
        self.top_xyz = self.node_xyz[top_global_idx].reshape(self.mesh_shape[1], self.mesh_shape[2], 3).copy()  # (mesh_n_row, mesh_n_col, 3) mesh 顶部节点坐标
        self.top_local_faces = top_local_faces
        self.top_global_faces = top_global_faces

        # 计算每个顶层节点 node 局部坐标系, 用于计算 uv 坐标到 3D 坐标的转换
        self.top_normal_axis = compute_normals(self.top_xyz.reshape(-1, 3), self.top_local_faces).reshape(self.mesh_shape[1], self.mesh_shape[2], 3)
        self.top_u_axis = None  # mesh_shape[1], mesh_shape[2], 3
        self.top_v_axis = None  # mesh_shape[1], mesh_shape[2], 3
        self.init_top_node_coord_system()

        # 生成 marker 插值矩阵, 用于将 marker 点信息插值到 mesh grid 上
        self.marker_nrow = grid_init.shape[0]  # marker 点行数
        self.marker_ncol = grid_init.shape[1]  # marker 点列数
        self.width_pix, self.height_pix = grid_coord_size  # unit: pixel
        self.marker_uv_init = grid_init.astype(np.float32)  # unit: pixel
        self.mesh_uv_init = None  # unit: pixel
        self.init_interp()

        # 用于计算 marker 位移
        self.marker_normal_axis = None
        self.marker_u_axis = None
        self.marker_v_axis = None
        self.init_marker_coord_system()
        self.marker_xyz_init = self.interp_mesh_to_marker.interp(self.top_xyz)

    def init_top_node_coord_system(self):
        """
        初始化每个顶部节点的局部坐标系, 即计算每个顶部节点的 u, v 轴单位向量, 与传感器类型有关
        """
        raise NotImplementedError

    def init_marker_coord_system(self):
        """
        初始化每个 marker 点的局部坐标系, 即计算每个 marker 点的 u, v 轴单位向量以及法向量, 根据 top node 的坐标系插值获得
        """
        self.marker_normal_axis = self.interp_mesh_to_marker.interp(self.top_normal_axis)
        self.marker_u_axis = self.interp_mesh_to_marker.interp(self.top_u_axis)
        self.marker_v_axis = self.interp_mesh_to_marker.interp(self.top_v_axis)

    def init_interp(self) -> None:
        """
        初始化从 marker 到 meshgrid 的双线性插值
        """
        mesh_n_row, mesh_n_col = self.mesh_shape[1], self.mesh_shape[2]
        mesh_u_linspace = np.linspace(0, self.width_pix-1, mesh_n_col, dtype=np.float32)
        mesh_v_linspace = np.linspace(0, self.height_pix-1, mesh_n_row, dtype=np.float32)
        self.mesh_uv_init  = np.stack(np.meshgrid(mesh_u_linspace, mesh_v_linspace, indexing="xy"), axis=-1)
        self.interp_mesh_to_marker = GridInterpolator(mesh_u_linspace, mesh_v_linspace, self.marker_uv_init)
        
        marker_u_linspace = self.marker_uv_init[0, :, 0].copy()
        marker_v_linspace = self.marker_uv_init[:, 0, 1].copy()
        self.interp_marker_to_mesh = GridInterpolator(marker_u_linspace, marker_v_linspace, self.mesh_uv_init)

    def _get_mesh_disp_uvn(self, marker_grid, depth):
        """
        已知 marker_grid 和深度图, 求解 mesh 节点 u v normal 方向位移场

        Parameters:
        - marker_grid : array-like of (n_marker, 2), marker 点坐标, unit: pixel
        - depth : array-like of (height, width), 深度图, unit: mm

        Returns:
        - disp_uv_normal : array-like of (mesh_n_row, mesh_n_col, 3), uvz方向位移场, 单位 mm
        """
        # 插值获得 mesh grid 对应的 pixel 坐标
        marker_grid = np.ascontiguousarray(marker_grid.reshape(self.marker_nrow, self.marker_ncol, 2))
        mesh_grid_uv = self.interp_marker_to_mesh.interp(marker_grid)

        # 再将 mesh_grid 作为 map, 从 depth 中插值获得深度, mm
        depth_h, depth_w = depth.shape
        scale = np.array([depth_w / self.width_pix, depth_h / self.height_pix], dtype=np.float32).reshape(1, 1, 2)
        disp_normal = cv2.remap(depth, map1=mesh_grid_uv * scale, map2=None, interpolation=cv2.INTER_LINEAR)

        disp_uv = ((mesh_grid_uv - self.mesh_uv_init) * self.mm_per_pix_uv).copy()

        self.contact_idx = np.where(disp_normal.flatten() < 0)[0]
        return np.concatenate([disp_uv, disp_normal[..., None]], axis=-1)

    def get_mesh_flow(self, marker_grid, depth):
        """
        已知 marker_grid 和深度图, 返回顶层 mesh 点的初始位置和 displacement, in xyz coordinate, 单位 mm

        Parameters:
        - marker_grid : array-like of (n_marker, 2), marker 点坐标, unit: pixel
        - depth : array-like of (height, width), 深度图, unit: mm

        Returns:
        - mesh_xyz_init : array-like of (mesh_n_row, mesh_n_col, 3), 单位 mm
        - mesh_xyz_disp : array-like of (mesh_n_row, mesh_n_col, 3), 单位 mm
        - mesh_norm_disp : array-like of (mesh_n_row, mesh_n_col, 3), 单位 mm
        """

        # ---- HACK : 插值是根据 marker config 的尺寸来的, 但是和实际上的 marker 略有偏差, 临时修正
        if not hasattr(self, "disp_init"):
            self.disp_init = self._get_mesh_disp_uvn(marker_grid, depth)

        disp_uvn = self._get_mesh_disp_uvn(marker_grid, depth)  - self.disp_init

        # disp_xyz = disp_uv_normal[..., [2]] * self.top_normal_axis
        disp_norm = disp_uvn[..., [2]] * self.top_normal_axis
        mesh_disp_xyz = disp_uvn[..., [0]] * self.top_u_axis + disp_uvn[..., [1]] * self.top_v_axis + disp_norm

        return mesh_disp_xyz, disp_norm

    @deprecated("get_mesh_flow is deprecated")
    def get_mesh_force(self, mesh_disp) -> np.ndarray:
        """
        已知顶面节点位移约束, 求解外部力载荷

        Parameters:
        - mesh_disp : array-like of (mesh_n_row, mesh_n_col, 3), 顶面节点位移约束, 单位 mm

        Returns:
        - F : array-like of (mesh_n_row * mesh_n_col, 3), 顶面节点外部力载荷
        """
        node_disp = np.zeros((*self.mesh_shape, 3))
        node_disp[0, ...] = mesh_disp
        F = self.solver_force_func(self.contact_idx, node_disp, True)[:self.mesh_shape[1]*self.mesh_shape[2]]
        return F.reshape(self.mesh_shape[1], self.mesh_shape[2], 3)

    def get_marker_flow(self, marker_grid, depth):
        """
        已知 marker_grid 和深度图, 返回 marker 点的初始位置和 displacement, in xyz coordinate, 单位 mm

        Parameters:
        - marker_grid : array-like of (n_marker, 2), marker 点坐标, unit: pixel
        - depth : array-like of (height, width), 深度图, unit: mm

        Returns:
        - marker_xyz_init : array-like of (mesh_n_row, mesh_n_col, 3), 单位 mm
        - marker_xyz_disp : array-like of (mesh_n_row, mesh_n_col, 3), 单位 mm
        """
        marker_grid = np.ascontiguousarray(marker_grid.reshape(self.marker_ncol, self.marker_ncol, 2))
        disp_uv = (marker_grid - self.marker_uv_init) * self.mm_per_pix_uv

        # 从 depth 中插值获得深度, mm
        depth_h, depth_w = depth.shape
        scale = np.array([depth_w / self.width_pix, depth_h / self.height_pix], dtype=np.float32).reshape(1, 1, 2)
        disp_n = cv2.remap( depth, map1=marker_grid * scale, map2=None, interpolation=cv2.INTER_LINEAR)
        marker_disp = disp_uv[..., [0]] * self.marker_u_axis + disp_uv[..., [1]] * self.marker_v_axis + disp_n[..., None] * self.marker_normal_axis
        return marker_disp

    def get_approx_force(self, marker_grid, depth, scale):
        # ---- HACK : 插值是根据 marker config 的尺寸来的, 但是和实际上的 marker 略有偏差, 临时修正
        if not hasattr(self, "disp_init"):
            self.disp_init = self._get_mesh_disp_uvn(marker_grid, depth)

        disp_uvn = self._get_mesh_disp_uvn(marker_grid, depth) - self.disp_init

        force_norm = (disp_uvn[..., [2]] * self.top_normal_axis * scale[2] * 1.5) / 10
        force_3d = (disp_uvn[..., [0]]*self.top_u_axis*scale[0] + disp_uvn[..., [1]]*self.top_v_axis*scale[1])/10 + force_norm

        return force_3d, force_norm
    
    def get_force_resultant(self, force):
        raise NotImplementedError("get_force_resultant is not implemented yet")

    def read_data(self, fem_path: Path):
        data = np.load(str(fem_path))
        node_xyz = data['node']  # (n_nodes, 3)  节点坐标
        c3d8_elements = data['elements']  # (n_elements, 8)
        top_global_idx = data['top_nodes']  # (n_top_nodes,)  顶部节点在全局节点中的索引
        top_global_faces = data['top_indices']  # (n_top_faces, 4)  顶部面的全局节点索引

        top_local_faces = data['top_vert_indices']  # (n_top_faces, 4)  顶部面在顶部点内部的节点索引
        top_local_faces = np.stack(
            [top_local_faces[:, :3], top_local_faces[:, [0, 2, 3]]],
            axis = 1
        ).reshape(-1, 3).astype(dtype=np.int32)  # 转换成三角形索引

        KF = SimpleCSR(data['KF_data'], data['KF_indices'], data['KF_indptr'], shape=data['KF_shape'])
        MF = SimpleCSR(data['MF_data'], data['MF_indices'], data['MF_indptr'], shape=data['MF_shape'])
        self.solver_force_func = FemSolver(MF, KF).solve_force

        # print("FEM model loaded:")
        # print(f"  num of nodes: {len(node_xyz)}")
        # print(f"  num of elements: {len(c3d8_elements)}")

        return node_xyz, c3d8_elements, top_global_idx, top_global_faces, top_local_faces

    def c3d8_faces(self):
        """
        生成六面体的面, 用三角面片表示, 用于可视化 c3d8 网格
        """
        faces = np.array([
            [0, 3, 2], [2, 1, 0], [4, 5, 6], [6, 7, 4], [0, 1, 5], [5, 4, 0],
            [2, 3, 7], [7, 6, 2], [1, 2, 6], [6, 5, 1], [0, 4, 7], [7, 3, 0]
            # [4, 5, 6], [5, 6, 7], [6, 7, 0], [7, 0, 3], [7, 1, 6], [7, 2, 5]
        ])
        return self.c3d8_elements[:, faces]

    @classmethod
    def create(cls, sensor_type, grid_init, grid_coord_size):
        """
        根据传感器类型创建有限元模型
        """
        if sensor_type.name == "Omni":
            return OmniFemModel(PROJ_DIR/"xenseInterface/guiConfig/g1-os.npz", grid_init, grid_coord_size)
        elif sensor_type.name == "VecTouch":
            return WedgeFemModel(PROJ_DIR/"xenseInterface/guiConfig/g1-ws.npz", grid_init, grid_coord_size)
        elif sensor_type.name == "Finger":
            return FingerFemModel(PROJ_DIR/"omni/assets/init_gel_depth.npy", grid_init, grid_coord_size)
        else:
            raise ValueError(f"Not supported sensor type: {sensor_type.name}")


class OmniFemModel(FemModel):

    def __init__(
        self,
        fem_path: Path,
        grid_init,
        grid_coord_size,
        mesh_shape: tuple[int, int, int] = (3, 35, 20),
        mm_per_pix_uv: tuple[float, float] = (20/400, 35/700),
        # mm_per_pix_uv: tuple[float, float] = (17.2/400, 28.5/700),
    ):
        super().__init__(fem_path, grid_init, grid_coord_size, mesh_shape, mm_per_pix_uv)

    def init_top_node_coord_system(self):
        """
        初始化每个顶部节点的局部坐标系, 即计算每个顶部节点的 u, v 轴单位向量
        """
        self.top_u_axis = np.zeros_like(self.top_xyz)
        self.top_u_axis[:] = (self.top_xyz[:, 1] - self.top_xyz[:, 0]).reshape(-1, 1, 3)

        self.top_v_axis = np.zeros_like(self.top_xyz)
        self.top_v_axis[:-1] = self.top_xyz[1:] - self.top_xyz[:-1]
        self.top_v_axis[-1] =  self.top_v_axis[-2]

        # self.top_normal_axis *= 1.4 # 增大按压深度

    def get_force_resultant(self, force):
        center = np.array([10.9, 0, 28.65])
        mesh_init = self.top_xyz.reshape(-1, 3)
        force = force.reshape(-1, 3)
        
        res_torque = np.sum(np.cross(mesh_init - center, force), axis=0) * 0.3
        res_force = np.sum(force, axis=0)
        return np.concatenate([res_force, res_torque], axis=0)       


class WedgeFemModel(FemModel):

    def __init__(
        self,
        fem_path: Path,
        grid_init,
        grid_coord_size,
        mesh_shape: tuple[int, int, int] = (3, 35, 20),
        mm_per_pix_uv: tuple[float, float] = (20/400, 35/700),
    ):
        super().__init__(fem_path, grid_init, grid_coord_size, mesh_shape, mm_per_pix_uv)

    def init_top_node_coord_system(self):
        """
        初始化每个顶部节点的局部坐标系, 即计算每个顶部节点的 u, v 轴单位向量
        """
        self.top_u_axis = np.zeros_like(self.top_xyz)
        self.top_u_axis[:] = np.array([1, 0, 0]).reshape(1, 1, 3)
        self.top_v_axis = np.zeros_like(self.top_xyz)
        self.top_v_axis[:] = np.array([0, -1, 0]).reshape(1, 1, 3)
    
    def get_force_resultant(self, force):
        center = np.array([0, 15, 0.])
        mesh_init = self.top_xyz.reshape(-1, 3)
        force = force.reshape(-1, 3)
        
        res_torque = np.sum(np.cross(mesh_init - center, force), axis=0) * 0.3
        res_force = np.sum(force, axis=0)
        return np.concatenate([res_force, res_torque], axis=0)


class FingerFemModel(FemModel):

    def __init__(
        self,
        fem_path: Path,
        grid_init,
        grid_coord_size,
        mesh_shape: tuple[int, int, int] = (1, 35, 20),
        mm_per_pix_uv: tuple[float, float] = (10/400, 15/700),
    ):
        self.mesh_shape = mesh_shape  # 网格节点形状: (mesh_n_row, mesh_n_col)
        self.width_pix, self.height_pix = grid_coord_size  # unit: pixel
        self.mm_per_pix_uv = mm_per_pix_uv
        self.marker_shape = (grid_init.shape[0], grid_init.shape[1])
        self.marker_nrow = grid_init.shape[0]  # marker 点行数
        self.marker_ncol = grid_init.shape[1]  # marker 点列数

        # 读取 mesh 数据
        self.top_xyz, self.top_normal_axis, self.top_u_axis, self.top_v_axis = self.get_mesh_data(fem_path, mesh_size=(self.mesh_shape[2], self.mesh_shape[1]))

        # 构造 mesh 到 marker 的插值
        self.marker_uv_init = grid_init.astype(np.float32)  # unit: pixel
        self.mesh_uv_init = None  # unit: pixel
        self.init_interp()

        # 用于计算 marker 位移
        self.marker_normal_axis = None
        self.marker_u_axis = None
        self.marker_v_axis = None
        self.init_marker_coord_system()
        self.marker_xyz_init = self.interp_mesh_to_marker.interp(self.top_xyz)
           
    def get_mesh_data(self, path, mesh_size):
        """ 获取指尖传感器的 mesh 数据 """
        init_gel = np.load(path)
        grid_size = np.array((400/4, 700/4), np.int32)
        gel = GLEllipseItem(ellipse_size=(1.1, 1.8), theta_range=(0.0, np.pi/2), grid_size=grid_size, init_zmap=init_gel)
        
        rectify = Rectify2(
            src_size=grid_size,
            src_grid=np.array(
                [[[400.000, 153.000], [266.000, 125.000], [134.000, 125.000], [  0.000, 153.000]],
                [[342.000, 272.000], [249.000, 258.000], [151.000, 258.000], [ 58.000, 272.000]],
                [[307.000, 417.000], [231.000, 403.000], [169.000, 403.000], [ 93.000, 417.000]],
                [[287.000, 567.000], [225.000, 558.000], [175.000, 558.000], [113.000, 567.000]],
                [[287.000, 694.000], [228.000, 699.000], [172.000, 699.000], [113.000, 694.000]]]
            ) / 4, 
            dst_size=mesh_size,
            pad=(0.1, 0.1, 0.1, 0.1),
            is_inverse=False,
        )
        vert = rectify(gel._init_vertices.reshape(grid_size[1], grid_size[0], 3))
        norm = rectify(gel._init_normals.reshape(grid_size[1], grid_size[0], 3))
        vert = np.flip(vert, axis=1)  # 翻转 x 轴, 使得左上角为起始点
        norm = np.flip(norm, axis=1)
        
        u_axis = np.zeros_like(vert)
        u_axis[:, :-1] = vert[:, 1:] - vert[:, :-1]
        u_axis[:, -1] = u_axis[:, -2]  # 最左侧的 u 轴和第二个节点相同
        u_axis /= np.linalg.norm(u_axis, axis=-1, keepdims=True)  # 归一化
        
        v_axis = np.zeros_like(vert)
        v_axis[:-1] = vert[1:] - vert[:-1]
        v_axis[-1] =  v_axis[-2]
        v_axis /= np.linalg.norm(v_axis, axis=-1, keepdims=True)
        return vert*10, norm, u_axis, v_axis  # vert *10 将单位从 cm 转换为 mm

    def get_force_resultant(self, force):
        center = np.array([0, 10, 15])
        mesh_init = self.top_xyz.reshape(-1, 3)
        force = force.reshape(-1, 3)
        
        res_torque = np.sum(np.cross(mesh_init - center, force), axis=0) * 0.3
        res_force = np.sum(force, axis=0)
        return np.concatenate([res_force, res_torque], axis=0) 