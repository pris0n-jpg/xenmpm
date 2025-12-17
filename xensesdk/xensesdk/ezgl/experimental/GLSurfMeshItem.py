import cv2
import numpy as np
import OpenGL.GL as gl

from ..GLGraphicsItem import GLGraphicsItem
from ..items import GLMeshItem

from . import compute_normals
from .GLEllipseItem import surface_indices, surface_vertexes


class MeshSmoother:

    def __init__(self, num_vertices, indices, num_neighbors=6):
        """
        Mesh 平滑器

        Parameters:
        - num_vertices : int, 顶点数
        - indices : np.ndarray(n, 3), 三角形索引
        - num_neighbors : int, optional, default: 6, 每个顶点的邻居数, 边缘点(少于6个邻居)不进行平滑
        """
        self.num_vert = num_vertices
        self.num_neigh = num_neighbors

        neighbors = [[] for _ in range(self.num_vert)]

        for quad in indices:
            for i in range(3):
                neighbors[quad[i]].extend([quad[(i+1)%3], quad[(i+2)%3]])

        # 去重邻居顶点
        for i in range(self.num_vert):
            neigh = list(set(neighbors[i]))
            if len(neigh) > self.num_neigh:
                neigh = neigh[:self.num_neigh]
            elif len(neigh) < self.num_neigh:
                neigh = [i] * self.num_neigh  # 边缘点为自身

            neighbors[i] = neigh

        self.neighbors = np.array(neighbors, dtype=np.int32)

    def smooth(self, vert_wise_data, niters=1):
        """
        平滑顶点数据

        Parameters:
        - vert_wise_data : np.ndarray(n, k), 顶点数据, n为顶点数, k为数据维度
        - niters : int, optional, default: 1, 迭代次数
        """
        for _ in range(niters):
            extracted_data = np.take_along_axis(vert_wise_data[None, ...], self.neighbors[..., None], axis=1)
            vert_wise_data = np.mean(extracted_data, axis=1, keepdims=False)

        return vert_wise_data


class GLSurfMeshItem(GLGraphicsItem):

    def __init__(
        self,
        shape: tuple,
        x_range: tuple,
        y_range: tuple,
        zmap: np.ndarray=None,
        lights=list(),
        material=None,
        show_edge=False,
        glOptions="opaque",
        parentItem=None
    ):
        """
        从深度图生成 Mesh 网格

        Parameters:
        - shape : tuple(nrow, ncol), 顶点行数, 列数
        - x_range : tuple, w 方向顶点坐标范围
        - y_range : tuple, h 方向顶点坐标范围
        - zmap : np.ndarray, 单通道深度图
        - lights : list, optional, default: list(), 光源
        - material : Material, optional, default: None
        - show_edge : bool, optional, default: False
        - glOptions : str, optional, default: "opaque"
        - parentItem : GLGraphicsItem, optional, default: None
        """
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self._nrow, self._ncol = shape
        self._x_range = x_range
        self._y_range = y_range
        self._indices = surface_indices(self._nrow, self._ncol)
        self._vertices_init = surface_vertexes(np.zeros((10, 10)), self._x_range, self._y_range, self._nrow, self._ncol)
        self._normals = np.zeros_like(self._vertices_init)
        self._normals[:, 2] = 1
        self._smoother = MeshSmoother(self._nrow*self._ncol, self._indices, 6)

        self.setData(zmap)

        self.mesh_item = GLMeshItem(
            vertexes=self._vertices_init,
            indices=self._indices,
            normals=self._normals,
            lights=lights,
            material=material,
            calc_normals=False,
            mode=gl.GL_TRIANGLES,
            show_edge=show_edge,
            parentItem=self
        )

    def setData(self, zmap=None, smooth=1):
        """
        设置深度图

        Parameters:
        - zmap : np.ndarray, default: None, 单通道深度图
        - smooth : int, optional, default: 1, 平滑次数
        """
        if zmap is None:
            return

        zmap = cv2.resize(zmap, (self._ncol, self._nrow), interpolation=cv2.INTER_LINEAR)
        vertices = self._vertices_init.copy()
        vertices[:, 2] = zmap.reshape(-1)
        self._normals = compute_normals(vertices, self._indices)

        if smooth:
            vertices = self._smoother.smooth(vertices, smooth)
            self._normals = self._smoother.smooth(self._normals, smooth)

        self.mesh_item.setData(vertexes=vertices, normals=self._normals)