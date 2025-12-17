"""
Description: 相机系统

Author: Jin Liu
Date: 2024/06/26
"""

from math import radians, tan
from ..transform3d import Matrix4x4, Quaternion, Vector3
from ..GLGraphicsItem import GLGraphicsItem
from .GLLinePlotItem import GLLinePlotItem
import numpy as np
from typing import Sequence

class Camera(GLGraphicsItem):

    axis_verts = np.array([ 0, 0, 0, 1, 0, 0,  1, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 1, 0,  0, 1, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 1,  0, 0, 1, 0, 0, 0])
    axis_colors = np.array(
        [1, 0, 0, 1,  1, 0, 0, 1,  1, 0, 0, 1,  1, 0, 0, 1,
            0, 1, 0, 1,  0, 1, 0, 1,  0, 1, 0, 1,  0, 1, 0, 1,
            0, 0, 1, 1,  0, 0, 1, 1,  0, 0, 1, 1,  0, 0, 1, 1]
    )
    frustum_verts = np.array([
        [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1], [-1, -1, -1],
        [1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1], [1, -1, -1],
        [1, 1, -1], [-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1],
        [1, 1, 1],  [0, 0, 1], [0, 0, 0]
    ])

    def __init__(
        self,
        eye: Sequence = (0., 0., 5),
        center: Sequence = (0., 0., 0),
        up: Sequence = (0., 1., 0),
        aspect = 4/3,
        fov = 45,
        ortho_space: Sequence = (-4, 4, -4, 4, -5, 20),
        proj_type = 'perspective',  # perspective or ortho
        near_far_wrt_distance: bool = True,
        glOptions = 'opaque',
        parentItem = None,
        frustum_visible = False,
    ):
        """
        相机系统

        Parameters:
        - eye : tuple, optional, default: (0., 0., 5), 相对于parent系的相机位置
        - center : tuple, optional, default: (0., 0., 0), 相对于parent系的观察目标位置, eye->center 为相机系 z 轴负方向
        - up : tuple, optional, default: (0., 1., 0), 相对于parent系的相机坐标系 y 轴正方向
        - aspect : float, optional, default: 1, 屏幕宽高比
        - fov : int, optional, default: 45, 相机视场角, 默认为透视投影, 可以使用 setProject 修改为正交投影
        - ortho_space : tuple, optional, default: (-4, 4, -4, 4, -20, 20), 正交投影空间, (left, right, bottom, top, near, far)
        - proj_type : str, optional, default: 'perspective', 投影类型, 'perspective' or 'ortho'
        - near_far_wrt_distance : bool, optional, default: True, near 和 far 是否相对于 origin 定义, 只在 ortho 模式下有效
        - glOptions : str, optional, default: 'opaque'
        - parentItem : GLGraphicsItem, optional, default: None
        - frustum_visible : bool, optional, default: False, 是否显示相机视锥体
        """

        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self.perspective_far = 100  # perspective 模式下, 默认远裁剪面距离
        self._near_far_wrt_distance = near_far_wrt_distance
        self.axis_plot = GLLinePlotItem(lineWidth=2, glOptions='opaque', parentItem=self)
        self.axis_plot.setData(self.axis_verts, self.axis_colors)
        self.frustum_plot = GLLinePlotItem(lineWidth=2, glOptions='opaque', parentItem=self)

        self.set_view_matrix(Matrix4x4.lookAt(eye, center, up))
        self.set_proj_matrix(aspect, fov, ortho_space, proj_type)
        self.setVisible(frustum_visible, recursive=True)

    def get_view_matrix(self) -> Matrix4x4:
        """
        世界坐标系到相机坐标系的变换矩阵
        """
        return self.transform(local = False).inverse()

    def set_view_matrix(self, view_matrix: Matrix4x4):
        """
        设置世界坐标系到相机坐标系的变换矩阵, 通过设置相机坐标系的变换矩阵来控制相机的位置和朝向 view_matrix = model_matrix.inverse()
        当相机有 parentItem 时, camera.model_global = view.inverse() =  parent.model_global * camera.model_local
        -> camera.model_local = (view * parent.model_global).inverse()

        Parameters:
        - view_matrix : Matrix4x4, 世界坐标系到相机坐标系的变换矩阵
        """
        if self.parentItem() is not None:
            self.setTransform((view_matrix * self.parentItem().transform(False)).inverse())
        else:
            self.setTransform(view_matrix.inverse())

    def get_vector6d(self) -> np.ndarray:
        return self.get_view_matrix().toVector6d()

    def set_vector6d(self, vector6d):
        self.set_view_matrix(Matrix4x4.fromVector6d(*vector6d))

    def get_distance(self):
        """计算原点在相机视线方向上距离相机的距离"""
        return -self.get_view_matrix()[2, 3]

    def get_center(self) -> np.ndarray:
        """计算相机朝向的点在世界坐标系下的坐标"""
        z = self.get_distance()
        center = self.transform(local=False) * np.array([0, 0, -z])
        return center

    def get_eye(self) -> np.ndarray:
        """计算相机在世界坐标系下的坐标"""
        return self.transform(local=False).toTranslation()

    def set_proj_matrix(self, aspect=None, fov=None, ortho_space=None, proj_type=None):
        """
        设置投影方式
        """
        if aspect is not None:
            self._aspect = aspect
        if fov is not None:
            self._fov = fov
        if ortho_space is not None:
            self._ortho_space = ortho_space
        if proj_type is not None:
            self._perspective = proj_type == 'perspective'

        # 更新视锥体在相机坐标系下的顶点
        self.frustum_plot.setData(self.get_proj_matrix().inverse() * self.frustum_verts)

    def get_proj_matrix(self) -> Matrix4x4:
        """
        计算投影矩阵
        """
        distance = self.get_distance()

        if self._perspective:
            return Matrix4x4.perspective(
                self._fov,
                self._aspect,
                max(0.01 * distance, 0.001),
                distance + self.perspective_far
            )
        else:
            near, far = self._ortho_space[4:]
            if self._near_far_wrt_distance:
                near, far = near + distance, far + distance
            return Matrix4x4.ortho(*self._ortho_space[:4], near, far)

    def get_proj_view_matrix(self) -> Matrix4x4:
        return self.get_proj_matrix() * self.get_view_matrix()

    def orbit(self, rotx, roty, rotz, base: Matrix4x4=None):
        """
        在相机坐标系中, 将物体围绕世界坐标系原点旋转

        Parameters:
        - rotx, roty, rotz : [degree], zyx欧拉角
        - base : Matrix4x4, optional, default: None, 相对基准 view_matrix
        """
        view_matrix = self.get_view_matrix() if base is None else base.copy()
        tr = Matrix4x4.fromVector6d(*view_matrix.toTranslation(), rotx, roty, rotz)
        self.set_view_matrix(tr * view_matrix.moveto(0,0,0))

    def pan(self, dx, dy, dz=0.0, scale=0.001, base=None):
        """
        在相机坐标系中, 平移世界坐标系

        Parameters:
        - dx, dy, dz : [m]
        - scale : float, optional, default: 0.001, 缩放系数
        - base : Matrix4x4, optional, default: None, 相对基准 view_matrix
        """
        view_matrix = self.get_view_matrix() if base is None else base.copy()
        distance = max(-view_matrix[2, 3], 1)
        scale = distance * 2. * tan(0.5 * radians(self._fov)) * scale
        view_matrix.translate(dx*scale, dy*scale, dz*scale, False)
        view_matrix[2, 3] = min(-0.1, view_matrix[2, 3])
        self.set_view_matrix(view_matrix)
