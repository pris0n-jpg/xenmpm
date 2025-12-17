import numpy as np
import cv2
import OpenGL.GL as gl
from typing import Sequence

from xensesdk.ezgl.items import (GLModelItem, GLMeshItem, GLScatterPlotItem, Texture2D, Material, Mesh, 
                                 vertex_normal, GLInstancedMeshItem, sphere, Shader, DepthCamera, GLArrowMeshItem, PointLight)
from xensesdk.ezgl.experimental.GLSurfMeshItem import GLSurfMeshItem, GLGraphicsItem
from xensesdk.ezgl import GLViewWidget
from xensesdk.ezgl import Matrix4x4
from xensesdk.ezgl.items.GLCircularArrowItem import GLCircularArrowItem
from xensesdk.ezgl.utils.colormap import cm
from . import PROJ_DIR


class GLWedgeSensorItem(GLGraphicsItem):

    def __init__(
        self,
        lights: Sequence,
        grid_size=(110, 200),
        glOptions='opaque',
        parentItem=None,
        smooth_iter:int=1,
        scale_factor=1,
    ):
        """
        楔形传感器模型

        Parameters:
        - lights : Sequence, 光源
        - grid_size : tuple, optional, default: (110, 200), 表面网格大小
        - glOptions : str, optional, default: 'opaque', OpenGL 选项
        - parentItem : GLGraphicsItem, optional, default: None
        - smooth_iter : int, optional, default: 1, 平滑迭代次数
        - scale_factor : int, optional, default: 10, 模型缩放因子
        """
        super().__init__(parentItem=parentItem)
        self._grid_size = grid_size  # (cols, rows)
        self._scale_factor = scale_factor
        self._smooth_iter = smooth_iter
        
        self.width = 1.73
        self.height = 2.914

        self.sensor_base = GLModelItem(
            PROJ_DIR / "assets/g1-ws/vm_g1-ws.obj",
            lights = lights,
            glOptions = glOptions,
            parentItem = self,
        ).rotate(90, 1, 0, 0).rotate(90, 0, 0, 1).translate(0, -1.32, -2.2)
        self.sensor_base.setPaintOrder([0])
        self.sensor_base.setMaterial(0, Material(ambient=(0.18, 0.18, 0.18), diffuse=(0.2, 0.2, 0.35), specular=(0.3, 0.3, 0.3), shininess=5))

        for mesh in self.sensor_base.meshes:
            mesh._vertexes.set_data( mesh._vertexes.data / scale_factor )  # 将单位转成 cm

        self._gel_texture = Texture2D()
        self._gel_material = Material(
            ambient=(0.3, 0.3, 0.3), diffuse=(0.9, 0.9, 0.9), specular=(0.1, 0.1, 0.1), shininess=5,
        )
        self._gel = GLSurfMeshItem(
            shape=(self._grid_size[1], self._grid_size[0]), 
            x_range=(-self.width/2, self.width/2), 
            y_range=(self.height+0.1, 0.1), 
            lights=lights, 
            show_edge=False,
            material=self._gel_material, 
            parentItem=self
        )
        
        # 设置纹理坐标
        uv = np.zeros((self._grid_size[1], self._grid_size[0], 2), dtype='f4')
        uv[:, :, 0] = np.linspace(1, 0, self._grid_size[0])[None, :]
        uv[:, :, 1] = np.linspace(0, 1, self._grid_size[1])[:, None]
        self._gel.mesh_item._mesh._texcoords.set_data(uv)

        verts, faces, uv, norms = sphere(0.02, 8, 8, True)
        self.marker = GLInstancedMeshItem(Mesh(verts, faces, normals=norms), lights=lights, color=(0.3, 0.6, 1.2), glOptions="translucent_cull", parentItem=self)
        self.marker.setDepthValue(2)
        self.light1 = PointLight(pos=(5, 10, 10), ambient=(0.7, 0.7, 0.7), diffuse=(1, 1, 1), visible=False, directional=True, render_shadow=False)
        self.arrow_plot = GLArrowMeshItem(tip_size=[0.02, 0.022], width=0.012, lights=[self.light1], color=[1, 0.2, 0.2], parentItem=self)
        self.torque_plot = GLCircularArrowItem(length=0, inner_radius=0.02, outer_radius=0.25, lights=[self.light1], color=[1, 0.2, 0.2], parentItem=self)
        self.force_plot = GLArrowMeshItem(tip_size=[0.035, 0.02], width=0.02, lights=[self.light1], color=[1, 0.2, 0.2], parentItem=self)
        self.torque_plot.setVisible(False)
        self.force_plot.setVisible(False)
    
    def toggle_ft_plot_visible(self, show: bool):
        self.torque_plot.setVisible(show, True)
        self.force_plot.setVisible(show, True)

    def set_image(self, img):
        """
        将图像设置为 gel 片体的纹理

        Parameters:
        - img : np.ndarray, 图像
        """
        if img is None:
            self._gel_material.textures = []
            return
        if len(self._gel_material.textures) == 0:
            self._gel_material.textures.append(self._gel_texture)
        self._gel_texture.setTexture(img)

    def set_depth(self, depth: np.ndarray):
        """
        设置深度图, 深度图方向为从外朝里看, 深度值负数为按压, 正数为凸起
        """
        depth = np.flip(-depth/6, 1)
        depth = np.minimum(depth, 0)
        self._gel.setData(depth, self._smooth_iter)

    def set_force(self, force, torque):
        center = np.array([0, 1.5, 0.2])
        # tfs, lens = direction_matrixs([0, 0, 0], [0, 0, torque[2]])
            
        f_len = np.linalg.norm(force)
        t_len = abs(torque[2]/5) # np.linalg.norm(torque)

        if torque[2] < 0:
            tf = Matrix4x4()
        else:
            tf = Matrix4x4.fromAxisAndAngle(0, 1, 0, 180)
            
        f_color = cm.yellow_red(f_len / 10)
        t_color = cm.yellow_red(t_len / 10)

        force[2] = -force[2]
        # set force 
        if not self.force_plot.visible():
            self.force_plot.setVisible(True)
        if not self.torque_plot.visible():
            self.torque_plot.setVisible(True)
        self.force_plot.setData(center, center + force/10, color=f_color)
        self.torque_plot.setTransform((tf*Matrix4x4.fromAxisAndAngle(0, 1, 0, 90)).moveto(*center))
        self.torque_plot.setData(t_len / 10, color=t_color)

    def set_3d_arrow(self, start_pos=None, end_pos=None, color=None):
        """
        设置 3d 箭头
        """
        self.arrow_plot.setData(start_pos, end_pos, color)
    
    def get_3d_marker(self, grid: np.ndarray) -> np.ndarray:
        """
        将标记点转换为 3d 坐标

        Parameters:
        - grid : np.ndarray of shape(nrows, ncols, 2), 标记点位置, 图像坐标系的 px, py, 0~1, py 相当于 t, px 平行于 y

        Returns:
        - grid_vert : np.ndarray with the same shape of grid, 标记点 3d 坐标 
        """
        # nrows, ncols = grid.shape[0:2]
        shape = list(grid.shape)
        shape[-1] = 3
        grid_vert = np.zeros(shape, dtype='f4')
        grid_vert[..., 0] = (grid[..., 0] - 0.5) * self.width
        grid_vert[..., 1] = (1 - grid[..., 1]) * self.height

        return grid_vert  # (nrows, ncols, 3)
    
    def set_3d_marker(self, marker: np.ndarray):
        """
        设置标记点可视化

        Parameters:
        - marker : np.ndarray of shape(..., 3), 标记点 3d 坐标
        """
        self.marker.setData(marker.reshape(-1, 3))

    def create_depth_cam(self, view: GLViewWidget, img_size=(192, 336)):
        """
        获取和传感器视野对齐的深度相机, 该相机位置和 gel 绑定, 从内向外看
        """
        depth_camera = DepthCamera(
            view, img_size, eye=(0, self.height/2, 0), center=(0, self.height/2, 1), up=(0, 1, 0),
            ortho_space=(-self.width/2, self.width/2, -self.height/2, self.height/2, -1, 1), 
            frustum_visible=False, actual_depth=True            
        )
        view.item_group.removeItem(depth_camera)

        self._gel.addChildItem(depth_camera)
        
        return depth_camera