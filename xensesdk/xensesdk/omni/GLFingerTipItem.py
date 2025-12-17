import cv2
import numpy as np
from . import PROJ_DIR
from xensesdk.ezgl import GLGraphicsItem, gl
from xensesdk.ezgl.items import GLModelItem, Material, Texture2D, sphere, GLInstancedMeshItem, Mesh, PointLight, GLArrowMeshItem
from xensesdk.ezgl.items.GLCircularArrowItem import GLCircularArrowItem
from xensesdk.ezgl.transform3d import *
from xensesdk.ezgl.experimental.GLEllipseItem import GLEllipseItem
from xensesdk.omni.transforms import Rectify2
from xensesdk.ezgl.utils.colormap import cm


class GLFingerTipItem(GLGraphicsItem):
    """ Displays a GelSlim model with a surface plot on top of it."""

    def __init__(
        self,
        lights: Sequence,
        grid_size: Tuple[int, int] = (120, 160),
        uv_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 1)),
        glOptions: str = "translucent",
        parentItem=None,
    ):
        """
        指尖传感器模型

        Parameters:
        - lights : Sequence, 光源
        - grid_size : Tuple[int, int], optional, default: (120, 160), 网格尺寸, cols, rows
        - uv_range : Tuple, optional, default: ((0.2, 0.8), (0.2, 1)), 纹理坐标对应的 phi, theta 在归一化坐标的范围,
            例如 u_range = (0.5, 1), 表示只有右半边显示纹理
        - glOptions : str, optional, default: "translucent", OpenGL 选项
        - parentItem : GLGraphicsItem, optional, default: None
        """
        super().__init__(parentItem=parentItem)
        self.grid_size = grid_size
        init_gel = np.load(PROJ_DIR / "assets/init_gel_depth.npy")
        self.init_gel = cv2.resize(init_gel.astype(np.float32), grid_size)  # 在椭球坐标系下定义的初始深度
        
        # 添加一个光源照亮另一侧
        light1 = PointLight(pos=(1, -2, 2), ambient=(0., 0., 0.), diffuse=(0.5, 0.5, 0.5), specular=(0, 0, 0),
                               visible=0, directional=True, render_shadow=False)
        lights = list(lights)
        lights.append(light1)

        self.gel = GLEllipseItem(
            ellipse_size=(1.1, 1.8),
            theta_range=(0.0, np.pi/2),
            lights=lights,
            grid_size=grid_size,
            init_zmap=self.init_gel,
            glOptions=glOptions,
            show_edge=False,
            parentItem=self
        )
        self.gel.setDepthValue(1)
        self.gel_mesh_data = self.gel._mesh_item._mesh

        # 设置材质纹理
        self._gel_texture = Texture2D(wrap_s=gl.GL_CLAMP_TO_BORDER, wrap_t=gl.GL_CLAMP_TO_BORDER)
        self._gel_material = Material(
            ambient=(0.3, 0.3, 0.3), diffuse=(0.6, 0.6, 0.6), specular=(0.1, 0.1, 0.1), shininess=4,
        )
        self.gel_mesh_data.setMaterial(self._gel_material)
        self.set_uv_range(uv_range)

        # 外壳模型
        self.base = GLModelItem(
            path = PROJ_DIR / "assets/shell.obj",
            lights = lights,
            glOptions = glOptions,
            parentItem = self,
        ).rotate(90, 0, 0, 1)

        # 图片映射
        self.rectify = Rectify2(
            src_size=(200, 350),
            src_grid=np.array(
                [[[400.000, 153.000], [266.000, 125.000], [134.000, 125.000], [  0.000, 153.000]],
                [[342.000, 272.000], [249.000, 258.000], [151.000, 258.000], [ 58.000, 272.000]],
                [[307.000, 417.000], [231.000, 403.000], [169.000, 403.000], [ 93.000, 417.000]],
                [[287.000, 567.000], [225.000, 558.000], [175.000, 558.000], [113.000, 567.000]],
                [[287.000, 694.000], [228.000, 699.000], [172.000, 699.000], [113.000, 694.000]]]
            ) / 2, 
            dst_size=(200, 350),
            border_value=(110, 110, 110),
            is_inverse=True,
            super_resolution=1
        )
        
        verts, faces, uv, norms = sphere(0.02, 8, 8, True)
        self.marker = GLInstancedMeshItem(Mesh(verts, faces, normals=norms), lights=lights, color=(0.3, 0.6, 1.2), glOptions="translucent_cull", parentItem=self)
        self.marker.setDepthValue(2)
        self.light1 = PointLight(pos=(5, 10, 10), ambient=(0.7, 0.7, 0.7), diffuse=(1, 1, 1), visible=False, directional=True, render_shadow=False)
        self.arrow_plot = GLArrowMeshItem(tip_size=[0.01, 0.011], width=0.006, lights=[self.light1], color=[1, 0.2, 0.2], parentItem=self)

        self.torque_plot = GLCircularArrowItem(length=0, inner_radius=0.02, outer_radius=0.25, lights=[self.light1], color=[1, 0.2, 0.2], parentItem=self)
        self.force_plot = GLArrowMeshItem(tip_size=[0.035, 0.02], width=0.02, lights=[self.light1], color=[1, 0.2, 0.2], parentItem=self)

        # 表面节点的法向量
        norm = self.rectify(self.gel._init_normals.reshape(grid_size[1], grid_size[0], 3), inverse=False)
        norm = cv2.resize(norm, (20, 35))
        norm = np.flip(norm, axis=1)
        self._force_norm_axis = norm
        
    def set_uv_range(self, uv_range):
        u = np.linspace(0, 1, self.grid_size[1])
        v = np.linspace(0, 1, self.grid_size[0])
        u = (u - uv_range[1][0]) / (uv_range[1][1] - uv_range[1][0])
        v = (v - uv_range[0][0]) / (uv_range[0][1] - uv_range[0][0])
        uu, vv = np.meshgrid(u, v, indexing='ij')  # (grid_size[1], grid_size[0])
        uv = np.stack([vv, uu], axis=2).astype('f4')

        # vert = self.gel_mesh_data._vertexes.data.copy()
        # x_max = np.max(vert[:, 0])
        # y_max = np.max(vert[:, 1])
        # v = (vert[:, 1] / y_max - uv_range[1][1]) / (uv_range[1][0] - uv_range[1][1])
        # ## v 越靠上越大
        # y_norm = vert[:, 1] / y_max # 归一化 y 坐标

        # x_max = x_max * ( 1 + y_norm * 0.15)
        # u = ((vert[:, 0] + x_max) / (2 * x_max) - uv_range[0][0]) / (uv_range[0][1] - uv_range[0][0])
        # uv = np.stack([u, v], axis=1).astype('f4')

        self.gel_mesh_data._texcoords.set_data(uv)

    def set_depth(self, depth=None, rectify=True):
        """
        设置深度数据, depth 为从外向内看的相对表面深度

        Parameters:
        - depth : np.ndarray, shape=gel._grid_size[::-1], 渲染得到的 0-1 深度图像, 默认为 init_gel
        - rectify : bool, optional, default: True, 是否使用内部矫正器进行校正
        """
        if depth is None:
            self.gel.setData()
        else:
            depth = - depth / 4
            if rectify:
                depth = self.rectify(depth, border_value=(0))
            self.gel.setData(depth)

    def set_image(self, img, rectify=True):
        """
        将图像设置为 gel 片体的纹理

        Parameters:
        - img : np.ndarray, 图像
        - rectify : bool, optional, default: True, 是否使用内部矫正器进行校正
        """
        if img is None:
            self._gel_material.textures = []
            return
        if len(self._gel_material.textures) == 0:
            self._gel_material.textures.append(self._gel_texture)

        if rectify:
            img = self.rectify(img, border_value=(110, 110, 110))
        self._gel_texture.setTexture(img)

    def set_3d_arrow(self, start_pos=None, end_pos=None, color=None):
        """
        设置 3d 箭头
        """
        self.arrow_plot.setData(start_pos, end_pos, color)
    
    def set_3d_marker(self, marker: np.ndarray):
        """
        设置标记点可视化

        Parameters:
        - marker : np.ndarray of shape(..., 3), 标记点 3d 坐标
        """
        self.marker.setData(marker.reshape(-1, 3))

    def set_force(self, force, torque):
        center = np.array([0, 1, 1.8])  # cm
        # force = force
        f_len = np.linalg.norm(force)
        t_len = abs(torque[0]/5) # np.linalg.norm(torque)

        if torque[2] < 0:
            tf = Matrix4x4.fromAxisAndAngle(0, 1, 0, 90)
        else:
            tf = Matrix4x4.fromAxisAndAngle(0, 1, 0, -90)
            
        f_color = cm.yellow_red(f_len / 15)
        t_color = cm.yellow_red(t_len / 10)

        self.force_plot.setData(center, center + force/15, color=f_color)
        self.torque_plot.setTransform(tf.moveto(*center))
        self.torque_plot.setData(t_len / 10, color=t_color)

    def process_force_for_show(self, force):
        # 求解力在法向的投影, force_norm, 然后反向
        # force = force.reshape(35, 20, 3)
        force_norm = np.sum(force * self._force_norm_axis, axis=2, keepdims=True)  # (nrows*ncols, 3) * (26, 1, 3) = (nrows*ncols, 26)
        force = force - 3 * force_norm * self._force_norm_axis
        return force.reshape(35, 20, 3)

    def convert_depth(self, depth):
        return self.gel.convert_depth(depth)

    def initializeGL(self):
        self._gel_texture.bind()
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, [0.4, 0.4, 0.4])

