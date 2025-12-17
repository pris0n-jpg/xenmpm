import numpy as np
import cv2
import OpenGL.GL as gl
from typing import Sequence

from xensesdk.ezgl.items import GLModelItem, GLMeshItem, Texture2D, Material, Mesh, vertex_normal, GLInstancedMeshItem, sphere, GLArrowMeshItem, PointLight
from xensesdk.ezgl.items.texture import Texture1D
from xensesdk.ezgl.experimental import compute_normals
from xensesdk.ezgl.GLGraphicsItem import GLGraphicsItem
from xensesdk.ezgl.items.GLCircularArrowItem import GLCircularArrowItem, Matrix4x4
from xensesdk.ezgl.items.light import empty_fragment_shader
from xensesdk.ezgl.items.shader import Shader
from xensesdk.ezgl.items.MeshData import fit_3d_spline_curve
from xensesdk.ezgl.utils.colormap import cm

from . import PROJ_DIR
from .utils import make_neighbors, smooth


def surface_indices(nrow, ncol):
    if ncol == 0 or nrow == 0:
        raise Exception("cols or rows is zero")

    faces = np.empty((nrow, 2, ncol, 3), dtype=np.int32)
    rowtemplate1 = np.arange(ncol).reshape(1, ncol, 1) + np.array([[[0     , ncol+1, 1]]])  # 1, ncols, 3
    rowtemplate2 = np.arange(ncol).reshape(1, ncol, 1) + np.array([[[ncol+1, ncol+2, 1]]])
    rowbase = np.arange(nrow).reshape(nrow, 1, 1) * (ncol+1)  # nrows, 1, 1
    faces[:, 0] = (rowtemplate1 + rowbase)  # nrows, 1, ncols, 3
    faces[:, 1] = (rowtemplate2 + rowbase)
    return faces.reshape(-1, 3)


class OmniSensorItem(GLGraphicsItem):

    def __init__(
        self,
        lights: Sequence,
        grid_size=(75, 150),
        depth_range=(-0.7, 0.7),  # unit: cm
        glOptions='opaque',
        parentItem=None,
        scale_factor = 1,
        has1DTexture = True
    ):
        super().__init__(parentItem=parentItem)
        self._grid_size = grid_size  # (cols, rows)
        self._scale_factor = scale_factor
        self._depth_range = np.array( depth_range) / scale_factor
        self._x_range = (1.8 / scale_factor, 0)
        self._z_range = (5/ scale_factor, 1.0/ scale_factor)

        self.sensor_base = GLModelItem(
            PROJ_DIR / "assets/g1-os/vm_g1-os.obj",
            lights = lights,
            glOptions = glOptions,
            parentItem = self,
        )
        self.sensor_base.setPaintOrder([0])
        self.sensor_base.setMaterial(0, Material(ambient=(0.08, 0.08, 0.08), diffuse=(0.15, 0.15, 0.3), specular=(0.3, 0.3, 0.3), shininess=5))

        for mesh in self.sensor_base.meshes:
            mesh._vertexes.set_data( mesh._vertexes.data / scale_factor)
        self._gel_texture = Texture2D()
        self._gel_material = Material(
            ambient=(0.3, 0.3, 0.3), diffuse=(0.9, 0.9, 0.9), specular=(0.1, 0.1, 0.1), shininess=5,
        )

        self._gel_mesh: Mesh = self.sensor_base.meshes.pop()
        self._calc_gel_mesh_grid(self._gel_mesh._vertexes.data / scale_factor)
        self.sensor_gel = GLMeshItem(
            mesh = self._gel_mesh,
            lights=lights,
            glOptions=glOptions,
            show_edge=False,
            parentItem=self,
        )
        self.sensor_gel.setMaterial(self._gel_material)

        verts, faces, uv, norms = sphere(0.02, 8, 8, True)
        self.marker = GLInstancedMeshItem(Mesh(verts, faces, normals=norms), lights=lights, color=(0.3, 0.6, 1.2), glOptions="translucent_cull", parentItem=self)
        self.marker.setDepthValue(2)
        
        self.light1 = PointLight(pos=(5, 10, 10), ambient=(0.7, 0.7, 0.7), diffuse=(1, 1, 1), visible=False, directional=True, render_shadow=False)
        self.arrow_plot = GLArrowMeshItem(tip_size=[0.02, 0.022], width=0.012, lights=[self.light1], color=[1, 0.2, 0.2], parentItem=self)

        self._has1DTexture = has1DTexture
        
        self.torque_plot = GLCircularArrowItem(length=0, inner_radius=0.02, outer_radius=0.25, lights=[self.light1], color=[1, 0.2, 0.2], parentItem=self)
        self.force_plot = GLArrowMeshItem(tip_size=[0.035, 0.02], width=0.02, lights=[self.light1], color=[1, 0.2, 0.2], parentItem=self)
        
        # HACK: 为了可视化分布力, 需要将分布力的法向取反，而切向不变，为此需要计算法向
        idx = np.linspace(0, len(self._right_normals)-1, 35, dtype=np.int32)
        self._force_norm_axis = self._right_normals[idx, None, :]  # (n_rows, 1, 3)
    
    def toggle_ft_plot_visible(self, show: bool):
        self.torque_plot.setVisible(show, True)
        self.force_plot.setVisible(show, True)
        
    def process_force_for_show(self, force):
        # 求解力在法向的投影, force_norm, 然后反向
        # force = force.reshape(35, 20, 3)
        force_norm = np.sum(force * self._force_norm_axis, axis=2, keepdims=True)  # (nrows*ncols, 3) * (26, 1, 3) = (nrows*ncols, 26)
        force = force - 4 * force_norm * self._force_norm_axis
        return force.reshape(35, 20, 3)

    def set_force(self, force, torque):
        # 正中间点的坐标 ([1.0938455 0.8899992 2.865364 ]) cm
        # 正中间点的法向 ([0.9994056  0.   0.03447453]) 
        
        center = np.array([1.09 + 0.5, 0, 2.865])
            
        force[0] = -force[0] * 2
        f_len = np.linalg.norm(force)
        t_len = abs(torque[0]/5) # np.linalg.norm(torque)

        if torque[0] < 0:
            tf = Matrix4x4.fromAxisAndAngle(0, 1, 0, 90)
        else:
            tf = Matrix4x4.fromAxisAndAngle(0, 1, 0, -90)
            
        f_color = cm.yellow_red(f_len / 10)
        t_color = cm.yellow_red(t_len / 10)

        self.force_plot.setData(center, center + force/10, color=f_color)
        self.torque_plot.setTransform((tf*Matrix4x4.fromAxisAndAngle(0, 1, 0, 90)).moveto(*center))
        self.torque_plot.setData(t_len / 10, color=t_color)
        
    def _calc_gel_mesh_grid(self, gel_mesh_vert):
        """
        根据 gel 片体计算网格

        Parameters:
        - gel_mesh_vert : np.ndarray of np.float32, 片体模型顶点
        """
        vert0_right = gel_mesh_vert[gel_mesh_vert[:, 1] > 0.4 / self._scale_factor]  # 取 y > 0 的部分
        vert0_right = vert0_right[np.argsort(-vert0_right[:, 2])]  # 按照 z 坐标排序, 得到右边线, 从下往上的顶点

        # 计算一条平滑的曲线代替原来的折线
        vert0_right = fit_3d_spline_curve(vert0_right, 100)

        # 计算投影在 xz 平面线段长度, 计算前缀和
        xz_segment_len = np.sqrt(np.sum(np.diff(vert0_right[:, [0, 2]], axis=0)**2, axis=1, keepdims=False))
        xz_segment_len = np.insert(np.cumsum(xz_segment_len), 0, 0)

        # 等距划分成 grid_size[1] 份
        n_cols, n_rows = self._grid_size
        t = np.linspace(0, xz_segment_len[-1], n_rows, endpoint=True, dtype='f4', axis=0)
        vert1_right = np.zeros((n_rows, 3), dtype='f4')
        for i in range(3):
            vert1_right[:, i] = np.interp(t, xz_segment_len, vert0_right[:, i], left=vert0_right[0, i], right=vert0_right[-1, i])

        # 构造 3d 网格
        vert1_left = vert1_right.copy()
        vert1_left[:, 1] = -vert1_left[:, 1]
        vertices = np.linspace(vert1_left, vert1_right, n_cols, dtype='f4')  # n_cols, n_rows, 3
        vertices = vertices.transpose(1, 0, 2).reshape(-1, 3)  # n_cols, n_rows, 3 -> n_rows*n_cols, 3
        indices = surface_indices(n_rows-1, n_cols-1)
        normals = compute_normals(vertices, indices)
        self._gel_mesh._vertexes.set_data(vertices)
        self._gel_mesh._indices.set_data(indices)
        self._gel_mesh._normals.set_data(normals)
        self._init_normals = normals
        self._init_vertices = vertices
        self._init_indices = indices
        self._right_vertices = vert1_right  # 右边线的顶点 (n_rows, 3), 从上到下排列
        self._right_normals = normals[0:n_rows*n_cols:n_cols].copy()  # 右边线的法向量 (n_rows, 3)
        self._neighbors = make_neighbors(vertices, indices)

        # 纹理坐标
        uv = np.zeros((n_rows, n_cols, 2), dtype='f4')
        uv[:, :, 0] = np.linspace(1, 0, n_cols)[None, :]
        uv[:, :, 1] = np.linspace(0, 1, n_rows)[:, None]
        self._gel_mesh._texcoords.set_data(uv)

    def get_texture(self):
        """
        获得坐标变换的纹理 t, d = tex(x, z)
        在 t_range 和 depth_range 范围内生成均匀的参数点, 计算对应的 x, z 坐标, 更新 tex 中对应 x, z 的值
        """
        sample_shape = (700, 400)  # (n, m), z, x
        t_pts = np.linspace(0, 1, sample_shape[0], endpoint=True, dtype='f4')
        d_pts = np.linspace(self._depth_range[0]*2, self._depth_range[1]*2, sample_shape[1], endpoint=True, dtype='f4')
        td_grid = np.stack(np.meshgrid(t_pts, d_pts, indexing='ij'), axis=2).reshape(-1, 2)  # (n*m, 2)
        _xp = np.linspace(0, 1, self._right_vertices.shape[0], endpoint=True, dtype='f4')
        x_pts = np.interp(t_pts, _xp, self._right_vertices[:, 0])
        z_pts = np.interp(t_pts, _xp, self._right_vertices[:, 2])
        xz_pts = np.stack([x_pts, z_pts], axis=1)
        nx_pts = np.interp(t_pts, _xp, self._right_normals[:, 0])
        nz_pts = np.interp(t_pts, _xp, self._right_normals[:, 2])
        nxz_pts = np.stack([nx_pts, nz_pts], axis=1)  # (n, 2)

        xz_grid = xz_pts[:, None, :] + nxz_pts[:, None, :] * d_pts[None, :, None]  # (n, 1, 2) + (n, 1, 2) * (1, m, 1) = (n, m, 2)
        xz_grid = xz_grid.reshape(-1, 2)

        # 纹理的尺寸
        tex_shape = (330, 190)
        tex = np.zeros([*tex_shape, 2], dtype='f4')
        tex_count = np.zeros(tex_shape, dtype='i4')

        z_i = np.round((xz_grid[:, 1] - self._z_range[0]) / (self._z_range[1] - self._z_range[0]) * (tex_shape[0]-1)).astype('i4')
        x_i = np.round((xz_grid[:, 0] - self._x_range[0]) / (self._x_range[1] - self._x_range[0]) * (tex_shape[1]-1)).astype('i4')
        mask = (x_i >= 0) & (x_i < tex_shape[1]) & (z_i >= 0) & (z_i < tex_shape[0])
        x_i, z_i = x_i[mask], z_i[mask]
        np.add.at(tex_count, (z_i, x_i), 1)
        np.add.at(tex, (z_i, x_i), td_grid[mask])
        tex /= np.maximum(tex_count, 1)[..., None]

        # 边缘无效值填充
        tex[tex_count == 0] = [1, 0]
        tex[:100][tex_count[:100] == 0] = [0, 0]  # 上半区 t 边缘值为 0

        # 滤波
        tex = cv2.GaussianBlur(tex, (5, 5), 11, tex)

        # y 纹理, 用于将 y 映射到 ndc 坐标系
        y_tex = np.interp(t_pts, _xp, self._right_vertices[:, 1]).astype('f4')
        return tex, y_tex

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

    def contact_region(self, img):
        """
        接触区域检测, 接触区域为 -0.1, 其他区域为 0

        Parameters:
        - img : np.ndarray of shape(n, m, 3), 传感器 rgb 图像
        """
        low_bound = np.array([110, 110, 110], dtype=np.uint8) - [18, 18, 20]  # rgb
        high_bound = np.array([110, 110, 110], dtype=np.uint8) + [18, 18, 20]
        img = cv2.pyrDown(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        depth = cv2.inRange(img, low_bound, high_bound).astype(np.float32)

        return (depth - 255) / 2550

    def set_depth(self, depth: np.ndarray):
        """
        设置深度图, 深度图方向为从外朝里看, 深度值负数为按压, 正数为凸起
        """
        depth = np.flip(-depth/6, 1)
        depth = cv2.resize(depth, (self._grid_size[0], self._grid_size[1]))
        vertices = self._init_vertices + self._init_normals * depth.reshape(-1, 1)
        normals = compute_normals(vertices, self._init_indices)
        vertices = smooth(self._neighbors, vertices)
        normals = smooth(self._neighbors, normals)
        self._gel_mesh._vertexes.set_data(vertices)
        self._gel_mesh._normals.set_data(normals)
    
    def set_3d_arrow(self, start_pos=None, end_pos=None, color=None):
        """
        设置 3d 箭头
        """
        self.arrow_plot.setData(start_pos, end_pos, color)

    def set_3d_marker(self, marker: np.ndarray):
        """
        设置标记点可视化

        Parameters:
        - grid : np.ndarray of shape(nrows, ncols, 3), 标记点 3d 坐标
        """
        self.marker.setData(marker.reshape(-1, 3))

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
        # 根据 py 插值
        grid_vert = np.zeros(shape, dtype='f4')
        _xp = np.linspace(0, 1, self._right_vertices.shape[0], endpoint=True, dtype='f4')
        for i in range(3):  # x, y, z
            grid_vert[..., i] = np.interp(grid[..., 1], _xp, self._right_vertices[:, i], left=self._right_vertices[0, i], right=self._right_vertices[-1, i])

        # 根据 px 缩放 y
        grid_vert[..., 1] = grid_vert[..., 1] * (2 * grid[..., 0] - 1)
        return grid_vert  # (nrows, ncols, 3)

    def initializeGL(self):
        return
        self._shader = Shader(xyz_to_param_shader_str, empty_fragment_shader)
        t_depth_tex, y_tex = self.get_texture()
        self._t_depth_tex = Texture2D(
            t_depth_tex, flip_x=False, flip_y=False,
            wrap_s=gl.GL_CLAMP_TO_EDGE , wrap_t=gl.GL_CLAMP_TO_EDGE ,
        )
        self._y_tex = Texture1D(y_tex, flip=False, wrap_s=gl.GL_CLAMP_TO_EDGE )

    def updateGL(self):
        with self._shader:
            self._shader.set_uniform("t_depth_tex", self._t_depth_tex.bindTexUnit(), "sampler2D")
            self._shader.set_uniform("x_range", self._x_range, "vec2")
            self._shader.set_uniform("z_range", self._z_range, "vec2")
            # 用于映射到 ndc 坐标系
            if self._has1DTexture:
                self._shader.set_uniform("y_tex", self._y_tex.bindTexUnit(), "sampler1D")
            self._shader.set_uniform("depth_range", self._depth_range, "vec2")

    def paint(self, camera):
        return
        self.updateGL()

    def paintWithShader(self, camera, shader, **kwargs):
        self.updateGL()

    def convert_depth(self, depth, base_depth=0):
        """
        将 0~1 深度转化为实际深度, 高于 base_depth 的点设置为 base_depth

        Parameters:
        - depth : np.ndarray, 0~1 深度
        - base_depth : int or np.ndarray, optional, default: 0, 基准深度
        """
        depth = depth * (self._depth_range[1] - self._depth_range[0]) + self._depth_range[0]
        mask = depth > base_depth  # 高于基准深度的点无效, 因为只能向内部凹陷
        depth[mask] = base_depth[mask] if isinstance(base_depth, np.ndarray) else base_depth
        return depth


# 使用这个 shader 的相机坐标系需要与模型的坐标系一致
xyz_to_param_shader_str = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

uniform sampler2D t_depth_tex;
uniform sampler1D y_tex;
uniform vec2 depth_range;
uniform vec2 x_range;
uniform vec2 z_range;

void main()
{
    vec4 pos = view * model * vec4(aPos, 1.0f);

    vec2 uv = vec2(
        (pos.x - x_range.x) / (x_range.y - x_range.x),
        (pos.z - z_range.x) / (z_range.y - z_range.x)
    );

    vec2 t_depth = texture(t_depth_tex, uv).xy;
    float y_max = -texture(y_tex, t_depth.x).x;
    float t = - t_depth.x * 2 + 1;
    float depth = (t_depth.y - depth_range.x) / (depth_range.y - depth_range.x) * 2 - 1.;
    float y = pos.y / y_max;

    gl_Position = vec4(y, t, depth, 1.0);
}
"""