import cv2
import numpy as np
import OpenGL.GL as gl

from ..GLGraphicsItem import GLGraphicsItem
from ..items import Shader, GLMeshItem, Texture2D
from ..items.light import empty_fragment_shader

from . import compute_normals


__all__ = ['GLEllipseItem']


def surface_indices(nrow, ncol):
    """
    生成网格曲面的索引

    Parameters:
    - nrow : int, 顶点行数
    - ncol : int, 顶点列数

    Returns:
    - np.ndarray of shape((nrow-1)*(ncol-1)*2, 3)
    """
    if ncol <= 1 or nrow <= 1:
        raise Exception("cols or rows must be greater than 1")
    nrow, ncol = nrow-1, ncol-1
    faces = np.empty((nrow, 2, ncol, 3), dtype=np.int32)
    rowtemplate1 = np.arange(ncol).reshape(1, ncol, 1) + np.array([[[0     , ncol+1, 1]]])  # 1, ncols, 3
    rowtemplate2 = np.arange(ncol).reshape(1, ncol, 1) + np.array([[[ncol+1, ncol+2, 1]]])
    rowbase = np.arange(nrow).reshape(nrow, 1, 1) * (ncol+1)  # nrows, 1, 1
    faces[:, 0] = (rowtemplate1 + rowbase)  # nrows, 1, ncols, 3
    faces[:, 1] = (rowtemplate2 + rowbase)
    return faces.reshape(-1, 3)


def surface_vertexes(zmap, x_range, y_range, nrow=100, ncol=100):
    zmap = cv2.resize(zmap, (ncol, nrow))
    h, w = zmap.shape

    x = np.linspace(x_range[0], x_range[1], w, dtype='f4')  # 从左到右 x 递增
    y = np.linspace(y_range[0], y_range[1], h, dtype='f4')

    xgrid, ygrid = np.meshgrid(x, y, indexing='xy')
    verts = np.stack([xgrid, ygrid, zmap.astype('f4')], axis=-1).reshape(-1, 3)

    return verts


def ellipse_param_func(points, a, b):
    """
    计算椭圆参数化函数

    Parameters:
    - points : np.ndarray, float32, shape=(n, 3), 输入点数组
    - a : float, 椭圆参数 a
    - b : float, 椭圆参数 b

    Returns:
    - points : np.ndarray, float32, shape=(n, 3), 输出点数组
    """
    ret = np.zeros_like(points, dtype=np.float32)
    phi = points[:, 0]
    theta = points[:, 1]
    t = points[:, 2]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    delta = np.sqrt(a * a * sin_theta * sin_theta + b * b * cos_theta * cos_theta)
    v1 = b + a * t / delta

    ret[:, 0] = -v1 * sin_theta * np.cos(phi)
    ret[:, 1] = (a + b * t / delta) * cos_theta
    ret[:, 2] = v1 * sin_theta * np.sin(phi)

    return ret

def make_neighbors(n_vert, indices, n_neigh=6):
    """
    生成邻居节点索引

    Parameters:
    - n_vert : int, 顶点数
    - indices : np.narray of shape(n, 3), 三角形索引
    - n_neigh : int, optional, default: 6, 每个节点的邻居数

    Returns:
    - _type_, _description_
    """
    neighbors = [[] for _ in range(n_vert)]

    for quad in indices:
        for i in range(3):
            neighbors[quad[i]].extend([quad[(i+1)%3], quad[(i+2)%3]])

    # 去重邻居顶点
    for i in range(n_vert):
        neigh = list(set(neighbors[i]))
        if len(neigh) > n_neigh:
            neigh = neigh[:n_neigh]
        elif len(neigh) < 6:
            neigh = [i] * 6  # 边缘点为自身

        neighbors[i] = neigh
    return np.array(neighbors, np.int32)


def smooth(neighbors, vert_wise_data):
    """
    对顶点数据进行平滑, 每个节点的平滑后的数据为其邻居节点数据的平均值

    Parameters:
    - neighbors : np.ndarray of shape(n, m), n 为节点数, m 为每个节点的邻居数
    - vert_wise_data : np.ndarray of shape(n, k), k 为每个节点的数据维度
    """
    # n*3 -> 1*n*3, n*4 -> n*4*1 --> n*4*3
    extracted_normals = np.take_along_axis(vert_wise_data[None, ...], neighbors[..., None], axis=1)
    new_data = np.mean(extracted_normals, axis=1, keepdims=False)
    return new_data


class GLEllipseItem(GLGraphicsItem):

    def __init__(
        self,
        ellipse_size = (2., 3.),
        phi_range = (0, np.pi),  # -x 轴为 0
        theta_range = (0, np.pi),
        grid_size = (300, 400),
        init_zmap = None,
        lights = list(),
        material = None,
        show_edge = False,
        glOptions='opaque',
        parentItem=None
    ):
        """
        椭球曲面

        Parameters:
        - ellipse_size : tuple, optional, default: (2., 3.), b, a, 椭圆短轴, 长轴, 分别沿 x, y 轴
        - phi_range : tuple, optional, default: (0, np.pi), 经度
        - theta_range : tuple, optional, default: (0, np.pi), 维度
        - grid_size : tuple, optional, default: (300, 400), 曲面网格数量, 也即对应的深度图 size: (ncol, nrow)
        - init_zmap : np.ndarray, optional, default: None, 0 按压深度表面对应的椭球坐标系深度图, None 表示全0
        - lights : list, optional, default: list()
        - material : Material, optional, default: None, 材质
        - show_edge : bool, optional, default: False, 是否显示边
        - glOptions : str, optional, default: 'opaque'
        - parentItem : GLGraphicsItem, optional, default: None
        """
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self._texture = None
        self._grid_size = grid_size
        self._init_zmap: np.ndarray = np.zeros((self._grid_size[1], self._grid_size[0]), dtype=np.float32) if init_zmap is None else init_zmap
        self._indices = surface_indices(self._grid_size[1], self._grid_size[0])
        self.setGeometry(ellipse_size, phi_range, theta_range, None)

        self._mesh_item = GLMeshItem(
            vertexes=self._init_vertices,
            normals=self._init_normals,
            indices=self._indices,
            lights=lights,
            material=material,
            calc_normals=False,
            show_edge=show_edge,
        )
        self.addChildItem(self._mesh_item)

    def setGeometry(self, ellipse_size=None, phi_range=None, theta_range=None, init_zmap=None):
        """
        定义椭球坐标系的几何参数, 更新 _init_vertices, _init_normals
        """
        if ellipse_size is not None:
            self._b = ellipse_size[0]
            self._a = ellipse_size[1]
            self._t_range = (-self._b, self._b)
            self._y_range = (-self._a - self._b, self._a + self._b)
            self._r_range = (0, self._b + self._b)  # r = sqrt(x^2 + z^2)
        if phi_range is not None:
            self._phi_range = phi_range
        if theta_range is not None:
            self._theta_range = theta_range
        if self._texture is not None:
            self._texture.setTexture(self.get_texture())
        if init_zmap is not None:
            self._init_zmap = init_zmap

        # update _init_vertices
        _vertices = surface_vertexes(self._init_zmap, self._phi_range, self._theta_range, self._grid_size[1], self._grid_size[0])
        self._init_vertices = ellipse_param_func(_vertices, self._a, self._b)
        self._init_normals = compute_normals(self._init_vertices, self._indices)
        self._neighbors = make_neighbors(len(self._init_vertices), self._indices)

    def setData(self, depth_map=None):
        """
        设置深度图

        Parameters:
        - depth_map : np.ndarray, optional, default: None, 从外朝内看的相对深度图, 按压区域相对深度 <0, 默认为 None 时, 为 0 深度
        """
        if depth_map is not None:
            _depth_map = cv2.resize(depth_map, (self._grid_size[0], self._grid_size[1]))
        else:
            _depth_map = np.zeros_like(self._init_zmap)

        vertices = self._init_vertices + self._init_normals * _depth_map.reshape(-1, 1)
        normals = compute_normals(vertices, self._indices)
        vertices = smooth(self._neighbors, vertices)
        normals = smooth(self._neighbors, normals)
        self._mesh_item._mesh._vertexes.set_data(vertices)
        self._mesh_item._mesh._normals.set_data(normals)

    def get_texture(self):
        """
        获得坐标变换的纹理 t, theta = tex(r, y)
        在 t_range 和 theta_range 范围内生成均匀的参数点, 计算对应的 r, y 坐标, 更新 tex 中对应 r (or x), y 的值
        """
        sample_shape = (1000, 1000)
        t_span = np.linspace(self._t_range[0], self._t_range[1]*2, sample_shape[0], dtype='f4')
        theta_span = np.linspace(0, np.pi, sample_shape[1], dtype='f4')
        params = np.stack(np.meshgrid(theta_span, t_span, indexing='xy'), axis=2).reshape(-1, 2)

        # 纹理的尺寸， width 对应 x~(0, b + b^2/a), height 对应 y~(-a - b^2/a, a + b^2/a)
        tex_shape = (200, 400)
        tex = np.zeros([*tex_shape, 2], dtype='f4')
        tex_count = np.zeros(tex_shape, dtype='i4')

        # 计算纹理坐标
        x, y = self.param_func_2d(params[:,1], params[:, 0], self._a, self._b)
        x_i = np.round((x - self._r_range[0]) / (self._r_range[1] - self._r_range[0]) * (tex_shape[0]-1)).astype('i4')
        y_i = np.round((y - self._y_range[0]) / (self._y_range[1] - self._y_range[0]) * (tex_shape[1]-1)).astype('i4')
        mask = (x_i >= 0) & (x_i < tex_shape[0]) & (y_i >= 0) & (y_i < tex_shape[1])
        x_i, y_i = x_i[mask], y_i[mask]
        np.add.at(tex_count, (x_i, y_i), 1)
        np.add.at(tex, (x_i, y_i), params[mask])
        tex /= np.maximum(tex_count, 1)[..., None]

        # 无效值填充, t 设置为最大值, 也就是距离远的点, 将被深度检测剔除
        # tex[tex_count == 0] = [100, 100]

        # 滤波
        tex = cv2.GaussianBlur(tex, (5, 5), 11, tex)
        return tex

    def param_func_2d(self, t, theta, a, b):

        """
        平面等距椭圆曲线参数方程

        Parameters:
        - t : float, 法向高度
        - theta : float, 维度参数
        - a : float, 长轴半径, 默认沿 y 轴
        - b : float, 短轴半径, 默认沿 x 轴

        Returns:
        - list[float], (x, y)
        """
        cos_th, sin_th = np.cos(theta), np.sin(theta)
        v1 = t / np.sqrt(a**2 * sin_th**2 + b**2 * cos_th**2)
        return [(b + a*v1)*sin_th, (a + b*v1)*cos_th]

    def initializeGL(self):
        self.ellipse_shader = Shader(xyz_to_param_shader_str, empty_fragment_shader)
        self._texture = Texture2D(
            self.get_texture(),
            flip_x=False, flip_y=False,
            wrap_s=gl.GL_CLAMP_TO_EDGE , wrap_t=gl.GL_CLAMP_TO_EDGE ,
        )
        self._texture.bind()
        # gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, [np.pi, 0])

    def updateGL(self):
        # 椭圆切面纹理对应的 xy 范围
        with self.ellipse_shader:
            self.ellipse_shader.set_uniform("params_tex", self._texture.bindTexUnit(), "sampler2D")
            self.ellipse_shader.set_uniform("r_range", self._r_range, "vec2")
            self.ellipse_shader.set_uniform("y_range", self._y_range, "vec2")
            # theta, t , 用于映射到 ndc 坐标系
            self.ellipse_shader.set_uniform("phi_range", self._phi_range, "vec2")
            self.ellipse_shader.set_uniform("theta_range", self._theta_range, "vec2")
            self.ellipse_shader.set_uniform("t_range", self._t_range, "vec2")

    def paint(self, camera):
        self.updateGL()

    def paintWithShader(self, camera, shader, **kwargs):
        self.updateGL()

    def convert_depth(self, depth):
        """
        将 0~1 深度转化为对于 _init_zmap 的相对高度, 高于 _init_zmap 的点设置为 0

        Parameters:
        - depth : np.ndarray, 0~1 深度

        Return:
        - np.ndarray, 相对高度, <= 0
        """
        depth = depth * (self._t_range[1] - self._t_range[0]) + self._t_range[0]  # 先转化到椭球坐标系
        if not hasattr(self, '_init_zmap_resized') or self._init_zmap_resized.shape != depth.shape:
            self._init_zmap_resized = cv2.resize(self._init_zmap, (depth.shape[1], depth.shape[0]))
        depth = depth - self._init_zmap_resized
        depth[depth > 0] = 0
        return depth



# 使用这个 shader 的相机坐标系需要与椭球曲面的坐标系一致, 渲染得到从内朝外看的深度图
xyz_to_param_shader_str = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

uniform sampler2D params_tex;
uniform vec2 r_range;
uniform vec2 y_range;
uniform vec2 phi_range;
uniform vec2 t_range;
uniform vec2 theta_range;


void main()
{
    vec4 pos = view * model * vec4(aPos, 1.0f);

    float r = sqrt(pos.x * pos.x + pos.z * pos.z);
    vec2 uv = vec2(
        (pos.y - y_range.x) / (y_range.y - y_range.x),
        (r - r_range.x) / (r_range.y - r_range.x)
    );

    vec2 params = texture(params_tex, uv).xy;
    float phi = atan(pos.z, pos.x);  // x 对应 0
    phi = 3.1415926535 - phi;  // 转成 -x 对应 0
    phi = (phi_range.y - phi) / (phi_range.y - phi_range.x) * 2 - 1;
    float theta = -(params.x - theta_range.x) / (theta_range.y - theta_range.x) * 2 + 1.;
    float t = (params.y - t_range.x) / (t_range.y - t_range.x) * 2 - 1;
    gl_Position = vec4(phi, theta, t, 1.0);
}
"""