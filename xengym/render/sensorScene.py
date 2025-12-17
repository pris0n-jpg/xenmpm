import numpy as np
from pathlib import Path
from typing import Sequence
import cv2

from xensesdk.ezgl.items.scene import Scene
from xensesdk.ezgl.experimental.GLSurfMeshItem import GLSurfMeshItem
from xensesdk.ezgl.items import *
from xensesdk.ezgl import Matrix4x4

from ..fem.simulation import FEMSimulator
from .. import ASSET_DIR


def gen_texcoords(n_row, n_col, u_range=(0, 1), v_range=(0, 1)):
    tex_u = np.linspace(*u_range, n_col)
    tex_v = np.linspace(*v_range, n_row)
    return np.stack(np.meshgrid(tex_u, tex_v), axis=-1).reshape(-1, 2)


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

void main() {
    TexCoords = aTexCoords;
    gl_Position = proj * view * model * vec4(aPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

struct Material {
    bool disable;  // 禁用材质时使用 oColor
    float opacity;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    bool use_texture;
    sampler2D tex_diffuse;
};
uniform Material material;

void main() {
    vec3 result = vec3(texture(material.tex_diffuse, TexCoords));
    FragColor = vec4(result, 1);
}
"""

class MarkerTextureCamera(RGBCamera):

    def init(self, mesh: GLMeshItem):
        self.mesh = mesh
        with self.view():
            self.init_fbo()
            self.shader = Shader(vertex_shader, fragment_shader)
            self._fbo.rgb_texture.type = "tex_diffuse"

    @property
    def texture(self):
        return self._fbo.rgb_texture

    def render(self):
        if self.view() is None:
            raise ValueError("view is None")

        with self.view():
            with self._fbo:
                if not self.mesh.isInitialized:
                    self.mesh.initialize()
                self.mesh.update_model_matrix()

                gl.glDepthFunc(gl.GL_ALWAYS)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT )
                self.mesh._mesh._material.set_uniform(self.shader, "material")
                self.mesh.paintWithShader(self, self.shader)
                gl.glDepthFunc(gl.GL_LESS)

    def get_texture_np(self):
        with self.view():
            return self._fbo.rgb_texture.getTexture()



class SensorScene(Scene):

    def __init__(
        self,
        fem_file,
        depth_size: Sequence,
        gel_size_mm: Sequence,
        marker_row_col: Sequence = (20, 11),
        # marker_dx_dy_mm: Sequence = (1.6, 1.45),
        marker_dx_dy_mm: Sequence = (1.31, 1.31),
        visible: bool = False,
        parent = None,
        title: str = "Sim Scene",
    ):
        """
        仿真场景

        Parameters:
        - fem_file : str or Path, FEM 文件,
        - depth_size : tuple, (width, height), 深度图尺寸
        - gel_size_mm: tuple, (width, height), mm, 硅胶尺寸
        - visible : bool, optional, default: False
        - parent : Scene, optional
        - title : str, optional, default: "FEM Scene"
        """
        super().__init__(win_height=630, win_width=375, visible=visible, parent=parent, title=title)
        self.align_view()
        self.axis = GLAxisItem(size=(0.5, 0.5, 0.5), tip_size=0.08).translate(-2.1, -3.4, 0)

        # ==== gel size ====
        self.gel_width_mm = gel_size_mm[0]
        self.gel_height_mm = gel_size_mm[1]
        self.fem_sim = FEMSimulator(
            self, fem_file,
            marker_row_col,
            marker_dx_dy_mm,
            depth_size = depth_size,
            gel_size_mm = gel_size_mm
        )
        scale_ratio = 4 / self.gel_width_mm
        base_tf = (Matrix4x4.fromScale(scale_ratio, scale_ratio, scale_ratio)
                            .translate(0, -self.gel_height_mm/2, 0, True)
                            .rotate(180, 0, 1, 0, True))

        # ==== lights =====
        self.light_white = PointLight(
            pos=(0, 0, 1), ambient=(0.1, 0.1, 0.1), diffuse=(0.1, 0.1, 0.1), specular=(0,0,0),
            visible=True, directional=False, render_shadow=False
        )

        self.light_r = LineLight(
            pos = np.array([2, -3.0, 1.5]), pos2=np.array([2, 3.0, 1.3]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )

        self.light_g = LineLight(
            pos = np.array([-2, -3.2, 1.5]), pos2=np.array([-2, 3.2, 1.3]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )

        self.light_b = LineLight(
            pos = np.array([-2, -3.2, 1]), pos2=np.array([2, -3.2, 1]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )

        self.light_b2 = LineLight(
            pos = np.array([-1.7, 3.2, 1]), pos2=np.array([1.7, 3.2, 1]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )
        lights = [self.light_white, self.light_r, self.light_g, self.light_b, self.light_b2]
        self.loadLight(str(ASSET_DIR/"data/light.txt"))

        # ==== fem texture ====
        init_marker = (self.fem_sim.depth_to_gel * self.fem_sim.get_marker().reshape(-1, 3))[:, :2]  # depth 图片中的 marker
        self.fem_mode_tex_np = self.make_marker_texture(init_marker, depth_size, tex_size=(320, 560), marker_radius=3)
        self.white_tex_np = np.full_like(self.fem_mode_tex_np, 255, dtype=np.uint8)
        self.fem_mode_tex = Texture2D(self.fem_mode_tex_np)

        # ==== fem mesh ====
        n_row, n_col = self.fem_sim.mesh_shape
        self.vis_fem_mesh = GLMeshItem(
            indices=self.fem_sim.top_indices,
            texcoords=gen_texcoords(n_row, n_col, u_range=(1, 0)),
            lights=lights,
            material=Material(ambient=(1,1,1), diffuse=(1,1,1), specular=(1,1,1), textures=[self.fem_mode_tex]),
            calc_normals=False,
            usage=gl.GL_DYNAMIC_DRAW,
        ).applyTransform(base_tf)
        self.vis_fem_mesh.setVisible(False)

        # ==== depth texture ====
        # depth 渲染模式下, 先用fem渲染出纹理, 然后把纹理 map 到深度图上
        self.depth_mode_tex_camera = MarkerTextureCamera(
            self, img_size=(320, 560), eye=(0, 0, 1), up=(0, 1, 0),
            ortho_space=(-self.gel_width_mm/2*scale_ratio, self.gel_width_mm/2*scale_ratio,
                         -self.gel_height_mm/2*scale_ratio, self.gel_height_mm/2*scale_ratio, 0, 10)
        )
        self.depth_mode_tex_camera.init(self.vis_fem_mesh)

        # ==== depth mesh =====
        self.vis_depth_mesh = GLSurfMeshItem(
            (140, 80),
            x_range=(self.gel_width_mm/2, -self.gel_width_mm/2),
            y_range=(self.gel_height_mm, 0),
            lights = lights,
            material= Material(ambient=(1,1,1), diffuse=(1,1,1), specular=(1,1,1), textures=[self.depth_mode_tex_camera.texture]),
        )
        self.vis_depth_mesh.applyTransform(base_tf)
        self.vis_depth_mesh.mesh_item.setData(texcoords=gen_texcoords(140, 80, v_range=(1, 0)))

        # ==== marker, contact node, force ====
        self.vis_contact_node = GLScatterPlotItem(None, color=(0,0,1), size=0.6, glOptions="ontop").applyTransform(base_tf)
        self.vis_force = GLArrowPlotItem(None, None, color=(0,1,0,1), glOptions="opaque").applyTransform(base_tf)
        self.vis_contact_node.setDepthValue(1)

        # ==== rgb camera ====
        w = (self.gel_width_mm-1) / 2 * scale_ratio  # NOTE: zoom in, 避免有限元边缘出现异常
        h = (self.gel_height_mm-1) / 2 * scale_ratio
        self.sim_camera = RGBCamera(
            self, img_size=(400, 700), eye=(0, 0, 10*scale_ratio), up=(0, 1, 0),
            ortho_space=(-w,w,-h,h,0,10),
            frustum_visible=False
        )
        self.sim_camera.render_group.update(self.vis_depth_mesh)
        
        # 添加深度相机用于获取vis_depth_mesh的深度信息
        self.depth_camera = DepthCamera(
            self, img_size=(400, 700), eye=(0, 0, 10*scale_ratio), up=(0, 1, 0),
            ortho_space=(-w,w,-h,h,0,10),
            frustum_visible=False, actual_depth=True
        )
        self.depth_camera.render_group.update(self.vis_depth_mesh)
        self.depth_offset = self.depth_camera.render()

        # ==== render options ====
        self._enable_smooth = True
        self._show_marker = True
        self._show_contact = False
        self._show_force = False
        self._show_fem_mesh = False

        # ==== get ref image ====
        self.set_show_marker(False)
        self.ref_image = self.get_image()
        self.set_show_marker(True)

    def set_friction_coefficient(self, mu: float):
        self.fem_sim.fric_coef = mu
        
    def make_marker_texture(self, src_marker, src_size, tex_size, marker_radius=3):
        """
        创建表面的 Marker 纹理

        Parameters:
        - src_marker : array-like, (n, 2{x, y}), marker 像素坐标
        - src_size : tuple, (width, height), 定义 src_marker 的坐标范围
        - tex_size : tuple, optional, 要生成的纹理尺寸
        - marker_radius : int, optional, default: 3, marker 大小
        """
        marker = src_marker * (np.array(tex_size) / np.array(src_size))
        tex = np.full((tex_size[1], tex_size[0], 3), 255, dtype=np.uint8)

        for x, y in marker:
            x, y = int(x), int(y)
            cv2.ellipse(tex, (x, y), (marker_radius, marker_radius), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
        return tex

    def align_view(self):
        """调整视角对齐传感器中心"""
        self.cameraLookAt([0, 0, 8.15], [0, 0, 0], [0, 1, 0])

    def step(self, obj_pose: Matrix4x4, sensor_pose: Matrix4x4, depth: np.ndarray):
        if depth is None or depth.size == 0:
            return
        self.fem_sim.step(obj_pose, sensor_pose, depth)

    def enable_smooth(self, enable: bool):
        self._enable_smooth = enable

    def set_show_marker(self, show: bool):
        self._show_marker = show
        if show:
            self.fem_mode_tex.setTexture(self.fem_mode_tex_np)
        else:
            self.fem_mode_tex.setTexture(self.white_tex_np)

    def set_show_contact(self, show: bool):
        self._show_contact = show
        self.vis_contact_node.setVisible(show)

    def set_show_force(self, show: bool):
        self._show_force = show
        self.vis_force.setVisible(show)

    def set_show_fem_mesh(self, show: bool):
        """
        设置使用有限元网格渲染曲面 / 使用深度图渲染曲面
        由于有限元网格和深度图网格的 indices 不同, 所以需要切换

        Parameters:
        - show : bool, 是否显示有限元网格
        """
        self._show_fem_mesh = show
        if show:
            self.sim_camera.render_group.update(self.vis_fem_mesh)
            self.vis_depth_mesh.setVisible(False, True)
            self.vis_fem_mesh.setVisible(True, True)
        else:
            self.sim_camera.render_group.update(self.vis_depth_mesh)
            self.vis_fem_mesh.setVisible(False, True)
            self.vis_depth_mesh.setVisible(True, True)

    def update_mesh_data(self):
        """
        更新网格可视化数据
        """
        vertex, normal = self.fem_sim.get_mesh_data(self._enable_smooth)
        self.vis_fem_mesh.setData(vertex, -normal)

        if not self._show_fem_mesh:
            self.depth_mode_tex_camera.render()
            # qtcv.imshow("depth", self.depth_mode_tex_camera.get_texture_np()*255)  # DEBUG

            depth = self.fem_sim.get_depth()
            depth = np.minimum(depth, 0)
            self.vis_depth_mesh.setData(depth, self._enable_smooth)

    def get_height(self):
        """
        获取传感器表面形变场（深度场形式）
        通过深度相机捕获self.vis_depth_mesh的深度信息
        Returns:
        - np.ndarray of shape(700, 400), float32
        """
        # 确保mesh数据是最新的
        self.update_mesh_data()
        
        # 使用深度相机获取vis_depth_mesh的深度信息
        depth_field = self.depth_camera.render() - self.depth_offset
        
        return depth_field.astype(np.float32) * 1000
        


    def update(self):
        self.update_mesh_data()
        if self._show_contact:
            self.vis_contact_node.setData(self.fem_sim.contact_vert)
        if self._show_force:
            self.vis_force.setData(self.fem_sim.contact_vert, self.fem_sim.contact_vert + self.fem_sim.contact_force*10)
        super().update()

    def get_image(self):
        """
        渲染传感器图像

        Returns:
        - np.ndarray of shape(n, m, 3), uint8
        """
        self.update_mesh_data()
        return (self.sim_camera.render() * 255).astype(np.uint8)
    
    def get_force(self):
        """
        获取节点上的三维力

        Returns:
        - np.ndarray of shape(n, m, 3), float32
        """
        return self.fem_sim.top_force
    
    def get_force_xyz(self):
        """
        获取三维力

        Returns:
        - np.ndarray of shape(3), float32
        """
        return -np.sum(self.fem_sim.contact_force, axis = 0)
    
    def get_diff_image(self):
        self.set_show_marker(False)
        cur_image = self.get_image()
        self.set_show_marker(True)
        return np.clip((cur_image - self.ref_image) + 110 , 0, 255).astype(np.uint8)

    def get_depth(self):
        """
        返回上一次 step 使用的深度图, >0 表示侵入

        Returns:
        - np.ndarray, float32, 单位 mm
        """
        depth = self.fem_sim.get_depth()
        return -np.minimum(depth, 0)

    def get_marker(self, center=True):
        """
        获取 Marker 位移

        Returns:
        - np.ndarray, shape=(20, 11, 3), dtype=np.float32
        """
        marker = self.fem_sim.get_marker()
        if center:
            marker[:,:,1] = marker[:,:,1] - self.fem_sim.marker_offset
        return marker
    
    def get_marker_displacement(self, toTrue=True):
        """
        获取 Marker 位移
        """
        # if toTrue:
        #     marker_displacement = self.fem_sim.get_marker_displacement()
        #     M = np.diag([-1, 1, -1]).astype(marker_displacement.dtype)
        #     return marker_displacement @ M.T
        # else:
        #     return self.fem_sim.get_marker_displacement()
        return self.fem_sim.get_marker_displacement()

    def get_distance(self):
        """
        获取传感器到物体的最小距离, 负数表示侵入深度

        Returns:
        - np.float32, 传感器表面到物体的最小距离, 单位 m
        """
        depth = self.fem_sim.get_depth()
        return np.min(depth)

    def saveLight(self, file_path: str):
        """
        将光源配置保存到文件
        """
        # 将light 的 dict 保存到文件
        with open(str(file_path), "w") as f:
            f.write(str(self.light_white.toDict()) + "\n")
            f.write(str(self.light_r.toDict()) + "\n")
            f.write(str(self.light_g.toDict()) + "\n")
            f.write(str(self.light_b.toDict()) + "\n")
            f.write(str(self.light_b2.toDict()) + "\n")
        print(f"Saved to {file_path}")

    def loadLight(self, file_path: str):
        """
        从 light.txt 文件加载光源配置
        """
        with open(str(file_path), "r") as f:
            self.light_white.loadDict(eval(f.readline()))
            self.light_r.loadDict(eval(f.readline()))
            self.light_g.loadDict(eval(f.readline()))
            self.light_b.loadDict(eval(f.readline()))
            self.light_b2.loadDict(eval(f.readline()))
        # print(f"Loaded from {file_path}.")def get_height_map(self):