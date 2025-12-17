import OpenGL.GL as gl
from typing import Sequence
from .camera import Camera
from .FrameBufferObject import FBO
from .render import RenderGroup
from ..GLViewWidget import GLViewWidget
from .light import shadow_vertex_shader, empty_fragment_shader
from .shader import Shader

__all__ = ['RGBCamera', 'DepthCamera']


class RGBCamera(Camera):

    def __init__(
        self,
        view: GLViewWidget,
        img_size: Sequence = (640, 480),  # width, height
        eye: Sequence = (0., 0., 5),
        center: Sequence = (0., 0., 0),
        up: Sequence = (0., 1., 0),
        aspect = 4/3,
        fov = 45,
        ortho_space: Sequence = (-4, 4, -3, 3, -5, 20),  # left, right, bottom, top, near, far
        proj_type = 'ortho',  # perspective or ortho
        glOptions = 'opaque',
        parentItem = None,
        frustum_visible = False,
    ):
        """
        RGBA 相机

        Parameters:
        - view : GLViewWidget, 相机自动添加视图中所有的渲染对象, 确保在创建相机之前已经向视图添加了所有的渲染对象
        - img_size : Sequence, optional, default: (640, 480), 相机图像大小
        """
        near_far_wrt_distance = False
        super().__init__(eye, center, up, aspect, fov, ortho_space, proj_type, near_far_wrt_distance, glOptions, parentItem, frustum_visible)
        self.img_size = img_size

        self.setView(view)

        # fbo
        self._fbo = None

    def setView(self, view: GLViewWidget):
        if view is None or self.view() == view:
            return

        self.render_group: RenderGroup = view.item_group.copy()
        self.light_group: RenderGroup = view.light_group.copy()
        view.item_group.addItem(self)  # 将相机添加到视图中, self.view() 可访问 view
        super().setView(view)

    def init_fbo(self):
        """
        初始化 fbo
        """
        self._fbo = FBO(width=self.img_size[0], height=self.img_size[1], type=FBO.Type.RGB)

    def render(self, update_shadow=True, shader=None):
        """
        渲染
        """
        if self.view() is None:
            raise ValueError("view is None")

        with self.view():
            # 更新阴影
            if update_shadow:
                for light in self.light_group:
                    light.renderShadow()

            if self._fbo is None:
                self.init_fbo()

            with self._fbo:
                gl.glDepthMask(gl.GL_TRUE)
                gl.glClear( gl.GL_DEPTH_BUFFER_BIT | gl.GL_COLOR_BUFFER_BIT )
                self.render_group.render(camera=self, shader=shader)

            return self._fbo.rgb_texture.getTexture()


class DepthCamera(Camera):

    def __init__(
        self,
        view: GLViewWidget,
        img_size: Sequence = (640, 480),  # width(x), height(y)
        eye: Sequence = (0., 0., 5),
        center: Sequence = (0., 0., 0),
        up: Sequence = (0., 1., 0),
        aspect = 4/3,
        fov = 45,
        ortho_space: Sequence = (-4, 4, -3, 3, -5, 20),  # left, right, bottom, top, near, far
        proj_type = 'ortho',  # perspective or ortho
        glOptions = 'opaque',
        parentItem = None,
        frustum_visible = False,
        actual_depth = False,
    ):
        """
        深度相机

        Parameters:
        - view : GLViewWidget
        - img_size : tuple, optional, default: (640, 480), 相机图像大小
        - actual_depth : bool, optional, default: False, 是否返回实际深度值, 默认返回 0~1 的深度值, 对应 near~far
        """
        near_far_wrt_distance = False
        super().__init__(eye, center, up, aspect, fov, ortho_space, proj_type, near_far_wrt_distance, glOptions, parentItem, frustum_visible)
        self._actual_depth = actual_depth
        self.img_size = img_size
        self.setView(view)

        # fbo
        self._fbo = None

    def setView(self, view: GLViewWidget):
        if view is None or self.view() == view:
            return

        self.render_group: RenderGroup = view.item_group.copy()
        view.item_group.addItem(self)  # 将相机添加到视图中, self.view() 可访问 view
        super().setView(view)

    def init_fbo(self):
        """
        初始化 fbo
        """
        self._fbo = FBO(width=self.img_size[0], height=self.img_size[1], type=FBO.Type.DEPTH)
        self._shader = Shader(shadow_vertex_shader, empty_fragment_shader)

    def render(self, shader=None):
        """
        渲染深度图, 纹理数据中 0 对应 near, 1 对应 far, 若 actual_depth==True, 需要将 0~1 转换为 near ~ far 范围内的距离
        """
        if self.view() is None:
            raise ValueError("view is None")

        near, far = self._ortho_space[4], self._ortho_space[5]
        if self._near_far_wrt_distance:
            distance = self.get_distance()
            near += distance
            far += distance

        with self.view():

            if self._fbo is None:
                self.init_fbo()

            with self._fbo:
                gl.glDepthMask(gl.GL_TRUE)
                gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
                gl.glEnable(gl.GL_DEPTH_TEST)
                # gl.glEnable(gl.GL_CULL_FACE)
                # gl.glCullFace(gl.GL_BACK)
                self.render_group.render(
                    camera=self,
                    shader=self._shader if shader is None else shader
                )

            depth = self._fbo.depth_texture.getTexture()
            if self._actual_depth:
                return depth.squeeze() * (far - near) + near
            return depth.squeeze()

