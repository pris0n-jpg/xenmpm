import numpy as np
import OpenGL.GL as gl
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .shader import Shader
from .Buffer import VAO, VBO, EBO, GLDataBlock
from .MeshData import surface, make_color
from .camera import Camera

__all__ = ['GLColorSurfaceItem']


class GLColorSurfaceItem(GLGraphicsItem):

    def __init__(
        self,
        zmap = None,
        x_size = 10, # scale width to this size
        color = (1, 1, 1),
        opacity = None,
        glOptions = 'translucent',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._shape = (0, 0)
        self._x_size = x_size
        self._vertexes = GLDataBlock(np.float32, 3)
        self._colors = GLDataBlock(np.float16, 4)
        self._indices = GLDataBlock(np.uint32, 3)
        self._opacity = opacity
        self.scale_ratio = 1
        self.setData(zmap, color, opacity)

    def setData(self, zmap=None, color=None, opacity=None):
        # update vertexes and indices
        if zmap is not None:
            zmap =np.array(zmap, dtype=np.float32)
            h, w = zmap.shape
            self.scale_ratio = self._x_size / w

            if self._shape != zmap.shape:
                self._shape = zmap.shape
                self.xy_size = (self._x_size, self.scale_ratio * h)
                _vertexes, _indices = surface(zmap, self.xy_size)
                self._vertexes.set_data(_vertexes)
                self._indices.set_data(_indices)
            else:
                _vertexes = self._vertexes.data
                _vertexes[:, 2] = zmap.reshape(-1) * self.scale_ratio
                self._vertexes.set_data(_vertexes)  # xy 不变，只更新 z

        if opacity is not None:
            self._opacity = opacity

        if color is not None:
            self._colors.set_data(make_color(color, self._opacity))

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        with VAO() as self.vao:
            self.vbo = VBO([self._vertexes, self._colors], True, gl.GL_DYNAMIC_DRAW)
            self.vbo.setAttrPointer([0, 1], attr_id=[0, 1])
            self.ebo = EBO(self._indices)

    def updateGL(self):
        self.vbo.commit()
        self.ebo.commit()

    def paint(self, camera: Camera):
        if self._shape[0] == 0:
            return

        self.updateGL()
        self.setupGLState()

        self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
        self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
        self.shader.set_uniform("model", self.model_matrix().glData, "mat4")

        with self.shader, self.vao:
            gl.glDrawElements(gl.GL_TRIANGLES, self.ebo.size(), gl.GL_UNSIGNED_INT, None)


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;

out vec4 oColor;

uniform mat4 view;
uniform mat4 model;
uniform mat4 proj;

void main() {
    gl_Position = proj * view * model * vec4(aPos, 1.0);
    oColor = aColor;
}
"""


fragment_shader = """
#version 330 core


in vec4 oColor;
out vec4 fragColor;

void main() {
    fragColor = oColor;
}
"""
