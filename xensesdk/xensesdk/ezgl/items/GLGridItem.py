import numpy as np
import OpenGL.GL as gl
from typing import Sequence

from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .Buffer import VAO, VBO, EBO, GLDataBlock
from .light import LightMixin, PointLight, light_fragment_shader
from .MeshData import make_color
from .camera import Camera

__all__ = ['GLGridItem']


def make_grid_data(size, spacing):
    x, y = size
    dx, dy = spacing
    xvals = np.arange(-x/2., x/2. + dx*0.001, dx, dtype=np.float32)
    yvals = np.arange(-y/2., y/2. + dy*0.001, dy, dtype=np.float32)

    xlines = np.stack(
        np.meshgrid(xvals, [yvals[0], yvals[-1]], indexing='ij'),
        axis=2
    ).reshape(-1, 2)
    ylines = np.stack(
        np.meshgrid([xvals[0], xvals[-1]], yvals, indexing='xy'),
        axis=2
    ).reshape(-1, 2)
    data = np.concatenate([xlines, ylines], axis=0)
    data = np.pad(data, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)
    return data


class GLGridItem(GLGraphicsItem, LightMixin):
    """
    Displays xy plane.
    """
    def __init__(
        self,
        size = (1., 1.),
        spacing = (1.,1.),
        color=(0.78, 0.71, 0.60, 1),
        lineColor=(0.4, 0.3, 0.2, 1),
        lineWidth = 1,
        lights: Sequence[PointLight] = list(),
        render_shadow = False,
        glOptions = 'translucent',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.__color = make_color(color)
        self.__lineColor = make_color(lineColor)
        self.__lineWidth = lineWidth
        self.__render_shadow = render_shadow
        self.setGLOptions(glOptions)

        x, y = size
        _line_vert = make_grid_data(size, spacing)
        _plane_vert = np.array([-x/2., -y/2., 0,
                                -x/2.,  y/2., 0,
                                x/2.,  -y/2., 0,
                                x/2.,   y/2., 0], dtype=np.float32).reshape(-1, 3)

        self._vert = GLDataBlock(np.float32, 3, np.vstack([_plane_vert, _line_vert]))
        self.rotate(90, 1, 0, 0)
        self.setDepthValue(-1)
        self.addLight(lights)

    def setColor(self, color):
        self.__color = make_color(color)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, light_fragment_shader)
        with VAO() as self.vao:
            self.vbo1 = VBO(self._vert, False)
            self.vbo1.setAttrPointer(0, attr_id=0)

    def paint(self, camera: Camera):
        self.setupGLState()
        gl.glLineWidth(self.__lineWidth)

        with self.shader, self.vao:
            self.setupLight(self.shader)
            self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", self.model_matrix().glData, "mat4")
            self.shader.set_uniform("ViewPos",camera.get_eye(), "vec3")
            self.shader.set_uniform("material.disable", True, "bool")
            self.shader.set_uniform("material.shininess", 32, "float")

            # draw surface
            self.shader.set_uniform("aColor", self.__color, "vec4")
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

            # draw lines
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_DEPTH_TEST)
            self.shader.set_uniform("aColor", self.__lineColor, "vec4")
            gl.glDrawArrays(gl.GL_LINES, 4, self._vert.count - 4)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)

    def paintWithShader(self, camera: Camera, shader: Shader, **kwargs):
        if not self.__render_shadow:
            return

        self.setupGLState()
        with shader, self.vao:
            shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            shader.set_uniform("model", self.model_matrix().glData, "mat4")
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec4 aColor;

layout (location = 0) in vec3 aPos;
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;
out vec4 oColor;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalize(mat3(transpose(inverse(model))) * vec3(0, 0, -1));
    oColor = aColor;
    gl_Position = proj * view * vec4(FragPos, 1.0);
}
"""


