import numpy as np
import OpenGL.GL as gl
from pathlib import Path
from threading import Lock
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .shader import Shader
from .Buffer import VAO, VBO, EBO, GLDataBlock
from .MeshData import grid3d, make_color
from .camera import Camera

BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GL3DGridItem']


class GL3DGridItem(GLGraphicsItem):

    def __init__(
        self,
        grid = None,
        color = (1, 1, 1, 1),
        lineWidth = 2,
        glOptions = 'translucent',
        parentItem = None,
        show_edge = True
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._show_edge = show_edge
        self._lineWidth = lineWidth
        self._verts = GLDataBlock(np.float32, 3)
        self._colors = GLDataBlock(np.float16, 4)
        self._inds = GLDataBlock(np.uint32, 1)
        self._lock = Lock()
        self.setData(grid, color)

    def setData(self, grid:np.ndarray=None, color=None):
        self._lock.acquire()
        if grid is not None:
            grid = np.array(grid, dtype=np.float32)

            if self._verts.used_size != grid.size:
                verts, inds = grid3d(grid)
                self._verts.set_data(verts)
                self._inds.set_data(inds)
            else:
                self._verts.set_data(grid.reshape(-1, 3))

        self._colors.set_data(make_color(color))
        self._lock.release()

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        with VAO() as self.vao:
            self.vbo = VBO([self._verts, self._colors], expandable=True)
            self.vbo.setAttrPointer([0, 1], attr_id=[0, 1])
            self.ebo = EBO(self._inds)

    def updateGL(self):
        self._lock.acquire()
        self.vbo.commit()
        self.ebo.commit()
        self._lock.release()

    def paint(self, camera: Camera):
        if self.vbo.size(0) == 0:
            return

        self.updateGL()
        self.setupGLState()

        gl.glLineWidth(self._lineWidth)
        self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
        self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
        self.shader.set_uniform("model", self.model_matrix().glData, "mat4")

        with self.shader, self.vao:
            gl.glDrawElements(gl.GL_QUADS, self.ebo.size(), gl.GL_UNSIGNED_INT, None)

            if self._show_edge:
                gl.glDisableVertexAttribArray(1)
                gl.glVertexAttrib4fv(1, np.array([0, 0, 0, 1], dtype=np.float32))
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glPolygonOffset(-1., -1.)
                gl.glDepthMask(gl.GL_FALSE)

                gl.glDrawElements(gl.GL_QUADS, self.ebo.size(), gl.GL_UNSIGNED_INT, None)

                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glDepthMask(gl.GL_TRUE)


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;

out vec4 oColor;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

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
