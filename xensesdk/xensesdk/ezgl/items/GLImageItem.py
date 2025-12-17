from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .shader import Shader
from .Buffer import VAO, VBO, GLDataBlock
from .texture import Texture2D
import numpy as np
import OpenGL.GL as gl

__all__ = ['GLImageItem']


class GLImageItem(GLGraphicsItem):
    """Display Image."""

    def __init__(
        self,
        img: np.ndarray = None,
        left_bottom = (0, 0),  # 左下角坐标 0 ~ 1
        width_height = (1, 1),  # 宽高 0 ~ 1
        glOptions = 'opaque',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._texture = Texture2D(None, flip_y=True)
        self._left_bottom = None
        self._width_height = None
        self._vert = np.array( [
                # 顶点坐标    # texcoord
                -1, -1, 0,   0.0, 0.0,
                1, -1, 0,   1.0, 0.0,
                1,  1, 0,   1.0, 1.0,
                1,  1, 0,   1.0, 1.0,
                -1,  1, 0,   0.0, 1.0,
                -1, -1, 0,   0.0, 0.0,
            ], dtype=np.float32).reshape(-1, 5)
        self._vert_block = GLDataBlock(np.float32, [3, 2], self._vert)
        self.setData(img=img, left_bottom=left_bottom, width_height=width_height)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        with VAO() as self.vao:
            self.vbo = VBO(self._vert_block, False, usage=gl.GL_STATIC_DRAW)
            self.vbo.setAttrPointer([0], attr_id=[[0,1]])

    def updateGL(self):
        self.vbo.commit()

    def setData(self, img:np.ndarray=None, left_bottom=None, width_height=None):
        if img is not None and img.dtype != np.uint8 and np.max(img) > 1:
            img = img.astype(np.uint8)

        self._texture.setTexture(img)

        if left_bottom is not None or width_height is not None:
            if left_bottom is not None:
                self._left_bottom = left_bottom
            if width_height is not None:
                self._width_height = width_height

            l, b = self._left_bottom
            l = l * 2 - 1
            b = b * 2 - 1
            w, h = self._width_height
            w, h = w * 2, h * 2

            self._vert[:, :2] = np.array([
                [l, b],
                [l + w, b],
                [l + w, b + h],
                [l + w, b + h],
                [l, b + h],
                [l, b]
            ])
            self._vert_block.set_data(self._vert)

    def paint(self, camera):
        if self._texture.img is None:
            return
        self.updateGL()
        self.setupGLState()

        with self.shader, self.vao:
            self.shader.set_uniform("texture1", self._texture.bindTexUnit(), "sampler2D")
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)


vertex_shader = """
#version 330 core

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec2 iTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(iPos, 1.0);
    TexCoord =iTexCoord;
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D texture1;

void main() {
    FragColor = vec4(texture(texture1, TexCoord).rgb, 1.0);
}
"""