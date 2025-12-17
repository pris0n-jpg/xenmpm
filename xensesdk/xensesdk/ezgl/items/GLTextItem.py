import sys
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .shader import Shader
from .Buffer import VAO, VBO, GLDataBlock
from .texture import Texture2D
from .MeshData import make_color
from .camera import Camera
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import OpenGL.GL as gl

__all__ = ['GLTextItem']


class GLTextItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""

    def __init__(
        self,
        text: str=None,
        pos = [0, 0, -10],
        font = None,  # "times.ttf", "msyh.ttc", "Deng.ttf"
        color = (255, 255, 255, 255),
        fontsize = 40,
        fixed = True,  # 是否固定在视图上, if True, pos is in viewport, else in world
        glOptions = 'additive',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self.setDepthValue(100)
        self._fixed = fixed

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
        self._pixel_wh = [0, 0]
        self._text_w = 0
        self._text_h = 0
        self._texture = Texture2D(None, flip_y=True, wrap_s=gl.GL_CLAMP_TO_EDGE, wrap_t=gl.GL_CLAMP_TO_EDGE)

        self.setData(text, font, color, fontsize, pos)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        with VAO() as self.vao:
            self.vbo = VBO([self._vert_block], False, usage=gl.GL_STATIC_DRAW)
            self.vbo.setAttrPointer([0], attr_id=[[0,1]])

    def updateGL(self, model_matrix):
        # calc text box size (w, h), mantain pixel size
        if self._fixed:
            w = self._pixel_wh[0] * 2 / self.view().deviceWidth()
            h = self._pixel_wh[1] * 2 / self.view().deviceHeight()
            pos = self._pos * 2 - 1  # map to [-1, 1]
            pos[2] = 0
        else:
            pixelsize = self.view().pixelSize(model_matrix * self._pos)
            w = self._pixel_wh[0] * pixelsize
            h = self._pixel_wh[1] * pixelsize
            pos = self._pos

        # 更新顶点数据
        if w != self._text_w or h != self._text_h:
            self._text_w = w
            self._text_h = h
            self._vert[:, :3] = np.array([
                [0, 0, 0], [w, 0, 0],
                [w, h, 0], [w, h, 0],
                [0, h, 0], [0, 0, 0]
            ])
            self._vert_block.set_data(self._vert)

        self.vbo.commit()
        return pos

    def setData(self, text: str=None, font=None, color=None, fontsize=None, pos=None):
        """
        设置文本内容, 字体, 颜色, 大小, 位置

        Parameters:
        - text : str, optional, default: None, 文本内容
        - font : str, optional, default: None, 字体路径
        - color : tuple, optional, default: None, 颜色
        - fontsize : int, optional, default: None, 字体大小
        - pos : tupe, optional, default: None, xyz位置
        """
        text_changed_flag = False
        if text is not None:
            self._text = text
            text_changed_flag = True

        if color is not None:
            color = make_color(color)[0]
            self._color = np.array(color*255, np.uint8)
            text_changed_flag = True

        if font is not None:
            self._font = font
            text_changed_flag = True
        else:
            if sys.platform == "win32":
                self._font = "Deng.ttf"
            elif sys.platform in ("linux", "linux2"):
                self._font = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"

        if fontsize is not None:
            self._fontsize = fontsize
            text_changed_flag = True

        if pos is not None:
            self._pos = np.array(pos)

        if not text_changed_flag:
            return

        font = ImageFont.truetype(self._font, self._fontsize, encoding="unic") # Deng.ttf, msyh.ttc
        self._pixel_wh = np.array(font.getbbox(text)[2:])
        image = Image.new("RGBA", tuple(self._pixel_wh), (0, 0, 0, 0))  # 背景透明
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), self._text, font=font, fill=tuple(self._color), encoding="utf-8")
        self._texture.setTexture(np.array(image, np.uint8))

    def paint(self, camera: Camera):
        if self._text is None:
            return
        model_matrix = self.model_matrix()
        pos = self.updateGL(model_matrix)
        self.setupGLState()

        self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
        self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("is_fixed", self._fixed, "bool")
        self.shader.set_uniform("text_pos", pos, "vec3")

        with self.shader, self.vao:
            self.shader.set_uniform("texture1", self._texture.bindTexUnit(), "sampler2D")
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform bool is_fixed;
uniform vec3 text_pos;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec2 iTexCoord;

out vec2 TexCoord;

void main() {
    if (is_fixed) {
        gl_Position = vec4(text_pos + iPos, 1.0);
    } else {
        //gl_Position = vec4(text_pos + iPos, 1.0);
        vec4 tpos = view * model * vec4(text_pos, 1.0);
        gl_Position = proj * vec4(tpos.xyz + iPos, 1.0);
    }
    TexCoord =iTexCoord;
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D texture1;

void main() {
    FragColor = texture(texture1, TexCoord);
}
"""