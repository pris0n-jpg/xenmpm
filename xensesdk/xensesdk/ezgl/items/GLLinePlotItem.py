from __future__ import annotations
import numpy as np
import OpenGL.GL as gl
from threading import Lock
from typing import Union
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .shader import Shader
from .Buffer import VAO, VBO, EBO, GLDataBlock
from .MeshData import make_color

# from .GLInstancedMeshItem import GLInstancedMeshItem

__all__ = ['GLLinePlotItem']


class GLLinePlotItem(GLGraphicsItem):

    def __init__(
        self,
        pos = None,
        lineWidth = 1,
        color = (1, 1, 1, 1),
        marker: 'GLInstancedMeshItem' = None,
        lineStipple = None,
        glOptions = 'translucent',
        parentItem = None
    ):
        """
        初始化一个线条绘制对象

        Parameters:
        - pos : np.array (N,3), optional, default: None, 线条顶点
        - lineWidth : int, optional, default: 1, 线条宽度
        - color : tuple, optional, default: (1, 1, 1, 1), 线条颜色, alpha 通道可选.
            * 注意: 如果初始化时设置了单个颜色值, 则之后添加的所有点共用该颜色值, 若想为每个点设置不同颜色, 设置为 None 或多个颜色值
        - marker : Union[GLInstancedMeshItem, Mesh], optional, default: None, 若设置了 marker, 则使用 InstancedMeshItem 绘制 marker
        - lineStipple : 0x0000~0xFFFF, optional, default: None, 线条样式, 例如 0x00FF 表示实线和空白线的交替, None 表示不启用线条样式
        - glOptions : str, optional, default: 'translucent', 管理OpenGL状态的字符串或字典
        - parentItem : GLGraphicsItem, optional, default: None, 父对象
        """
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._lineWidth = lineWidth
        self._lineStipple = lineStipple
        self._verts = GLDataBlock(np.float32, 3)
        self._color = GLDataBlock(np.float16, 4)
        self._lock = Lock()

        if marker is not None:
            self._marker_plot = marker
            self._marker_plot.setParentItem(self)
        else:
            self._marker_plot = None

        self.setData(pos, color)

    def setData(self, pos=None, color=None):
        """
        设置数据

        Parameters:
        - pos : np.ndarray, shape=(n,3) or (3,), optional, default: None, 顶点坐标, None 表示不改变数据,
            若要清空数据, 传入 np.empty(0, dtype) 或 []
        - color : Union[list, np.ndarray], 可以为 0-255 或 0-1, shape=(3,) or (4,) or (n, 3) or (n, 4)
        """
        self._lock.acquire()
        self._verts.set_data(pos)
        self._color.set_data(color)
        if self._marker_plot is not None:
            self._marker_plot.setData(pos)
        self._lock.release()

    def addData(self, pos=None, color=None):
        """
        添加数据

        Parameters:
        - pos : np.ndarray, shape=(n,3) or (3,), optional, default: None, 顶点坐标
        - color : Union[list, np.ndarray], 可以为 0-255 或 0-1, shape=(3,) or (4,) or (n, 3) or (n, 4)
        """
        self._lock.acquire()
        self._verts.add_data(pos)
        self._color.add_data(make_color(color))
        if self._marker_plot is not None:
            self._marker_plot.addData(pos)
        self._lock.release()

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        with VAO() as self.vao:
            self.vbo = VBO([self._verts, self._color], True, usage = gl.GL_DYNAMIC_DRAW)
            self.vbo.setAttrPointer([0, 1], attr_id=[0, 1])

    def updateGL(self):
        self._lock.acquire()
        self.vbo.commit()
        self._lock.release()

    def paint(self, camera):
        if self.vbo.size(0) == 0:
            return

        self.updateGL()
        self.setupGLState()

        gl.glLineWidth(self._lineWidth)

        if self._lineStipple is not None:
            gl.glEnable(gl.GL_LINE_STIPPLE)
            gl.glLineStipple(1, self._lineStipple)

        with self.shader, self.vao:
            self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", self.model_matrix().glData, "mat4")
            gl.glDrawArrays(gl.GL_LINE_STRIP, 0, self.vbo.count(0))

        if self._lineStipple is not None:
            gl.glDisable(gl.GL_LINE_STIPPLE)


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