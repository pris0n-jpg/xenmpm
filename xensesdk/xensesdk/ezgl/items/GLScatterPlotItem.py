from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .shader import Shader
from .Buffer import VAO, VBO, GLDataBlock
from .MeshData import make_color
from .camera import Camera
import numpy as np
import OpenGL.GL as gl

__all__ = ['GLScatterPlotItem']


class GLScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""

    def __init__(
        self,
        pos = None,
        size = 1,
        color = [1.0, 1.0, 1.0, 1.0],
        glOptions = 'opaque',
        auto_size = False,
        parentItem = None
    ):
        """
        Initializes the scatter plot item.

        Parameters:
        - pos : np.ndarray(..., 3), optional, default: None
        - size : int, optional, default: 1, 点的大小
        - color : list, optional, default: [1.0, 1.0, 1.0, 1.0]
        - glOptions : str, optional, default: 'opaque'
        - auto_size : bool, optional, default: False, 是否根据相机距离自动调整点的大小
        - parentItem : GLGraphicsItem, optional, default: None
        """
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._color = GLDataBlock(np.float32, 4)
        self._pos = GLDataBlock(np.float16, 3)
        self._size = None
        self._auto_size = auto_size
        self.setData(pos, color, size)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        with VAO() as self.vao:
            self.vbo = VBO([self._pos, self._color], True, usage=gl.GL_DYNAMIC_DRAW)
            self.vbo.setAttrPointer([0, 1], attr_id=[0, 1])

    def updateGL(self):
        self.vbo.commit()

    def setData(self, pos=None, color=None, size=None):
        """
        设置数据

        Parameters:
        - pos : np.ndarray, optional, default: None, (N,3) array of floats specifying point locations.
        - color : np.ndarray, optional, default: None, (N,4) array of floats (0.0-1.0) specifying a single color for all spots.
        - size : float, optional, default: None, a single value to apply to all spots.
        """
        self._pos.set_data(pos)
        self._color.set_data(make_color(color))
        if size is not None:
            self._size = np.float32(size)

    def paint(self, camera: Camera):
        if self.vbo.size() == 0:
            return

        self.updateGL()
        self.setupGLState()

        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)

        with self.shader, self.vao:
            self.shader.set_uniform("size", self._size, "float")
            self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", self.model_matrix().glData, "mat4")
            self.shader.set_uniform("auto_size", self._auto_size, "bool")
            gl.glDrawArrays(gl.GL_POINTS, 0, self.vbo.count(0))


vertex_shader = """
#version 330 core

uniform float size;
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform bool auto_size;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec4 iColor;

out vec4 oColor;

void main() {
    // 根据 camPos 和 iPos 计算出距离
    gl_Position = proj * view * model * vec4(iPos, 1.0);
    if(auto_size){
        float distance = gl_Position.z / 1.;
        gl_PointSize = 100 * size / distance;
    }
    else{
        gl_PointSize = 10 * size;
    }
    oColor = iColor;
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;
in vec4 oColor;

void main() {
    FragColor = oColor;
}
"""