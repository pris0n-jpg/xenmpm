import OpenGL.GL as gl
import numpy as np
from threading import Lock
from typing import List, Sequence
from .shader import Shader
from .Buffer import VBO, GLDataBlock
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .MeshData import make_color, Mesh
from .light import LightMixin, light_fragment_shader
__all__ = ['GLInstancedMeshItem']


class GLInstancedMeshItem(GLGraphicsItem, LightMixin):

    def __init__(
        self,
        mesh: Mesh,
        pose: Sequence[Matrix4x4] = None,  # nx16
        lights: Sequence = list(),
        color = None,
        color_divisor = 1,
        glOptions = 'translucent_cull',
        parentItem = None
    ):
        """
        Initialization

        Parameters:
        - mesh : Mesh
        - pose : Sequence[Matrix4x4], optional, default: None, 位姿矩阵
        - lights : Sequence, optional, default: list(), 光源
        - color : optional, default: None, 单个或多个颜色值, 若为None, 将使用 mesh 的 material
        - color_divisor : int, optional, default: 1, 若为 1, 每个实例使用一个颜色值, 若为0, 每个顶点使用一个颜色值
        - glOptions : str, optional, default: 'translucent_cull'
        - parentItem : GLGraphicsItem, optional, default: None
        """
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self._lock = Lock()
        self._color_divisor = color_divisor
        self._mesh = mesh
        self._pose = GLDataBlock(np.float32, [4,4,4,4], None)
        self._color = GLDataBlock(np.float16, 4, None)
        self.setData(pose, color)
        self.addLight(lights)

    def _make_pose(self, pose):
        """
        规范化输入的位姿矩阵

        Parameters:
        - pose : 位姿矩阵数据, 可以为如下类型:
            * Matrix4x4
            * List[Matrix4x4]
            * np.ndarray, shape=(n, 3), 只定义 xyz
            * np.ndarray, shape=(n, 4, 4), 完整的位姿矩阵, 4x4 部分按 GLData 顺序排列, 即单位坐标轴在前三行

        Returns:
        - List[List[float]], shape=(n, 16)
        """
        if pose is None:
            return None
        if isinstance(pose, Matrix4x4):
            return [pose.glData]
        elif isinstance(pose, list):
            return [p.glData for p in pose]
        elif isinstance(pose, np.ndarray):
            if pose.ndim == 1 and pose.shape[0] == 3:
                pose = pose.reshape(1, 3)
            
            if pose.ndim == 2 and pose.shape[1] == 3:
                return [Matrix4x4.fromTranslation(*p).glData for p in pose]
            elif pose.ndim == 3 and pose.shape[1:] == (4, 4):
                return pose.flatten()  # 将numpy数据行优先改成列优先
            else:
                raise ValueError("Invalid pose data")
        else:
            raise ValueError("Invalid pose data")

    def setData(self, pose=None, color=None):
        """
        设置数据

        Parameters:
        - pose : 位姿矩阵数据, 可以为如下类型:
            * Matrix4x4
            * List[Matrix4x4]
            * np.ndarray, shape=(n, 3) or (3,), 只定义 xyz
        - color : Union[list, np.ndarray], 可以为 0-255 或 0-1, shape=(3,) or (4,) or (n, 3) or (n, 4)
        """
        with self._lock:
            self._color.set_data(make_color(color))
            self._pose.set_data(self._make_pose(pose))

    def setOpacity(self, opacity: float):
        color = self._color._data
        color[:, 3] = opacity
        self._color.set_data(color)

    def addData(self, pose=None, color=None):
        """
        添加数据

        Parameters:
        - pose : 位姿矩阵数据, 可以为如下类型:
            * Matrix4x4
            * List[Matrix4x4]
            * np.ndarray, shape=(n, 3) or (3,), 只定义 xyz
        - color : Union[list, np.ndarray], 可以为 0-255 或 0-1, shape=(3,) or (4,) or (n, 3) or (n, 4)
        """
        with self._lock:
            self._color.add_data(make_color(color))
            self._pose.add_data(self._make_pose(pose))

    def initializeGL(self):
        self.shader = Shader(vertex_shader, light_fragment_shader)
        self._mesh.initializeGL()

        # 若设置了颜色, 则禁用材质
        if self._color.used_size > 0:
            self.shader.set_uniform("material.disable", True, "bool")

        # vbo 由 self._mesh.vao 管理
        with self._mesh.vao:
            self.vbo_pose = VBO([self._pose, self._color], expandable=True, usage=gl.GL_DYNAMIC_DRAW)
            self.vbo_pose.setAttrPointer([0, 1], [[3,4,5,6], 7], divisor=[1, self._color_divisor])

    def updateGL(self):
        self._lock.acquire()
        self.vbo_pose.commit()
        self._lock.release()

    def paint(self, camera):
        if self.vbo_pose.size(0)==0:
            return

        self.updateGL()
        self.setupGLState()

        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        with self.shader, self._mesh.vao:
            self.setupLight(self.shader)
            self._mesh._material.set_uniform(self.shader, "material")
            if self._mesh.vbo.count(2) == 0:
                self.shader.set_uniform("material.use_texture", False, 'bool')
            self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", self.model_matrix().glData, "mat4")
            self.shader.set_uniform("ViewPos",camera.get_eye(), "vec3")

            if self._mesh.ebo.size():
                gl.glDrawElementsInstanced(
                    gl.GL_TRIANGLES,
                    self._mesh.ebo.size(),
                    gl.GL_UNSIGNED_INT, None,
                    self.vbo_pose.count(0),
                )
            else:
                gl.glDrawArraysInstanced(
                    gl.GL_TRIANGLES,
                    0,
                    self._mesh.vbo.count(0),
                    self.vbo_pose.count(0),
                )
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec4 row1;
layout (location = 4) in vec4 row2;
layout (location = 5) in vec4 row3;
layout (location = 6) in vec4 row4;
layout (location = 7) in vec4 aColor;

out vec3 FragPos;
out vec3 Normal;
out vec4 oColor;
out vec2 TexCoords;

void main() {
    oColor = aColor;
    TexCoords = aTexCoords;
    mat4 pose = model * mat4(row1, row2, row3, row4);
    FragPos = vec3(pose * vec4(aPos, 1.0));
    Normal = normalize(mat3(transpose(inverse(pose))) * aNormal);
    gl_Position = proj * view * vec4(FragPos, 1.0);
}
"""

