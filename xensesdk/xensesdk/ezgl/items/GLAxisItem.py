import OpenGL.GL as gl
import numpy as np

from .shader import Shader
from .Buffer import VBO, EBO, VAO, GLDataBlock
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .MeshData import cone
from .camera import Camera

__all__ = ['GLAxisItem']


class GLAxisItem(GLGraphicsItem):
    """
    Displays three lines indicating origin and orientation of local coordinate system.
    """

    def __init__(
        self,
        size=Vector3(1.,1.,1.),
        width=2,
        tip_size=1,
        glOptions='opaque',
        fix_to_corner=False,
        parentItem=None
    ):
        super().__init__(parentItem=parentItem)
        self.__size = Vector3(size)
        self.__width = width
        self.__fix_to_corner = fix_to_corner

        self.setGLOptions(glOptions)
        if fix_to_corner:
            # 保证坐标轴不会被其他物体遮挡
            self.updateGLOptions({"glClear": (gl.GL_DEPTH_BUFFER_BIT,)})
            self.setDepthValue(1000)  # make sure it is drawn last

        self._verts = GLDataBlock(np.float32, layout=3, data = [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                                 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] )

        self._colors = GLDataBlock(np.float16, layout=3, data = [ 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                                  0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                                                  0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        cone_vertices, cone_indices = cone(0.06*width*tip_size, 0.15*width*tip_size)
        self._cone_verts = GLDataBlock(np.float32, layout=3, data=cone_vertices)
        self._cone_colors = GLDataBlock(np.float16, layout=3, data=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self._cone_inds = GLDataBlock(np.uint32, layout=1, data=cone_indices)

        tfs = [Matrix4x4.fromEulerAngles(0, 90, 0).moveto(self.__size.x, 0, 0).matrix44.T,
               Matrix4x4.fromEulerAngles(-90, 0, 0).moveto(0, self.__size.y, 0).matrix44.T,
               Matrix4x4.fromTranslation(0, 0, self.__size.z).matrix44.T]

        self._transforms = GLDataBlock(np.float32, layout=[4,4,4,4], data=np.concatenate(tfs, axis=0))

    def setSize(self, x=None, y=None, z=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z.
        """
        x = x if x is not None else self.__size.x
        y = y if y is not None else self.__size.y
        z = z if z is not None else self.__size.z
        self.__size = Vector3(x,y,z)

    def size(self):
        return self.__size.xyz

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.shader_cone = Shader(vertex_shader_cone, fragment_shader)
        with VAO() as self.vao:
            self.vbo = VBO([self._verts, self._colors, self._cone_verts, self._cone_colors, self._transforms], False)
            self.vbo.setAttrPointer([0, 1, 2, 3, 4], attr_id=[0, 1, 2, 3, [4,5,6,7]], divisor=[0, 0, 0, 1, 1])
            self.ebo = EBO(self._cone_inds)

    def paint(self, camera: Camera):
        self.setupGLState()
        gl.glLineWidth(self.__width)

        self.vbo.commit()
        self.vao.bind()

        proj_view = self.proj_view_matrix(camera).glData
        model_matrix = self.model_matrix().glData
        with self.shader:
            self.shader.set_uniform("sizev3", self.size(), "vec3")
            self.shader.set_uniform("view", proj_view, "mat4")
            self.shader.set_uniform("model", model_matrix, "mat4")
            gl.glDrawArrays(gl.GL_LINES, 0, 6)

        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        with self.shader_cone:
            self.shader_cone.set_uniform("view", proj_view, "mat4")
            self.shader_cone.set_uniform("model", model_matrix, "mat4")
            gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.ebo.size(), gl.GL_UNSIGNED_INT, None, 3)
        gl.glDisable(gl.GL_CULL_FACE)
        self.vao.unbind()

    def proj_view_matrix(self, camera) -> Matrix4x4:
        if self.__fix_to_corner:
            view = self.view()
            proj = Matrix4x4.perspective(
                20, 1 / view.deviceRatio(), 1, 200.0
            )
            # 计算在这个投影矩阵下, 窗口右上角点在相机坐标系下的坐标
            pos = proj.inverse() * (0.75, 0.75, 0.96)
            return proj * camera.get_view_matrix().moveto(*pos)
        else:
            return camera.get_proj_view_matrix()


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform vec3 sizev3;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec3 iColor;

out vec3 oColor;

void main() {
    gl_Position =  view * model * vec4(iPos * sizev3, 1.0);
    oColor = iColor;
}
"""

vertex_shader_cone = """
#version 330 core

uniform mat4 model;
uniform mat4 view;

layout (location = 2) in vec3 iPos;
layout (location = 3) in vec3 iColor;
layout (location = 4) in vec4 row1;
layout (location = 5) in vec4 row2;
layout (location = 6) in vec4 row3;
layout (location = 7) in vec4 row4;
out vec3 oColor;

void main() {
    mat4 transform = mat4(row1, row2, row3, row4);
    gl_Position =  view * model * transform * vec4(iPos, 1.0);
    oColor = iColor;
}
"""

fragment_shader = """
#version 330 core

in vec3 oColor;
out vec4 fragColor;

void main() {
    fragColor = vec4(oColor, 1.0f);
}
"""