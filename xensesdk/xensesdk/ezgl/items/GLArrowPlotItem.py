import OpenGL.GL as gl
import numpy as np
from .shader import Shader
from .Buffer import VBO, EBO, VAO, GLDataBlock
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .MeshData import cone, direction_matrixs, make_color, Mesh, cylinder
from .camera import Camera
from threading import Lock
from .GLInstancedMeshItem import GLInstancedMeshItem
import platform

__all__ = ['GLArrowPlotItem']


class GLArrowPlotItem(GLGraphicsItem):
    """
    Displays Arrows.
    """

    def __init__(
        self,
        start_pos = None,
        end_pos = None,
        color = [1., 1., 1., 1.],
        tip_size = [0.1, 0.2],  # radius, height
        tip_pos = 0,  # bias of tip position, end + tip_pos * (end - start)/norm(end - start)
        width = 1.,
        glOptions='opaque',
        parentItem=None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self._lock = Lock()
        self._width = width

        cone_vertices, cone_indices = cone(tip_size[0]*width, tip_size[1]*width)
        cone_vertices += np.array([0, 0, tip_pos], dtype=np.float32)

        self._cone_vert = GLDataBlock(layout=3, data=cone_vertices)
        self._cone_ind = GLDataBlock(dtype=np.uint32, data=cone_indices)
        self._st_pos = GLDataBlock(dtype=np.float32, layout=3)
        self._ed_pos = GLDataBlock(dtype=np.float32, layout=3)
        self._color = GLDataBlock(dtype=np.float16, layout=4)
        self._transform = GLDataBlock(dtype=np.float32, layout=[4,4,4,4])
        self.setData(start_pos, end_pos, color)  # to update transform

    def initializeGL(self):
        """
        初始化 shader, vao, vbo, ebo
        """
        self.shader = Shader(vertex_shader, fragment_shader, geometry_shader)
        self.shader_cone = Shader(vertex_shader_cone, fragment_shader)

        with VAO() as self.vao:
            # cone
            self.ebo_cone = EBO(self._cone_ind)
            self.vbo_cone = VBO(self._cone_vert, usage=gl.GL_STATIC_DRAW)
            self.vbo_cone.setAttrPointer(0, attr_id=7, divisor=0)

            # line
            self.vbo_shaft = VBO(
                [self._st_pos, self._ed_pos, self._color, self._transform],
                expandable=True,
                usage=gl.GL_DYNAMIC_DRAW
            )
            self.vbo_shaft.setAttrPointer(
                [0, 1, 2, 3],
                attr_id=[0, 1, 2, [3,4,5,6]],
                divisor=[0, 0, 1, 1],
            )

    def setData(self, start_pos=None, end_pos=None, color=None):
        self._lock.acquire()
        self._st_pos.set_data(start_pos)
        self._ed_pos.set_data(end_pos)
        self._color.set_data(make_color(color))

        if self._st_pos.pending or self._ed_pos.pending:  # 如果起点或终点有变化, 则更新 transform
            assert self._st_pos.used_nbytes == self._ed_pos.used_nbytes, \
                    "start_pos and end_pos must have the same size"
            transform, _ = direction_matrixs(self._st_pos.data.reshape(-1,3),
                                            self._ed_pos.data.reshape(-1,3))
            self._transform.set_data(transform)
        self._lock.release()

    def addData(self, start_pos: np.ndarray, end_pos: np.ndarray, color: np.ndarray=None):
        self._lock.acquire()
        self._st_pos.add_data(start_pos)
        self._ed_pos.add_data(end_pos)
        self._color.add_data(color)

        assert self._st_pos.used_nbytes == self._ed_pos.used_nbytes, \
                "start_pos and end_pos must have the same size"

        transform, _ = direction_matrixs(start_pos.reshape(-1,3), end_pos.reshape(-1,3))
        self._transform.add_data(transform)
        self._lock.release()

    def updateGL(self):
        self._lock.acquire()
        self.vbo_cone.commit()
        self.ebo_cone.commit()
        self.vbo_shaft.commit()
        self._lock.release()

    def paint(self, camera: Camera):
        if self._st_pos.used_size == 0:
            return

        self.updateGL()
        self.setupGLState()
        gl.glLineWidth(self._width)

        self.vao.bind()
        model_matrix = self.model_matrix()

        with self.shader:
            self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.vbo_shaft.setAttrPointer(2, divisor=0, attr_id=2)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.vbo_shaft.count(0))

        with self.shader_cone:
            self.shader_cone.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            self.shader_cone.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            self.shader_cone.set_uniform("model", model_matrix.glData, "mat4")
            self.vbo_shaft.setAttrPointer(2, divisor=1, attr_id=2)
            gl.glDrawElementsInstanced(
                gl.GL_TRIANGLES,
                self.ebo_cone.size(),
                gl.GL_UNSIGNED_INT, None,
                self.vbo_shaft.count(0),
            )
        self.vao.unbind()


class GLArrowMeshItem(GLGraphicsItem):
    """
    使用 InstancedMeshItem 绘制箭头
    """
    
    def __init__(
        self,
        start_pos = None,
        end_pos = None,
        tip_size = [0.06, 0.1],  # radius, height
        width = 0.03,
        color = [1., 1., 1., 1.],
        color_divisor = 1,
        lights = list(),
        glOptions='opaque',
        parentItem=None
    ):
        super().__init__(parentItem=parentItem)
        cols = 4 if platform.machine()=="aarch64" else 8
        cone_verts, cone_faces = cylinder([tip_size[0], 0], tip_size[1], 1, cols)
        shaft_verts, shaft_faces = cylinder([width, width], 1, 1, cols)

        self.cone_item = GLInstancedMeshItem(Mesh(cone_verts, cone_faces), lights=lights, color=color, color_divisor=color_divisor, glOptions=glOptions, parentItem=self)
        self.shaft_item = GLInstancedMeshItem(Mesh(shaft_verts, shaft_faces), lights=lights, color=color, color_divisor=color_divisor, glOptions=glOptions, parentItem=self)
        self.setData(start_pos, end_pos, color)
            
    def setData(self, start_pos=None, end_pos=None, color=None):
        
        if start_pos is None or end_pos is None:
            return
        
        cone_tf, arrow_lens = direction_matrixs(start_pos.reshape(-1,3), end_pos.reshape(-1,3))
        shaft_tf = cone_tf.copy()
        shaft_tf[:, 3, :3] = start_pos.reshape(-1,3)
        shaft_tf[:, 2, :3] *= arrow_lens[:, np.newaxis]
        
        self.cone_item.setData(cone_tf, color)
        self.shaft_item.setData(shaft_tf, color)
    
    
    
vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 stPos;
layout (location = 1) in vec3 endPos;
layout (location = 2) in vec4 aColor;

out V_OUT {
    vec4 endPos;
    vec4 color;
} v_out;

void main() {
    mat4 matrix = proj * view * model;
    gl_Position =  matrix * vec4(stPos, 1.0);
    v_out.endPos = matrix * vec4(endPos, 1.0);
    v_out.color = aColor;
}
"""

vertex_shader_cone = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 7) in vec3 iPos;
layout (location = 2) in vec4 aColor;
layout (location = 3) in vec4 row1;
layout (location = 4) in vec4 row2;
layout (location = 5) in vec4 row3;
layout (location = 6) in vec4 row4;
out vec4 oColor;

void main() {
    mat4 transform = mat4(row1, row2, row3, row4);
    gl_Position =  proj * view * model * transform * vec4(iPos, 1.0);
    oColor = vec4(aColor * vec4(0.9, 0.9, 0.9, 1));
}
"""

geometry_shader = """
#version 330 core
layout(points) in;
layout(line_strip, max_vertices = 2) out;

in V_OUT {
    vec4 endPos;
    vec4 color;
} gs_in[];
out vec4 oColor;

void main() {
    oColor = gs_in[0].color;
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    gl_Position = gs_in[0].endPos;
    EmitVertex();
    EndPrimitive();
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