import numpy as np
import OpenGL.GL as gl

from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .Buffer import VAO, VBO, GLDataBlock
from .MeshData import cube
from .camera import Camera

__all__ = ['GLBoxItem']

class GLBoxItem(GLGraphicsItem):
    """
    Displays Box.
    """
    def __init__(
        self,
        size=Vector3(1.,1.,1.),
        color=Vector3(1.0, 0.5, 0.31),
        glOptions='opaque',
        parentItem=None
    ):
        super().__init__(parentItem=parentItem)
        self.__size = Vector3(size)
        self.__color = Vector3(color)
        self.setGLOptions(glOptions)
        vert, norm, _ = cube(self.__size.x, self.__size.y, self.__size.z)
        self._vert = GLDataBlock(np.float32, 3, vert)
        self._norm = GLDataBlock(np.float32, 3, norm)

    def size(self):
        return self.__size.xyz

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        with VAO() as self.vao:
            self.vbo1 = VBO([self._vert, self._norm], False)
            self.vbo1.setAttrPointer([0, 1], attr_id=[[0, 1]])

    def paint(self, camera: Camera):
        self.setupGLState()

        self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
        self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
        self.shader.set_uniform("model", self.model_matrix().glData, "mat4")

        self.shader.set_uniform("lightPos", Vector3([3, 2.0, 2.0]), "vec3")
        self.shader.set_uniform("lightColor", Vector3([1.0, 1.0, 1.0]), "vec3")
        self.shader.set_uniform("objColor", self.__color, "vec3")

        with self.shader, self.vao:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec3 iNormal;

out vec3 FragPos;
out vec3 Normal;

void main() {
    FragPos = vec3(model * vec4(iPos, 1.0));
    Normal = normalize(mat3(transpose(inverse(model))) * iNormal);

    gl_Position = proj * view * vec4(FragPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 objColor;

void main() {

    // ambient
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * lightColor * objColor;

    // diffuse
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(Normal, lightDir), 0.0);
    vec3 diffuse = lightColor * (diff * objColor);

    vec3 result = ambient + diffuse;
    FragColor = vec4(result, 1.0);
}
"""