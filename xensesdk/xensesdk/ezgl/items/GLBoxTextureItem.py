import numpy as np
import OpenGL.GL as gl

from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .texture import Texture2D
from .Buffer import VAO, VBO, EBO, GLDataBlock
from .MeshData import cube
from .camera import Camera
from ..functions import get_path

__all__ = ['GLBoxTextureItem']

class GLBoxTextureItem(GLGraphicsItem):
    """
    Displays Box.
    """

    def __init__(
        self,
        size=Vector3(1.,1.,1.),
        glOptions='opaque',
        parentItem=None
    ):
        super().__init__(parentItem=parentItem)
        self.__size = Vector3(size)
        self.setGLOptions(glOptions)
        # texture
        self.texture = Texture2D(
            source = get_path() / "resources/textures/box.png",
            tex_type = "tex_diffuse"
        )
        vert, norm, uv = cube(self.__size.x, self.__size.y, self.__size.z)
        self._vert = GLDataBlock(np.float32, 3, vert)
        self._norm = GLDataBlock(np.float32, 3, norm)
        self._uv = GLDataBlock(np.float32, 2, uv)

    def size(self):
        return self.__size.xyz

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        with VAO() as self.vao:
            self.vbo = VBO([self._vert, self._norm, self._uv], False)
            self.vbo.setAttrPointer([0, 1, 2], attr_id=[0, 1, 2])

    def paint(self, camera: Camera):
        self.setupGLState()

        self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
        self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
        self.shader.set_uniform("model", self.model_matrix().glData, "mat4")

        self.shader.set_uniform("lightPos", Vector3([3, 2.0, 2.0]), "vec3")
        self.shader.set_uniform("lightColor", Vector3([1.0, 1.0, 1.0]), "vec3")

        with self.shader, self.vao:
            self.shader.set_uniform("texture1", self.texture.bindTexUnit(), "sampler2D")
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec3 iNormal;
layout (location = 2) in vec2 iTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void main() {
    FragPos = vec3(model * vec4(iPos, 1.0));
    Normal = normalize(mat3(transpose(inverse(model))) * iNormal);
    TexCoord = iTexCoord;

    gl_Position = proj * view * vec4(FragPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D texture1;

in vec3 FragPos;
in vec3 Normal;
uniform vec3 lightColor;
uniform vec3 lightPos;


void main() {
    vec3 objColor = texture(texture1, TexCoord).rgb;
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