import numpy as np
import cv2
import OpenGL.GL as gl
from typing import Sequence
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .shader import Shader
from .Buffer import VAO, VBO, EBO, GLDataBlock
from .MeshData import Material, surface
from .texture import Texture2D
from .light import LightMixin, light_fragment_shader
from .camera import Camera


__all__ = ['GLSurfacePlotItem']


class GLSurfacePlotItem(GLGraphicsItem, LightMixin):

    def __init__(
        self,
        zmap = None,
        x_size = 10, # scale width to this size
        material = dict(),
        lights: Sequence = list(),
        glOptions = 'translucent',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self._shape = (0, 0)
        self._x_size = x_size
        self._vertexes = GLDataBlock(np.float32, 3, None)
        self._indices = GLDataBlock(np.uint32, 3, None)
        self._normal_texture = Texture2D(None, flip_x=False, flip_y=True)
        self.scale_ratio = 1
        self.setData(zmap)
        # material
        self.setMaterial(material)

        # light
        self.addLight(lights)

    def setData(self, zmap=None):
        if zmap is None:
            return
        else:
            zmap = np.array(zmap, dtype=np.float32)

        h, w = zmap.shape
        self.scale_ratio = self._x_size / w

        # calc vertexes
        if self._shape != zmap.shape:
            self._shape = zmap.shape
            self.xy_size = (self._x_size, self.scale_ratio * h)
            _vertexes, _indices = surface(zmap, self.xy_size)

            self._vertexes.set_data(_vertexes)
            self._indices.set_data(_indices)

        else:
            _vertexes = self._vertexes.data
            _vertexes[:, 2] = zmap.reshape(-1) * self.scale_ratio
            self._vertexes.set_data(_vertexes)

        # calc normals texture
        v = self._vertexes.data.copy()[self._indices.data.copy()]  # Nf x 3 x 3
        v = np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]) # face Normal Nf(c*r*2) x 3
        v = v.reshape(h-1, 2, w-1, 3).sum(axis=1, keepdims=False)  # r x c x 3
        v = cv2.GaussianBlur(v, (5, 5), 0)  #
        _normal_img = v / np.linalg.norm(v, axis=-1, keepdims=True)
        self._normal_texture.setTexture(_normal_img)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, light_fragment_shader)
        with VAO() as self.vao:
            self.vbo = VBO(self._vertexes, True, usage = gl.GL_DYNAMIC_DRAW)
            self.vbo.setAttrPointer(0, attr_id=0)
            self.ebo = EBO(self._indices, gl.GL_DYNAMIC_DRAW)

    def updateGL(self):
        self.vbo.commit()
        self.ebo.commit()

    def paint(self, camera: Camera):
        if self._shape[0] == 0:
            return
        self.updateGL()
        self.setupGLState()

        with self.shader, self.vao:
            self.setupLight(self.shader)
            self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            self.shader.set_uniform("model",self.model_matrix().glData, "mat4")
            self.shader.set_uniform("ViewPos",camera.get_eye(), "vec3")
            self._material.set_uniform(self.shader, "material")
            self.shader.set_uniform("norm_texture", self._normal_texture.bindTexUnit(), "sampler2D")
            self.shader.set_uniform("texScale", self.xy_size, "vec2")

            gl.glDrawElements(gl.GL_TRIANGLES, self.ebo.size(), gl.GL_UNSIGNED_INT, None)

    def setMaterial(self, material):
        if isinstance(material, dict):
            self._material = Material(material)
        elif isinstance(material, Material):
            self._material = material

    def getMaterial(self):
        return self._material

    def paintWithShader(self, camera: Camera, shader: Shader, **kwargs):
        self.updateGL()
        self.setupGLState()
        with shader, self.vao:
            shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            shader.set_uniform("model", self.model_matrix().glData, "mat4")
            gl.glDrawElements(gl.GL_TRIANGLES, self.ebo.size(), gl.GL_UNSIGNED_INT, None)


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;
out vec4 oColor;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;
uniform vec2 texScale;
uniform sampler2D norm_texture;

void main() {
    TexCoords = (aPos.xy + texScale/2) / texScale;
    vec3 aNormal = texture(norm_texture, TexCoords).rgb;
    Normal = normalize(mat3(transpose(inverse(model))) * aNormal);

    FragPos = vec3(model * vec4(aPos, 1.0));
    gl_Position = proj * view * vec4(FragPos, 1.0);
}
"""
