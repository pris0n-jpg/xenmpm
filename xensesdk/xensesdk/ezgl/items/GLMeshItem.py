import OpenGL.GL as gl
import numpy as np
from .shader import Shader
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .MeshData import vertex_normal, Mesh
from .light import LightMixin, light_fragment_shader
from .camera import Camera
from typing import Sequence

__all__ = ['GLMeshItem']


class GLMeshItem(GLGraphicsItem, LightMixin):

    def __init__(
        self,
        vertexes = None,
        indices = None,
        normals = None,
        texcoords = None,
        lights: Sequence = list(),
        material = None,
        calc_normals = True,
        mode = gl.GL_TRIANGLES,  # gl.GL_TRIANGLES, gl.GL_QUADS
        usage = gl.GL_STATIC_DRAW,
        show_edge = False,
        mesh : Mesh = None,
        glOptions = 'opaque',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self._show_edge = show_edge
        if mesh is not None:
            self._mesh = mesh
        else:
            self._mesh = Mesh(vertexes, indices, texcoords, normals,
                          material, None, usage, calc_normals, mode=mode)
        self.addLight(lights)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, light_fragment_shader)
        self._mesh.initializeGL()

    def updateGL(self):
        self._mesh.vbo.commit()

    def setData(self, vertexes=None, normals=None, indices=None, texcoords=None):
        if vertexes is not None:
            self._mesh._vertexes.set_data(vertexes)
        if normals is not None:
            self._mesh._normals.set_data(normals)
        if indices is not None:
            self._mesh._indices.set_data(indices)
        if texcoords is not None:
            self._mesh._texcoords.set_data(texcoords)

    def paint(self, camera: Camera):
        self.updateGL()
        self.setupGLState()

        with self.shader:
            self.setupLight(self.shader)
            self.shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", self.model_matrix().glData, "mat4")
            self.shader.set_uniform("ViewPos",camera.get_eye(), "vec3")
            self._mesh.paint(self.shader)

            if self._show_edge:
                self.shader.set_uniform("material.disable", True, "bool")
                gl.glDisableVertexAttribArray(1)
                gl.glVertexAttrib4fv(3, np.array([0, 0, 0, 1], dtype=np.float32))
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glPolygonOffset(-1., -1.)
                gl.glDepthMask(gl.GL_FALSE)

                self._mesh.paint(self.shader)

                self.shader.set_uniform("material.disable", False, "bool")
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)
                gl.glDepthMask(gl.GL_TRUE)

    def paintWithShader(self, camera: Camera, shader: Shader, **kwargs):
        self.updateGL()
        self.setupGLState()
        with shader:
            shader.set_uniform("view", camera.get_view_matrix().glData, "mat4")
            shader.set_uniform("proj", camera.get_proj_matrix().glData, "mat4")
            shader.set_uniform("model", self.model_matrix().glData, "mat4")
            self._mesh.paintShadow()

    def setMaterial(self, material):
        self._mesh.setMaterial(material)

    def getMaterial(self):
        return self._mesh.getMaterial()

    def setMaterialData(self, **kwargs):
        self._mesh._material.set_data(**kwargs)

vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec4 aColor;

out vec2 TexCoords;
out vec3 FragPos;
out vec3 Normal;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;
out vec4 oColor;

void main() {
    oColor = aColor;
    TexCoords = aTexCoords;
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
    gl_Position = proj * view * vec4(FragPos, 1.0);
}
"""
