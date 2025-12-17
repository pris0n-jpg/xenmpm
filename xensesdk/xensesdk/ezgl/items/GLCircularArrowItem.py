import OpenGL.GL as gl
import numpy as np
from ..transform3d import Matrix4x4
from .MeshData import cone, cylinder, vertex_normal
from .GLMeshItem import GLMeshItem


__all__ = ['GLCircularArrowItem']


class GLCircularArrowItem(GLMeshItem):
    """
    Displays Circular Arrow.
    """

    def __init__(
        self,
        length = 1,  # 0~1
        inner_radius = 0.1,  # radius, height
        outer_radius = 1,
        rings = 20,  # 环数
        sides = 10,  # 每个环的边数
        color = [.6, .2, .2],
        lights = list(),
        glOptions='opaque',
        parentItem=None
    ):
        self._length = length
        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._cyli_vert, cylinder_ind = cylinder([1, 1], 1, rings, sides)
        self._tip_vert, tip_ind = cone(1, 1, sides)
        self._n_cyli = len(self._cyli_vert)
        self._n_tip = len(self._tip_vert)
        indices = np.concatenate([cylinder_ind, tip_ind + self._n_cyli], axis=0)

        super().__init__(indices=indices, lights=lights, calc_normals=False, usage=gl.GL_DYNAMIC_DRAW, glOptions=glOptions, parentItem=parentItem)
        self.setData(length, color, inner_radius, outer_radius)        

    def setData(self, length=None, color=None, inner_radius=None, outer_radius=None):
        if color is not None: # len == 3
            self._mesh._material.diffuse = np.array(color).reshape(3) * 0.6
            self._mesh._material.ambient = np.array(color).reshape(3) * 0.4
            self._mesh._material.specular = np.array(color).reshape(3) * 0.2
        
        self._length = length if length is not None else self._length
        self._inner_radius = inner_radius if inner_radius else self._inner_radius
        self._outer_radius = outer_radius if outer_radius else self._outer_radius
                
        if length is not None or inner_radius or outer_radius:
            rot = np.clip(self._length, 0, 1) * 2 * np.pi
            verts = np.zeros((self._n_cyli + self._n_tip, 3), dtype=np.float32)
            verts[:self._n_cyli, 0] = self._cyli_vert[:, 0] * self._inner_radius
            verts[:self._n_cyli, 1] = (self._outer_radius + self._cyli_vert[:, 1]*self._inner_radius) * np.cos(self._cyli_vert[:, 2] * rot)
            verts[:self._n_cyli, 2] = (self._outer_radius + self._cyli_vert[:, 1]*self._inner_radius) * np.sin(self._cyli_vert[:, 2] * rot)

            tip_pos = [0, self._outer_radius*np.cos(rot), self._outer_radius*np.sin(rot)]
            tf = Matrix4x4.fromVector6d(*tip_pos, rot*180/np.pi, 0 , 0).scale(self._inner_radius*2, self._inner_radius*2, self._inner_radius*1.5)
            verts[self._n_cyli:, :] = tf * self._tip_vert
            normal = vertex_normal(verts, self._mesh._indices.data) if length else None           
            super().setData(vertexes=verts, normals=normal)