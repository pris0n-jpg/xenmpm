from typing import Sequence
import numpy as np
import OpenGL.GL as gl
from ..GLGraphicsItem import GLGraphicsItem
from ..functions import get_path
from .GLModelItem import GLModelItem
from .GLSurfacePlotItem import GLSurfacePlotItem
from .GLScatterPlotItem import GLScatterPlotItem
from .MeshData import Material


__all__ = ['GLGelSlimItem']


class GLGelSlimItem(GLGraphicsItem):
    """ Displays a GelSlim model with a surface plot on top of it."""

    def __init__(
        self,
        lights: Sequence,
        parentItem=None,
    ):
        super().__init__(parentItem=parentItem)

        self.gelslim_base = GLModelItem(
            path = get_path() / "resources/objects/GelSlim_obj/GelSlim.obj",
            lights = lights,
            glOptions = "translucent_cull",
            parentItem = self,
        )
        self.gelslim_base.scale(0.1, 0.1, 0.1)
        self.gelslim_base.setPaintOrder([1, 0])
        self.gelslim_base.setDepthValue(0)
        _texcoord = self.gelslim_base.meshes[1]._texcoords.data
        self.gelslim_base.meshes[1]._texcoords.set_data(_texcoord / 10)

        self.gelslim_gel = GLSurfacePlotItem(
            zmap = np.zeros((30, 40), dtype=np.float32),
            x_size = 1.35,  # 22.4 mm, 640 px
            lights = lights,
            glOptions = "translucent_cull",
            parentItem = None,
        )
        self.gelslim_gel.rotate(90, 0, 0, 1)
        self.gelslim_gel.setMaterial(self.gelslim_base.getMaterial(0))
        self.gelslim_gel.setDepthValue(1)

        self.marker = GLScatterPlotItem(
            size=0.8, color=(0, 0, 0, 1), glOptions="ontop"
        ).rotate(90, 0, 0, 1).scale(1.35/20.36, 1.35/20.36, 1.35/20.36) # mm to gl unit
        self.marker.setDepthValue(2)

        self.addChildItem(self.gelslim_gel)
        self.addChildItem(self.marker)

    def setDepth(self, zmap=None, marker=None):
        if zmap is not None:
            self.gelslim_gel.setData(zmap)
        if marker is not None:
            self.marker.setData(marker)

    def setMaterial(self, material: Material):
        self.gelslim_gel.setMaterial(material)
        self.gelslim_base.setMaterial(0, material)