import numpy as np
import OpenGL.GL as gl
from typing import List, Sequence

from ..GLGraphicsItem import GLGraphicsItem

__all__ = ['GLGroupItem']


class GLGroupItem(GLGraphicsItem):
    """
    Container for multiple GLGraphicsItem.
    """
    def __init__(
        self,
        items: Sequence[GLGraphicsItem] = None,
        parentItem: GLGraphicsItem = None,
        depthValue: int = 0,
    ):
        super().__init__(parentItem=parentItem, depthValue=depthValue)

        self.items = dict()

        if items is not None:
            self.addItem(*items)

    def addItem(self, *items: Sequence[GLGraphicsItem]):
        for item in items:
            if item is None:
                continue
            super().addChildItem(item)
            self.items[item.label] = item

    def getItem(self, label: int):
        return self.items[label]