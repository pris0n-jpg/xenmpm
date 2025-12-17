"""
Description: 渲染组, 用于管理一组渲染对象

Author: Jin Liu
Date: 2024/06/26
"""

from ..GLGraphicsItem import GLGraphicsItem
from .shader import Shader
from .camera import Camera
from ..functions import SortedSet, printExc
from typing import Iterable, List, Sequence


class RenderGroup(SortedSet):

    def __init__(self, iterable=None, *args):
        """
        初始化, 渲染组是一个按照渲染深度排序的有序集合, 渲染深度值越大的物体越晚渲染
        """
        super().__init__(iterable, key_func=lambda a: a.depthValue())

    def __repr__(self):
        return f"RenderGroup({self.items})"

    def addItem(self, *item: Sequence[GLGraphicsItem]):
        """
        添加渲染对象
        """
        for item in item:
            self.add(item)

    def removeItem(self, *item: Sequence[GLGraphicsItem]):
        """
        移除一个渲染对象
        """
        for item in item:
            self.remove(item)

    def recursiveItems(self) -> Iterable[GLGraphicsItem]:
        """
        递归获取所有渲染对象
        """
        for item in self.items:
            yield item
            yield from item.recursiveChildItems()

    def render(
        self,
        camera: Camera,
        shader: Shader=None,
        **kwargs
    ):
        """
        渲染 render_group 中的所有对象

        Parameters:
        - render_group : RenderGroup, 渲染组
        - camera : Camera, 相机
        - shader : Shader, 外部传入的 shader, 只支持重载了 paintWithShader 的 item
        - **kwargs : dict, 传递给 paintWithShader 的参数
        """
        for item in self.recursiveItems():
            if not item.visible() :
                continue
            elif not item.isInitialized:
                item.initialize()
            item.update_model_matrix()

            try:
                if shader is None:
                    item.paint(camera)
                else:
                    item.paintWithShader(camera, shader, **kwargs)
            except:
                printExc()
                print(f"Error while drawing item: {item}, label: {item.label}, shader: {shader}.")



