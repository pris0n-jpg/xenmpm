from typing import List, Set, Iterable, Sequence
from OpenGL import GL as gl
from math import radians, cos, sin, tan, sqrt
from .functions import ReadWriteLock
from .transform3d import Matrix4x4, Quaternion, Vector3
from .GLGraphicsItem import GLGraphicsItem
from .items.camera import Camera
from .items.render import RenderGroup
from .items.MeshData import make_color

from qtpy.QtCore import Qt, Signal
from qtpy import QtCore, QtGui, API
from qtpy.QtQuick import QQuickFramebufferObject, QQuickItem

if API == "pyside6":  # pyside6 需要设置图形 API 为 OpenGL
    from qtpy.QtQuick import QQuickWindow
    from qtpy.QtQuick import QSGRendererInterface
    QQuickWindow.setGraphicsApi(QSGRendererInterface.GraphicsApi.OpenGL)


class QQuickGLViewRenderer(QQuickFramebufferObject.Renderer):
    """
    QQuickGLViewRenderer 类, 用于渲染 QQuickGLView
    """

    def __init__(self, widget: "QQuickGLView"):
        super().__init__()
        self.widget: QQuickGLView = widget
        self.context = QtGui.QOpenGLContext.currentContext()
        self.window = self.context.surface()

    def render(self):
        self.widget.rw_lock.acquire_read()
        # 设置背景颜色并清屏
        gl.glClearColor(*self.widget.bg_color)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # 更新光源和渲染物体
        for light in self.widget.light_group:
            light.renderShadow()

        # 绘制场景中的物体
        self.widget.light_group.render(self.widget.camera)
        self.widget.item_group.render(self.widget.camera)

        self.widget.rw_lock.release_read()

    def createFramebufferObject(self, size):
        format = QtGui.QOpenGLFramebufferObjectFormat()
        format.setAttachment(QtGui.QOpenGLFramebufferObject.Depth)
        format.setSamples(4)
        self.fbo = QtGui.QOpenGLFramebufferObject(size, format)
        return self.fbo


class QQuickGLView(QQuickFramebufferObject):

    render_finished = Signal()

    def __init__(
        self,
        bg_color = (0.7, 0.7, 0.7, 1),
        parent: QQuickItem = None
    ):
        """
        QQuickGLView 类, 用于在 QML 中显示 OpenGL 内容

        Parameters:
        - bg_color : tuple, optional, default: (0.7, 0.7, 0.7, 1), 背景颜色
        - parent : QQuickItem, optional, default: None
        """
        super(QQuickGLView, self).__init__(parent)
        self.setAcceptedMouseButtons(Qt.AllButtons)  #
        self.setMirrorVertically(True)

        self.bg_color = bg_color
        self._camera = Camera(eye=(0, 0, 10), center=(0, 0, 0), up=(0, 1, 0), fov=45)
        self.item_group = RenderGroup()  # 渲染物体组
        self.light_group = RenderGroup()  # 光源组

        self.last_pos = None
        self.press_pos = None
        self.rw_lock = ReadWriteLock()

        # renderer
        self.renderer_instance: QQuickGLViewRenderer = None

    def createRenderer(self):
        """
        创建 QQuickGLViewRenderer 实例
        """
        self.renderer_instance = QQuickGLViewRenderer(self)
        return self.renderer_instance

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def _(self, val):
        raise AttributeError("camera can not be set directly")

    def setBackgroundColor(self, color):
        """
        Set the background color of the widget. Accepts the same arguments as
        """
        self.bg_color = make_color(color)[0]

    #==============================================================================
    def devicePixelRatioF(self):
        # 通过 window() 获取 devicePixelRatio
        window = self.window()
        if window:
            return window.devicePixelRatio()
        return 1.0

    def deviceWidth(self):
        # 使用 devicePixelRatio 来计算宽度, 单位为像素
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr)

    def deviceHeight(self):
        # 使用 devicePixelRatio 来计算高度, 单位为像素
        dpr = self.devicePixelRatioF()
        return int(self.height() * dpr)

    def deviceRatio(self):
        # 计算设备的纵横比
        return self.height() / self.width()

    def getViewport(self):
        return (0, 0, self.deviceWidth(), self.deviceHeight())

    def geometryChanged(self, newGeometry, oldGeometry):
        """
        Called when the geometry of the item changes.
        """
        # 更新相机的投影矩阵
        self._camera.set_proj_matrix(aspect=self.width() / (self.height()+1))
        super(QQuickGLView, self).geometryChanged(newGeometry, oldGeometry)

    def geometryChange(self, newGeometry, oldGeometry):
        """
        Called when the geometry of the item changes.
        """
        self._camera.set_proj_matrix(aspect=self.width() / (self.height()+1))
        super(QQuickGLView, self).geometryChange(newGeometry, oldGeometry)
        pass

    def pixelSize(self, pos=Vector3(0, 0, 0)):
        """
        Return the approximate (y) size of a screen pixel at the location pos
        Pos may be a Vector or an (3,) array of locations
        """
        pos = self._camera.get_view_matrix() * pos  # convert to view coordinates
        fov = self._camera._fov
        return max(-pos[2], 0) * 2. * tan(0.5 * radians(fov)) / self.deviceHeight()

    # ====
    def addItem(self, item: GLGraphicsItem):
        """
        向场景中添加一个 item, 将 item tree 中所有的光源添加到 light_group 中, 设置所有 item 的 view 为当前视图

        Parameters:
        - item : GLGraphicsItem, _description_
        """
        self.rw_lock.acquire_write()

        self.item_group.addItem(item)

        for item in item.treeItems():
            item.setView(self)
            if len(item.lights) > 0:   # 添加光源
                self.light_group |= item.lights
                # for light in item.lights:
                #     light.setView(self)

        self.rw_lock.release_write()

    def addItems(self, items: Sequence[GLGraphicsItem]):
        for item in items:
            self.addItem(item)

    def removeItem(self, item):
        """
        从场景中移除一个 item, 设置 item 的 view 为 None
        """
        self.rw_lock.acquire_write()
        for it in item.treeItems():
            self.item_group.remove(it)
            it.setView(None)
            if hasattr(it, 'removeAllLight'):
                it.removeAllLight()
        self.rw_lock.release_write()

    def clear(self):
        """
        Remove all items from the scene.
        """
        self.rw_lock.acquire_write()
        for item in self.item_group.recursiveItems():
            item.setView(None)
        self.item_group.clear()
        self.rw_lock.release_write()

    # ==== Mouse Event
    def mousePressEvent(self, event):
        lpos = event.localPos()
        self.press_pos = lpos
        self.last_pos = lpos
        self.cam_pressed_matrix = self._camera.get_view_matrix()

    def mouseMoveEvent(self, event):
        ctrl_down = (event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift_down = (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        alt_down = (event.modifiers() & Qt.KeyboardModifier.AltModifier)
        lpos = event.localPos()

        cam_matrix = self._camera.get_view_matrix()
        diff = lpos - self.last_pos
        self.last_pos = lpos

        if shift_down and not alt_down:
            cam_matrix = self.cam_pressed_matrix
            diff = lpos - self.press_pos
            if abs(diff.x()) > abs(diff.y()):
                diff.setY(0)
            else:
                diff.setX(0)

        if ctrl_down:
            diff *= 0.1

        if alt_down:
            roll = diff.x() / 5

        if event.buttons() == Qt.MouseButton.LeftButton:
            if alt_down:
                self._camera.orbit(0, 0, roll, base=cam_matrix)
            else:
                self._camera.orbit(diff.y(), diff.x(), 0, base=cam_matrix)
        elif event.buttons() == Qt.MouseButton.MiddleButton:
            self._camera.pan(diff.x(), -diff.y(), 0, base=cam_matrix)
        # self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._camera._fov *= 0.999 ** delta
        else:
            self._camera.pan(0, 0, delta, scale=0.001)
        # self.update()

    def __enter__(self):
        # 设置当前 opengl 上下文
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 释放当前 opengl 上下文
        pass

    def resize(self, width, height):
        pass

def asQQuickGLView(cls: "GLViewWidget") -> QQuickGLView:
    """
    将一个 GLViewWidget 子类转换为 QQuickGLView 类的子类

    Parameters:
    - cls : GLViewWidget 子类
    """
    return type(
        "A." + cls.__name__,
        (QQuickGLView, ),
        {key: value for key, value in cls.__dict__.items()}
    )

