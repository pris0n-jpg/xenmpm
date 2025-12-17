import cv2
import numpy as np
from typing import List, Set, Iterable, Sequence
from OpenGL import GL as gl
from math import radians, cos, sin, tan, sqrt
from qtpy import QtCore, QtWidgets, QtGui
from .items.camera import Camera
from .functions import get_path, ReadWriteLock
from .transform3d import Matrix4x4, Quaternion, Vector3
from .GLGraphicsItem import GLGraphicsItem
from .items.MeshData import make_color
from .items.render import RenderGroup

class GLViewWidget(QtWidgets.QOpenGLWidget):

    def __init__(
        self,
        bg_color = (0.7, 0.7, 0.7, 1),
        parent=None,
    ):
        """
        Basic widget for displaying 3D data
          - Rotation/scale controls
        """
        QtWidgets.QOpenGLWidget.__init__(self, parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self.bg_color = bg_color
        self._camera = Camera(eye=(0,0,10), center=(0,0,0), up=(0,1,0), fov=45)
        self.item_group = RenderGroup()  # 渲染物体组
        self.light_group = RenderGroup()  # 光源组

        self.last_pos = None
        self.press_pos = None
        self.rw_lock = ReadWriteLock()

        # 设置多重采样抗锯齿
        format = QtGui.QSurfaceFormat()
        format.setSamples(4)
        self.setFormat(format)
        self.setWindowIcon(QtGui.QIcon(str(get_path()/"resources/textures/triangle.png")))

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def _(self, val):
        raise AttributeError("camera can not be set directly")

    def deviceWidth(self):
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr)

    def deviceHeight(self):
        dpr = self.devicePixelRatioF()
        return int(self.height() * dpr)

    def deviceRatio(self):
        return self.height() / self.width()

    def getViewport(self):
        return (0, 0, self.deviceWidth(), self.deviceHeight())

    def resizeEvent(self, ev):
        # 设置相机的投影矩阵
        self._camera.set_proj_matrix(aspect=self.width() / (self.height() + 1))
        super().resizeEvent(ev)

    def reset(self):
        self._camera.lookAt((0, 0, 10), (0, 0, 0), (0, 1, 0), 45)

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

    def setBackgroundColor(self, color):
        """
        Set the background color of the widget. Accepts the same arguments as
        """
        self.bg_color = make_color(color)[0]

    # def initializeGL(self):
    #     """初始化 OpenGL, 在初始化之后自动调用 1 次"""
    #     self.item_group |= self.light_group

    def paintGL(self):
        self.rw_lock.acquire_read()
        gl.glClearColor(*self.bg_color)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glClear( gl.GL_DEPTH_BUFFER_BIT | gl.GL_COLOR_BUFFER_BIT )

        # 更新阴影贴图
        for light in self.light_group:
            light.renderShadow()

        # draw items
        self.light_group.render(self._camera)
        self.item_group.render(self._camera)

        self.rw_lock.release_read()

    def pixelSize(self, pos=Vector3(0, 0, 0)):
        """
        Return the approximate (y) size of a screen pixel at the location pos
        Pos may be a Vector or an (3,) array of locations
        """
        pos = self._camera.get_view_matrix() * pos  # convert to view coordinates
        fov = self._camera._fov
        return max(-pos[2], 0) * 2. * tan(0.5 * radians(fov)) / self.deviceHeight()

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.press_pos = lpos
        self.last_pos = lpos
        self.cam_pressed_matrix = self._camera.get_view_matrix()

    def mouseMoveEvent(self, ev):
        ctrl_down = (ev.modifiers() & QtCore.Qt.ControlModifier)
        shift_down = (ev.modifiers() & QtCore.Qt.ShiftModifier)
        alt_down = (ev.modifiers() & QtCore.Qt.AltModifier)
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()

        cam_matrix = self._camera.get_view_matrix()
        diff = lpos - self.last_pos
        self.last_pos = lpos

        if shift_down and not alt_down:  # 锁定水平或垂直转动
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

        if ev.buttons() == QtCore.Qt.LeftButton:
            if alt_down:
                self._camera.orbit(0, 0, roll, base=cam_matrix)
            else:
                self._camera.orbit(diff.y(), diff.x(), 0, base=cam_matrix)
        elif ev.buttons() == QtCore.Qt.MiddleButton:
            self._camera.pan(diff.x(), -diff.y(), 0, base=cam_matrix)
        # self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().x()
        if delta == 0:
            delta = ev.angleDelta().y()
        if (ev.modifiers() & QtCore.Qt.ControlModifier):
            self._camera._fov *= 0.999**delta
        else:
            self._camera.pan(0, 0, delta, scale=0.001)
        # self.update()

    def readQImage(self):
        """
        Read the current buffer pixels out as a QImage.
        """
        return self.grabFramebuffer()

    def readImage(self):
        """
        Read the current buffer pixels out as a cv2 Image.
        """
        qimage = self.grabFramebuffer()
        w, h = self.width(), self.height()
        bytes = qimage.bits().asstring(qimage.byteCount())
        img = np.frombuffer(bytes, np.uint8).reshape((h, w, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def isCurrent(self):
        """
        Return True if this GLWidget's context is current.
        """
        return self.context() == QtGui.QOpenGLContext.currentContext()

    def __enter__(self):
        self.makeCurrent()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An exception occurred: {exc_type}: {exc_value}")
        self.doneCurrent()

    def keyPressEvent(self, a0) -> None:
        """按键处理"""
        if a0.text() == '1':
            print("camera: ", np.array2string(self._camera.get_vector6d(), separator=', ', precision=3, floatmode='fixed'))
        elif a0.text() == '2':
            self._camera.set_vector6d([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLViewWidget(None)
    win.show()
    sys.exit(app.exec_())