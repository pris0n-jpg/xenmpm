"""
Description: Widgets for displaying the image and 3D model.

Author: Jin Liu
Date: 2024/09/30
"""

import cv2
import numpy as np
import sys
from qtpy.QtCore import Slot
from qtpy import QtCore, QtGui, QtWidgets

from xensesdk.ezgl import tb
from xensesdk.ezgl.utils.QtTools import QImageViewWidget, QSurfaceWidget
from xensesdk.ezgl import GLViewWidget
from xensesdk.ezgl.items import GLAxisItem, GLGridItem, PointLight
from .GLOmniItem import OmniSensorItem
from .GLWedgeSensorItem import GLWedgeSensorItem
from .GLFingerTipItem import GLFingerTipItem
from xensesdk.xenseInterface.XenseSensor import Sensor
from xensesdk.xenseInterface.sensorEnum import SensorType
from xensesdk.ezgl.utils.colormap import cm
from xensesdk import MACHINE


class ExampleView(GLViewWidget):
    def __scale_grid(self, grid):
        grid = grid.astype(np.float32) / (400, 700)
        grid = grid * [0.94, 0.93] + [0.03, 0.003]
        return grid

    def __init__(self, sensor, *args):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        self.app = QtWidgets.QApplication(sys.argv)
        
        super().__init__(parent=None)
        self.sensor = sensor
        self.initUI()
        tb.add_timer("timer", 10, self.onTimeout, self)

    def initUI(self):
        self.camera.set_vector6d((0.3, -1.4, -11, 30, -67, -121))
        self.axis = GLAxisItem(size=(3, 3, 3))
        self.axis.setVisible(0)
        # -- lights
        self.light = PointLight(pos=(5, 10, 10), ambient=(0.6, 0.6, 0.6), diffuse=(0.6, 0.6, 0.6),
                               visible=False, directional=True, render_shadow=False)
        self.light.setShadow(-3, 3, -3, 3, -3, 3, bias=0.05)

        # -- grid
        self.griditem = GLGridItem(
            size=(11, 11), spacing=(0.5, 0.5), lineWidth=1, color=np.array([0.78, 0.71, 0.60])*1.5,
            lineColor=(0.4, 0.3, 0.2), lights=[self.light]
        ).rotate(90, 1, 0, 0)

        # -- model
        if self.sensor.sensor_type == SensorType.Omni:
            self.omni_sensor = OmniSensorItem(lights=[self.light], grid_size=(100, 200))
        elif self.sensor.sensor_type == SensorType.VecTouch:
            self.camera.set_vector6d([-0.135, -1.117, -4.151, -26.246, -0.937, 0.052])
            self.omni_sensor = GLWedgeSensorItem(lights=[self.light], grid_size=(110, 200))
            self.griditem.translate(0,0,-2.1)
        elif self.sensor.sensor_type in [SensorType.OmniB]:
            self.omni_sensor = OmniSensorItem(lights=[self.light])
        elif self.sensor.sensor_type == SensorType.Finger:
            self.omni_sensor =  GLFingerTipItem(lights=[self.light]).rotate(90,0,0,1).rotate(90,0,1,0).translate(0,0,0.1)
        else:
            raise Exception(f"No sensor type {self.sensor.sensor_type}!")
        if MACHINE != "aarch64":
            self.addItem(self.griditem)
            self.addItem(self.axis)
        self.addItem(self.omni_sensor)
    
    def setCallback(self, function):
        self._callback_func = function

    # @profile
    def onTimeout(self):
        self._callback_func()
        self.update()
    
    def setMarkerFlow(self, marker_init, marker3d):
        self.omni_sensor.set_3d_arrow(marker_init / 10, marker3d / 10)
    
    def setMarkerUnorder(self, marker_unordered):
        physical_marker = marker_unordered.copy()
        physical_marker[...,0] = 400 - physical_marker[...,0]
        marker_curr = self.omni_sensor.get_3d_marker(self.__scale_grid(physical_marker))
        self.omni_sensor.set_3d_marker(marker_curr)

    def setForceFlow(self, force, res_force, mesh_init):
        F_len = np.linalg.norm(force, axis=2)*4
        color = cm.yellow_red(F_len)
        force[F_len < 0.05] = 0  # 滤掉杂力

        if isinstance(self.omni_sensor, GLWedgeSensorItem):
            self.omni_sensor.set_force(res_force[:3], res_force[3:])
            force[..., 2] = -force[..., 2] * 2
            self.omni_sensor.set_3d_arrow(mesh_init[1:-1, 1:-1]/10, mesh_init[1:-1, 1:-1]/10 + force[1:-1, 1:-1], color[1:-1, 1:-1])

        if isinstance(self.omni_sensor, OmniSensorItem):
            self.omni_sensor.set_force(res_force[:3], res_force[3:])
            force = self.omni_sensor.process_force_for_show(force)
            self.omni_sensor.set_3d_arrow(mesh_init[1:-1, 1:-1]/10, mesh_init[1:-1, 1:-1]/10 + force[1:-1, 1:-1], color[1:-1, 1:-1])

        if isinstance(self.omni_sensor, GLFingerTipItem):
            self.omni_sensor.set_force(res_force[:3], res_force[3:])
            force = self.omni_sensor.process_force_for_show(force)
            self.omni_sensor.set_3d_arrow(mesh_init[1:-1, 1:-1]/10, mesh_init[1:-1, 1:-1]/10 + force[1:-1, 1:-1], color[1:-1, 1:-1])
            
    def setMarker(self, marker3d):
        self.omni_sensor.set_3d_marker((marker3d).reshape(-1, 3) / 10)
        
    def setDepth(self, depth):
        self.omni_sensor.set_depth(depth)
    
    def create2d(self, *args):
        self._view2d = self.View2d(*args, parent=self)
        return self._view2d

    def show(self):
        self.app.exec_()
        
    class View2d():
        def __init__(self, *args, parent):
            assert all(isinstance(arg, Sensor.OutputType) for arg in args), "All arguments must be of type {Sensor.OutputType}"
            self.winlist = dict()
            with tb.window("tb", size=(1100, 600)):
                with tb.group("view", horizontal=True, show=False):
                    for name in args:
                        self.winlist[name] = tb.add_image_view(name.name , None, img_size=(300, 500), img_format="bgr")
                    tb.add_widget("3d_view",parent)

        def setData(self, name, img):
            if name == Sensor.OutputType.Depth:
                img = cv2.cvtColor(cm.jet(img), cv2.COLOR_RGB2BGR)
                self.winlist[name].setData(img * 255)
            else:
                self.winlist[name].setData(img)


        

class GLOmniWidget(GLViewWidget):

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.camera.set_vector6d((0.3, -1.4, -11, 30, -67, -121))

        self.axis = GLAxisItem(size=(3, 3, 3))

        # -- lights
        self.light = PointLight(pos=(5, 10, 10), ambient=(0.6, 0.6, 0.6), diffuse=(0.6, 0.6, 0.6),
                               visible=False, directional=True, render_shadow=False)
        self.light.setShadow(-3, 3, -3, 3, -3, 3, bias=0.05)

        # -- grid
        self.grid = GLGridItem(
            size=(11, 11), spacing=(0.5, 0.5), lineWidth=1, color=np.array([0.78, 0.71, 0.60])*1.5,
            lineColor=(0.4, 0.3, 0.2), lights=[self.light]
        ).rotate(90, 1, 0, 0)

        # -- model
        self.sensor = OmniSensorItem(lights=[self.light])

        self.addItem(self.grid)
        self.addItem(self.axis)
        self.addItem(self.sensor)


class GLFingerTipWidget(GLViewWidget):

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.camera.set_vector6d((0.18, -0.725, -4.76, 11.8, -14.6, -3.5))

        self.axis = GLAxisItem(size=(3, 3, 3))
        self.ax_fixed = GLAxisItem(fix_to_corner=True)

        # -- lights
        self.light = PointLight(pos=(2, 3, 2), ambient=(0.5, 0.5, 0.5), diffuse=(0.6, 0.6, 0.6), specular=(0.14, 0.14, 0.14),
                               visible=False, directional=True, render_shadow=False)
        self.light.setShadow(-3, 3, -3, 3, -3, 3, bias=0.05)

        # -- grid
        self.grid = GLGridItem(
            size=(11, 11), spacing=(0.5, 0.5), lineWidth=1, color=np.array([0.78, 0.71, 0.60])*2.,
            lineColor=(0.4, 0.3, 0.2), lights=[self.light]
        ).translate(0, -0.5, -0)

        # -- model
        self.sensor = GLFingerTipItem(lights=[self.light], grid_size=(30, 40))

        self.addItems([self.sensor, self.axis, self.ax_fixed, self.grid])


class SensorVisWidget(QtWidgets.QFrame):
    def __init__(
        self,
        parent,
        sensor_type="omni",
        surface_view=True,
    ):
        super().__init__(parent)
        self.init_ui(sensor_type, surface_view)
        self.setData = self.set_data
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def init_ui(self, sensor_type, surface_view):
        self.vbox = QtWidgets.QVBoxLayout(self)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(0)

        self.tab_widget = QtWidgets.QTabWidget(self)
        self.vbox.addWidget(self.tab_widget)

        self.image_view = QImageViewWidget(self, auto_scale=True)
        self.tab_widget.addTab(self.image_view, "Tactile Imprint")  # 0

        self.gl_view = GLOmniWidget(self) if sensor_type == "omni" else GLFingerTipWidget(self)
        self.tab_widget.addTab(self.gl_view, "3D")  # 1
        self.tab_widget.setCurrentIndex(1)

        if surface_view:
            self.surface_view = QSurfaceWidget(self)
            self.tab_widget.addTab(self.surface_view, "surface")  # 2

        with tb.window("colormap", None, frameless=False):
            with tb.group("horizontal1", show=False, horizontal=True):
                with tb.group("Vertical1", show=False, horizontal=False):
                    tb.add_combo(" channel", ["B", "G", "R"])
                    tb.add_combo("colormap", ["coolwarm", "viridis", "jet"])
                tb.add_spacer(30, True)
                drag_array = tb.add_drag_array("scale_bias", value=[130, 0], min_val=[1, -0.5], max_val=[250, 0.5],
                                               step=[1, 0.01], decimals=[0, 2], format=["scale: %d", "bias: %.2f"], horizontal=False, show_label=False)

        self.color_map = tb.get_window("colormap")
        self.color_map.setFixedHeight(80)
        self.color_map.setVisible(False)
        self.vbox.addWidget(self.color_map)

    def set_data(self, img=None, tab_index=None):
        if tab_index is not None:
            self.tab_widget.setCurrentIndex(tab_index)

        tab_name = self.tab_widget.tabText(self.tab_widget.currentIndex())
        if tab_name == "Tactile Imprint" and img is not None:
            self.tab_widget.currentWidget().setData(img)

        elif tab_name == "surface" and img is not None:
            while (img.shape[1] > 400 or img.shape[0] > 400):
                img = cv2.pyrDown(img) #TODO / 2
            img = cv2.GaussianBlur(img, (5, 5), 2)

            channel = tb.get_value(" channel")
            channel = ["B", "G", "R"].index(channel)
            colormap = tb.get_value("colormap")
            scale, bias = tb.get_value("scale_bias")
            self.tab_widget.currentWidget().setData(img, channel, colormap, scale, bias)

        elif tab_name == "3D" and img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.gl_view.sensor.set_image(img)
            if tb.get_value("enable depth"):
                self.gl_view.sensor.set_depth( self.gl_view.sensor.contact_region(img))
            self.gl_view.update()

    @Slot()
    def on_tab_changed(self):
        idx = self.tab_widget.currentIndex()
        tab_name = self.tab_widget.tabText(idx)
        if tab_name == "surface":
            self.color_map.setVisible(True)
        else:
            self.color_map.setVisible(False)