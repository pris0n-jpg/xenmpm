from xensesdk.omni.GLWedgeSensorItem import GLWedgeSensorItem
from xensesdk.xenseInterface.XenseSensor import Sensor
from xensesdk.ezgl.items import *
from xensesdk.ezgl import GLViewWidget, tb
import numpy as np
import cv2
from xensesdk.ezgl.utils.colormap import cm


class InferView(GLViewWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.initUI()
        self.img_cap = Sensor.create(0, config_path= r"/home/linux/xensesdk/xensesdk/examples/config_0.2.1/W0")
        tb.add_timer("timer", 10, self.onTimeout, self)
    
    def initUI(self):
        self.camera.set_vector6d((0.3, -1.4, -11, 30, -67, -121))
        self.axis = GLAxisItem(size=(3, 3, 3))
        # -- lights
        self.light = PointLight(pos=(5, 10, 10), ambient=(0.6, 0.6, 0.6), diffuse=(0.6, 0.6, 0.6),
                               visible=False, directional=True, render_shadow=True)
        self.light.setShadow(-3, 3, -3, 3, -3, 3, bias=0.05)

        # -- grid
        self.griditem = GLGridItem(
            size=(11, 11), spacing=(0.5, 0.5), lineWidth=1, color=np.array([0.78, 0.71, 0.60])*1.5,
            lineColor=(0.4, 0.3, 0.2), lights=[self.light]
        ).rotate(90, 1, 0, 0).translate(0,0,-1)

        # -- model
        self.omni_sensor = GLWedgeSensorItem(lights=[self.light])
        self.addItem(self.griditem)
        self.addItem(self.axis)
        self.addItem(self.omni_sensor)

        with tb.window("tb", size=(300, 600)):
            with tb.group("view", horizontal=True, collapsible=True):
                self.src_img_view = tb.add_image_view("src" , None, img_size=(300, 500), img_format= "bgr")
                self.depth_view = tb.add_image_view("" , None, img_size=(300, 500))
                self.marker_img_view = tb.add_image_view("marker" , None, img_size=(300, 500), img_format= "bgr")

    def onTimeout(self):
        force, depth_map, marker3d, src_img, mesh_init = self.img_cap.selectSensorInfo(Sensor.OutputType.Force,
                                                    Sensor.OutputType.Depth,
                                                    Sensor.OutputType.Marker3D,
                                                    Sensor.OutputType.Rectify,
                                                    Sensor.OutputType.Mesh3DInit)

        self.src_img_view.setData( np.ascontiguousarray(src_img) )
        marker_img = self.img_cap.drawMarkerMove(src_img)
        self.marker_img_view.setData(marker_img)

        self.omni_sensor.set_depth(np.flip(-depth_map/6, 1))
        self.depth_view.setData(depth_map * 200)

        F_len = (1 - np.linalg.norm(force, axis=2) * 8)
        color = cm.autumn(F_len)
        self.omni_sensor.set_3d_arrow(mesh_init/10, mesh_init/10 - force*1.5, color)
        self.omni_sensor.set_3d_marker((marker3d).reshape(-1, 3) / 10)
        self.update()

def main():
    import sys
    from qtpy import QtCore, QtWidgets

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    win = InferView(None)
    win.show()
    app.exec_()
    sys.exit()

if __name__ == '__main__':
    main()