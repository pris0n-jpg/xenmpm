from xensesdk.omni.GLOmniItem import OmniSensorItem
from xensesdk.omni.GLFingerTipItem import GLFingerTipItem
from xensesdk.xenseInterface.XenseSensor import Sensor, SensorType
from xensesdk.ezgl.items import *
from xensesdk.ezgl import GLViewWidget, tb
from xensesdk.ezgl.utils.colormap import cm
import numpy as np


class InferView(GLViewWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.initUI()
        self.img_cap = Sensor.create(0, sensor_type=SensorType.Finger)
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
        ).rotate(90, 1, 0, 0)

        # -- model
        self.omni_sensor = GLFingerTipItem(lights=[self.light])
        self.arrow_plot = GLArrowPlotItem(tip_size=[0.003, 0.003], color=(1, 0.2, 0.2), parentItem=self.omni_sensor, width=5, glOptions="ontop")
        self.addItem(self.griditem)
        self.addItem(self.axis)
        self.addItem(self.omni_sensor)

    # @profile
    def onTimeout(self):
        [src_img, depth_map, diff_img] = self.img_cap.selectSensorInfo("src_img", "depth_map" ,"diff_img")
        # tb.get_widget("src").setData( np.ascontiguousarray(src_img) )

        tb.get_widget("marker").setData(diff_img)
        self.omni_sensor.set_depth(-depth_map/4)
        tb.get_widget("depth").setData(cm.jet(depth_map) * 255)

        self.update()

def main():
    import sys
    from qtpy import QtCore, QtWidgets

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)


    win = InferView(None)
    with tb.window("tb", size=(1100, 600)):
        with tb.group("view", horizontal=True, show=False):
            # src_img_view = tb.add_image_view("src" , None, img_size=(300, 500), img_format= "bgr")
            marker_img_view = tb.add_image_view("marker" , None, img_size=(300, 400), img_format= "bgr")
            depth_view = tb.add_image_view("depth" , None, img_size=(300, 400))
            tb.add_widget("3d", win)
    app.exec_()
    sys.exit()

if __name__ == '__main__':
    main()