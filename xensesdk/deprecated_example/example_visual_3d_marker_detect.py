from xensesdk.omni.GLWedgeSensorItem import GLWedgeSensorItem
from xensesdk.xenseInterface.XenseSensor import Sensor
from xensesdk.ezgl.items import *
from xensesdk.ezgl import GLViewWidget, tb
from xensesdk.ezgl.utils.colormap import cm
import numpy as np

class InferView(GLViewWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.initUI()
        self.img_cap = Sensor.create(0, config_path= r"/home/linux/xensesdk/xensesdk/examples/config_0.2.1/W0")
        tb.add_timer("timer", 1, self.onTimeout, self)

    def __scale_grid(self, grid):
        grid = grid.astype(np.float32) / (400, 700)
        grid = grid * [0.94, 0.93] + [0.03, 0.003]
        return grid
    
    def initUI(self):
        self.camera.set_vector6d((0.3, -1.4, -11, 30, -67, -121))

        self.axis = GLAxisItem(size=(3, 3, 3))

        # -- lights
        self.light = PointLight(pos=(5, 10, 10), ambient=(0.6, 0.6, 0.6), diffuse=(0.6, 0.6, 0.6),
                               visible=False, directional=True, render_shadow=False)
        self.light.setShadow(-3, 3, -3, 3, -3, 3, bias=0.05)

        # -- grid
        self.griditem = GLGridItem(
            size=(11, 11), spacing=(0.5, 0.5), lineWidth=1, color=np.array([0.78, 0.71, 0.60])*1.5,
            lineColor=(0.4, 0.3, 0.2), lights=[self.light]
        ).rotate(90, 1, 0, 0).translate(0,0,-1)

        # -- model
        self.omni_sensor = GLWedgeSensorItem(lights=[self.light])
        self.arrow_plot = GLArrowPlotItem(tip_size=[0.003, 0.003], color=(1, 0.2, 0.2), parentItem=self.omni_sensor, width=5, glOptions="ontop")
        self.addItem(self.griditem)
        self.addItem(self.axis)
        self.addItem(self.omni_sensor)


            # self.surf = VisualizeWidget(None)
            # self.surf.show()


    # @profile
    def onTimeout(self):
        src_img, marker_unordered, depth_map = self.img_cap.selectSensorInfo(Sensor.OutputType.Rectify, Sensor.OutputType.MarkerUnorder ,Sensor.OutputType.Depth)

        marker_img = self.img_cap.drawMarker(src_img, marker_unordered)
        tb.get_widget("marker").setData(marker_img)
        self.omni_sensor.set_depth(np.flip(-depth_map/6, 1))
        tb.get_widget("depth").setData(cm.jet(depth_map) * 255)
        physical_marker = marker_unordered.copy()
        physical_marker[...,0] = 400 - physical_marker[...,0]
        marker_curr = self.omni_sensor.get_3d_marker(self.__scale_grid(physical_marker))
        self.omni_sensor.set_3d_marker(marker_curr)
        self.update()

def main():
    import sys
    from qtpy import QtCore, QtWidgets

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    win = InferView(None)
    with tb.window("tb", size=(1100, 400)):
        with tb.group("view", horizontal=True, show=False):
            # src_img_view = tb.add_image_view("src" , None, img_size=(300, 500), img_format= "bgr")
            depth_view = tb.add_image_view("depth" , None, img_size=(300, 500))
            marker_img_view = tb.add_image_view("marker" , None, img_size=(300, 500), img_format= "bgr")
            tb.add_widget("3d", win)

    app.exec_()
    sys.exit()

if __name__ == '__main__':
    main()