from xensesdk.omni.GLOmniItem import OmniSensorItem
from xensesdk.xenseInterface.XenseSensor import Sensor
from xensesdk.ezgl.items import *
from xensesdk.ezgl import GLViewWidget, tb
import numpy as np

class InferView(GLViewWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.initUI()
        self.img_cap_1 = Sensor.create(0, config_path=r'D:\gitlab\xensesdk\xensesdk\examples\config\F01')
        self.img_cap_2 = Sensor.create(3, config_path=r'D:\gitlab\xensesdk\xensesdk\examples\config\F02')
        self.img_cap_3 = Sensor.create(1, config_path=r'D:\gitlab\xensesdk\xensesdk\examples\config\F03')
        self.img_cap_4 = Sensor.create(2, config_path=r'D:\gitlab\xensesdk\xensesdk\examples\config\F04')
        tb.add_timer("timer", 10, self.onTimeout, self)
    
    def initUI(self):
        # self.camera.set_vector6d((0.3, -1.4, -11, 30, -67, -121))

        # self.axis = GLAxisItem(size=(3, 3, 3))

        # # -- lights
        # self.light = PointLight(pos=(5, 7, 10), ambient=(0.6, 0.6, 0.6), diffuse=(0.6, 0.6, 0.6),
        #                        visible=False, directional=True, render_shadow=False)
        # self.light.setShadow(-3, 3, -3, 3, -3, 3, bias=0.05)

        # # -- grid
        # self.griditem = GLGridItem(
        #     size=(11, 11), spacing=(0.5, 0.5), lineWidth=1, color=np.array([0.78, 0.71, 0.60])*1.5,
        #     lineColor=(0.4, 0.3, 0.2), lights=[self.light]
        # ).rotate(90, 1, 0, 0)

        # -- model
        # self.omni_sensor = OmniSensorItem(lights=[self.light])
        # self.arrow_plot = GLArrowPlotItem(tip_size=[0.003, 0.003], color=(1, 0.2, 0.2), parentItem=self.omni_sensor, width=5, glOptions="ontop")
        # self.addItem(self.griditem)
        # self.addItem(self.axis)
        # self.addItem(self.omni_sensor)

        with tb.window("tb", size=(300, 600)):
            with tb.group("view", horizontal=True, collapsible=True):
                self.src_img_view_1 = tb.add_image_view("src" , None, img_size=(300, 500), img_format= "bgr")
                self.src_img_view_2 = tb.add_image_view("src" , None, img_size=(300, 500), img_format= "bgr")
                self.src_img_view_3 = tb.add_image_view("src" , None, img_size=(300, 500), img_format= "bgr")
                self.src_img_view_4 = tb.add_image_view("src" , None, img_size=(300, 500), img_format= "bgr")
                self.depth_view_1 = tb.add_image_view("depth_1" , None, img_size=(300, 500))
                self.depth_view_2 = tb.add_image_view("depth_2" , None, img_size=(300, 500))
                self.depth_view_3 = tb.add_image_view("depth_3" , None, img_size=(300, 500))
                self.depth_view_4 = tb.add_image_view("depth_4" , None, img_size=(300, 500))


    # @profile
    def onTimeout(self):
        
        src_img_1, depth_map_1, _ = self.img_cap_1.selectSensorInfo(Sensor.OutputType.Rectify, Sensor.OutputType.Depth, Sensor.OutputType.Difference)
        src_img_2, depth_map_2, _ = self.img_cap_2.selectSensorInfo(Sensor.OutputType.Rectify, Sensor.OutputType.Depth, Sensor.OutputType.Difference)
        src_img_3, depth_map_3, _ = self.img_cap_3.selectSensorInfo(Sensor.OutputType.Rectify, Sensor.OutputType.Depth, Sensor.OutputType.Difference)
        src_img_4, depth_map_4, _ = self.img_cap_4.selectSensorInfo(Sensor.OutputType.Rectify, Sensor.OutputType.Depth, Sensor.OutputType.Difference)
        
        
        self.src_img_view_1.setData( np.ascontiguousarray(src_img_1) )
        self.src_img_view_2.setData( np.ascontiguousarray(src_img_2) )
        self.src_img_view_3.setData( np.ascontiguousarray(src_img_3) )
        self.src_img_view_4.setData( np.ascontiguousarray(src_img_4) )
        self.depth_view_1.setData(depth_map_1 * 200)
        self.depth_view_2.setData(depth_map_2 * 200)
        self.depth_view_3.setData(depth_map_3 * 200)
        self.depth_view_4.setData(depth_map_4 * 200)


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