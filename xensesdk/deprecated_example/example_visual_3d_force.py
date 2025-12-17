from xensesdk.omni.GLOmniItem import OmniSensorItem
from xensesdk.xenseInterface.XenseSensor import Sensor
from xensesdk.ezgl.items import *
from xensesdk.ezgl import GLViewWidget, tb
import numpy as np
import cv2
from xensesdk.ezgl.utils.colormap import cm
from xensesdk.omni.fem import OmniFemModel


class InferView(GLViewWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.initUI()
        # self.img_cap = Sensor(r"J:\xensesdk\data\sensor_0_rectify_video.mp4", sensor_type=Sensor.SensorType.VirtualOmni)
        self.img_cap = Sensor.create(1, config_path= r"C:\Users\Administrator\Downloads\QJ-G43")
        self.fem = OmniFemModel(r"C:\Users\Administrator\Downloads\fem_omni.npz", self.img_cap._marker_config)

        tb.add_timer("timer", 10, self.onTimeout, self)

    def __scale_grid(self, grid):
        grid = grid.astype(np.float32) / (400, 700)
        grid = grid * [0.94, 0.93] + [0.03, 0.003]
        return grid
    
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
        self.omni_sensor = OmniSensorItem(lights=[self.light])
        self.addItem(self.griditem)
        self.addItem(self.axis)
        self.addItem(self.omni_sensor)

        with tb.window("tb", size=(300, 600)):
            with tb.group("view", horizontal=True, collapsible=True):
                self.src_img_view = tb.add_image_view("src" , None, img_size=(300, 500), img_format= "bgr")
                self.depth_view = tb.add_image_view("" , None, img_size=(300, 500))
                self.marker_img_view = tb.add_image_view("marker" , None, img_size=(300, 500), img_format= "bgr")
            # self.surf = VisualizeWidget(None)
            # self.surf.show()


    # @profile
    def onTimeout(self):
        sensor_info = self.img_cap.getAllSensorInfo()

        self.src_img_view.setData( np.ascontiguousarray(sensor_info['src_img']) )
        # self.surf.set_data(np.ascontiguousarray(src_img))
        marker_img = self.img_cap.drawMarkerMove(sensor_info['src_img'])
        self.marker_img_view.setData(marker_img)
        self.omni_sensor.set_depth(np.flip(-sensor_info['depth_map']/6, 1))
        self.depth_view.setData(sensor_info['depth_map'] * 200)

        physical_marker = sensor_info["marker"][0].copy()
        physical_marker[...,0] = 400 - physical_marker[...,0]
        physical_marker = np.flip(physical_marker, 1)
        # physical_marker_init = self.img_cap.getInitMarker().copy()
        # physical_marker_init[...,0] = 400 - physical_marker_init[...,0]
        # marker_curr = self.omni_sensor.get_3d_marker(self.__scale_grid(physical_marker))
        # marker_init = self.omni_sensor.get_3d_marker(self.__scale_grid(physical_marker_init))

        # self.omni_sensor.set_3d_marker(marker_curr)
        # self.arrow_plot.setData(marker_init, marker_curr)
        marker_init, marker_disp= self.fem.get_marker_flow(physical_marker, np.flip(-sensor_info['depth_map'], 1))
        xyz, disp_xyz, _ = self.fem.get_mesh_flow(physical_marker, np.flip(-sensor_info['depth_map'], 1))
        F = self.fem.get_mesh_force(disp_xyz)
        F_len = (1 - np.linalg.norm(F, axis=2) * 8)
        color = cm.autumn(F_len)
        self.omni_sensor.set_3d_arrow(xyz/10, xyz/10 - F*1.5, color)
        self.omni_sensor.set_3d_marker((marker_init + marker_disp).reshape(-1, 3) / 10)
        
        
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