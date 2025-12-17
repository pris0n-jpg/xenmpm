from xensesdk.xenseInterface.XenseSensor import Sensor
from xensesdk.ezgl.items import *
from xensesdk.ezgl import GLViewWidget, tb
from xensesdk.utils.encrypt import encrypt_config, decrypt_config
import numpy as np
import sys
from qtpy import QtCore, QtWidgets
from xensesdk.ezgl.utils.QtTools import QImageViewWidget
import cv2
from dataclasses import asdict
from xensesdk.xenseInfer.markerConfig import MarkerConfig
from pathlib import Path
from xensesdk.xenseInterface.sensorEnum import SensorType
from xensesdk.omni.rectifyConfig import RectifyConfig
import pickle
import time
PROJ_DIR = Path(__file__).resolve().parent


class InferView():
    rectify_row = 20
    rectify_col = 11
    sensor_type = "VecTouch"
    low_thresh = 16
    min_area = 30
    def __init__(self, parent=None):
        # super().__init__(parent=parent)
        self.draw_map_grid = False
        self.initUI()
        self.sensor = Sensor.create(0, check_serial=False, infer_size=(144,240))
        self.sensor._dag_runner._otvalue= 10000
        self.map_back_marker=None
        
    def saveConfig(self):
        rectify_img = self.sensor.selectSensorInfo(Sensor.OutputType.Rectify)
        pad = [tb.get_widget("padding top  ").value, tb.get_widget("padding bot  ").value,
               tb.get_widget("padding left ").value, tb.get_widget("padding right").value]
        if self.map_back_marker is None:
            rectify_conf = self.sensor.fetchRectifyConfig()
            rectify_conf["padding"] = pad
            rectify_conf = RectifyConfig(**rectify_conf)
            
        else:
            rectify_conf = RectifyConfig(base_grid=self.map_back_marker.transpose(1,0,2).tolist(), padding=pad)
        marker_dst_grid = self.sensor._real_camera.rectify.dst_grid
        marker_conf = MarkerConfig(x0= int(marker_dst_grid[0,0,0]), 
                                   y0= int(marker_dst_grid[0,0,1]), 
                                   dx= float(marker_dst_grid[0,1,0] - marker_dst_grid[0,0,0]),
                                   dy= float(marker_dst_grid[1,0,1] - marker_dst_grid[0,0,1]),
                                   ncol=self.rectify_col,
                                   nrow=self.rectify_row)
        rectify_conf_dict = asdict(rectify_conf)
                # rectify_config = data["RectifyConfig"]
        rectify_conf_dict.pop("width", None)
        rectify_conf_dict.pop("height", None)
        data_to_write = {
            "SensorType": self.sensor_type,
            "MarkerConfig": asdict(marker_conf),
            "RectifyConfig": rectify_conf_dict,
            "ExtraParams": [1,1,1]
        }
        save_pkl_path = r"D:\gitlab\xensesdk\deprecated_example\config\data.pkl"
        with open(save_pkl_path, "wb") as f:
            pickle.dump(data_to_write, f)
        # write to flash
        self.sensor.resetMarkerConfig(asdict(marker_conf))
        self.sensor.resetRectifyConfig(rectify_conf_dict)
        self.sensor._config_manager.sensor_config.sensor_type = SensorType[self.sensor_type]
        self.sensor._config_manager.sensor_config.force_calibrate_param = [1,1,1]
        self.sensor.saveConfigFlash()

        save_path = tb.get_widget("save path").value
        (PROJ_DIR / "config").mkdir(exist_ok=True)
        encrypt_config(data_to_write, PROJ_DIR / "config" / save_path, "Wz8mmWz2ALJ6X5Ic")
        cv2.imwrite(PROJ_DIR / "reference_img" / f"{save_path}.png", rectify_img)
        
        print("############# integrity-checking ################")
        with open(save_pkl_path, "rb") as f:
            loaded_data = pickle.load(f)
        saved_grid = np.array(self.sensor._config_manager.rectify_config.base_grid).flatten()
        loaded_grid = np.array(loaded_data['RectifyConfig']['base_grid']).flatten()
        diff = saved_grid!=loaded_grid
        print(f"grid data difference count: {len(np.where(diff)[0])}")
        print("############# integrity-checking end ############")

    def loadConfig(self):
        load_path = tb.get_widget("load path").value

        if "/" in load_path:
            file_name = load_path[load_path.rfind("/") + 1:]
        else:
            file_name = load_path
        tb.set_value("save path", file_name)

        self.sensor.loadConfig(load_path)
        config = self.sensor.fetchRectifyConfig()
        pad = config["padding"]
        tb.get_widget("padding top  ").value = pad[0]
        tb.get_widget("padding bot  ").value = pad[1]
        tb.get_widget("padding left ").value = pad[2]
        tb.get_widget("padding right").value = pad[3]
        

    def sortPoints(self, points):
        # 计算网格
        sorted_data = points[np.argsort(points[:, 1])]

        # 重塑为 (18, 9, 2)
        reshaped_data = sorted_data.reshape(self.rectify_row, self.rectify_col, 2)
        sorted_grid = np.array([row[np.argsort(row[:, 0])] for row in reshaped_data])
        return sorted_grid

    def drawLine(self, img, mesh = None):
        # 在图像上绘制线段
        for row in mesh:
            valid_points = [point for point in row if not np.isnan(point).any()]  # 过滤掉包含 np.nan 的点
            for i in range(len(valid_points) - 1):
                start_point = tuple(map(int, valid_points[i]))      # 起点
                end_point = tuple(map(int, valid_points[i + 1]))    # 终点
                cv2.line(img, start_point, end_point, color=(0, 255, 0), thickness=2)  # 绿色线条

        return img
    
    def mapBack(self):
        # get mapx and mapy
        _map_x = self.sensor._real_camera.rectify._mapx.copy()
        _map_y = self.sensor._real_camera.rectify._mapy.copy()

        # get grid on rectified picc
        self.map_back_marker = np.zeros_like(self.ordered_marker)

        for y in range(self.rectify_row):
            for x in range(self.rectify_col):
                self.map_back_marker[y,x] = np.array(
                    [
                        _map_x[self.ordered_marker[y,x][1], self.ordered_marker[y,x][0]],
                        _map_y[self.ordered_marker[y,x][1], self.ordered_marker[y,x][0]],
                    ]
                ) 
        self.draw_map_grid = True
        print("map ok ")


    def mousePressEvent(self, position):
        threshold = 10 # pixel
        selected_point = np.array([position[0], position[1]])
        
        distances = np.linalg.norm(self.marker_unordered - selected_point, axis=1)
        if np.any(distances < threshold):
            index_to_remove = np.where(distances < threshold)[0][0]  # 找到第一个满足条件的点索引
            self.marker_unordered = np.delete(self.marker_unordered, index_to_remove, axis=0)
        else:
            # add to list 
            self.marker_unordered = np.vstack((self.marker_unordered, selected_point))
        
        # show points 
        raw_img, src_img  = self.sensor.selectSensorInfo(Sensor.OutputType.Raw, Sensor.OutputType.Rectify)
        self.src_img_view.setData(raw_img)
        self.rectify_view.setData( np.ascontiguousarray(src_img) )
        marker_img = self.sensor.drawMarker(src_img, self.marker_unordered)
        self.marker_img_view.setData(marker_img)

    def mousePressMoveEvent(self, info):
        print(info)

    def calculateGrid(self):
        self.ordered_marker = self.sortPoints(self.marker_unordered)
        # show points 
        raw_img, src_img  = self.sensor.selectSensorInfo(Sensor.OutputType.Raw, Sensor.OutputType.Rectify)
        self.src_img_view.setData(raw_img)
        self.rectify_view.setData( np.ascontiguousarray(src_img) )
        marker_img = self.sensor.drawMarker(src_img, self.marker_unordered)
        marker_line = self.drawLine(marker_img, self.ordered_marker)
        self.marker_img_view.setData(marker_line)


    def refreshMarker(self):
        raw_img, src_img, self.marker_unordered = self.sensor.selectSensorInfo(Sensor.OutputType.Raw, Sensor.OutputType.Rectify, Sensor.OutputType.MarkerUnorder)
        # self.marker_unordered = self.sensor._image_processor.detectMarker(src_img, self.low_thresh, self.min_area)
        self.src_img_view.setData(raw_img)
        self.rectify_view.setData( np.ascontiguousarray(src_img) )
        marker_img = self.sensor.drawMarker(src_img, self.marker_unordered)

        
        self.marker_img_view.setData(marker_img)
    
    def refreshSrc(self):
        raw_img, src_img  = self.sensor.selectSensorInfo(Sensor.OutputType.Raw, Sensor.OutputType.Rectify)

        raw_img_c = raw_img.copy()
        if self.draw_map_grid:
            # self.map_back_marker[:,:,1] = 336 / 700 * self.map_back_marker[:,:,1]
            # self.map_back_marker[:,:,0] = 192 / 400 * self.map_back_marker[:,:,0]
            for y in range(self.rectify_row):
                for x in range(self.rectify_col):
                    cv2.circle(raw_img_c, (self.map_back_marker[y,x][0], self.map_back_marker[y,x][1]),
                        radius=0, color=(3, 253, 253), thickness=2, )  # 黄点


        self.src_img_view.setData(raw_img_c)
        self.rectify_view.setData(np.ascontiguousarray(src_img) )

    def setPad(self):
        pad = [tb.get_widget("padding top  ").value, tb.get_widget("padding bot  ").value,
               tb.get_widget("padding left ").value, tb.get_widget("padding right").value]
        config = self.sensor.fetchRectifyConfig()
        config["padding"] = pad
        self.sensor.resetRectifyConfig(config)
    
    def setRectify(self):
        scale = tb.get_widget("scale").value
        rotate = tb.get_widget("rotate").value
        trans_x = tb.get_widget("trans_x").value
        trans_y = tb.get_widget("trans_y").value
        config = self.sensor.fetchRectifyConfig()
        
        rectify_config: dict = {"rot": float(rotate), "trans_x": int(trans_x), "trans_y": int(trans_y), "scale": float(scale), "base_grid": config["base_grid"]}
        # print(rectify_config)
        self.sensor.resetRectifyConfig(rectify_config)

    def autoPad(self):
        tb.set_value("padding top  ",-30)
        tb.set_value("padding bot  ",-25)
        tb.set_value("padding left ",-30)
        tb.set_value("padding right",-30)
        self.setPad()

    def reloadGrid(self):
        pad = [tb.get_widget("padding top  ").value, tb.get_widget("padding bot  ").value,
               tb.get_widget("padding left ").value, tb.get_widget("padding right").value]
        rectify_conf = RectifyConfig(base_grid=self.map_back_marker.transpose(1,0,2).tolist(), 
                                     padding= pad)
        self.sensor.resetRectifyConfig(asdict(rectify_conf)) 

    def setMarkerDetect(self):
        self.low_thresh = tb.get_widget("low thresh").value
        self.min_area = tb.get_widget("min area").value
        config = self.sensor.fetchMarkerConfig()
        config["lower_threshold"] = self.low_thresh
        config["min_area"] = self.min_area
        self.sensor.resetMarkerConfig(config)
    
    def onSensorTypeChange(self, sensor_type):
        print("Current sensor_type:", SensorType[sensor_type])
        self.sensor_type = sensor_type
        self.rectify_row, self.rectify_col = {
            "VecTouch": (20, 11),
            "Omni": (18, 9),
            "Finger": (18, 9)
        }[sensor_type]

    def initUI(self):

        with tb.window("tb", size=(1300,2100)):
            tb.add_timer("timer", interval_ms=20, callback= self.onTimeout)
            self.src_img_view = QImageViewWidget()
            # self.rectify_view  = QImageViewWidget()  
            self.marker_img_view = QImageViewWidget()  
            self.marker_img_view.sigMousePressed.connect(self.mousePressEvent) 
            # self.marker_img_view.clicked.connect(self.mousePressEvent)
            with tb.group("left_win", horizontal=True, show=False):
                with tb.group("left_win", horizontal=False, show=False):
                    with tb.group(f"Current sensor type: {self.sensor_type}, Marker size: {self.rectify_col}x{self.rectify_row}", horizontal=True, collapsible=True):
                        tb.add_widget("src", self.src_img_view)
                        # tb.add_widget("rectify", self.rectify_view)
                        self.rectify_view = tb.add_image_view("" , None, img_size=(300, 500), img_format= "bgr" )
                        tb.add_widget("marker", self.marker_img_view)

                    with tb.group("control", horizontal=True, collapsible=True):
                        tb.add_button("reset marker", callback= self.refreshMarker)
                        tb.add_button("calculate mesh",  callback= self.calculateGrid)
                        tb.add_button("map back",  callback= self.mapBack)

                    with tb.group("param", horizontal=True, collapsible=True):
                        tb.add_text_editor("save path", value= "")
                        tb.add_button("save config", callback= self.saveConfig)
                        tb.add_filepath("load path" )
                        tb.add_button("load config", callback= self.loadConfig)
                        tb.add_button("reset cam", callback= self.resetCam)
                        tb.add_button("load new grid", callback= self.reloadGrid)

                with tb.group("panel", horizontal=False, collapsible=False, show=False):
                    tb.add_combo("sensor type", ["VecTouch", "Omni", "Finger"], callback=self.onSensorTypeChange)
                    with tb.group("rectify", horizontal=False, collapsible=False):
                        tb.add_slider("scale",1,0.5,1.5,0.01, decimals=3,callback= self.setRectify)
                        tb.add_slider("rotate",0,-20,20,0.01, callback= self.setRectify)
                        tb.add_slider("trans_x",0,-100,100,1, callback= self.setRectify)
                        tb.add_slider("trans_y",0,-100,100,1, callback= self.setRectify)

                    with tb.group("pad", horizontal=False, collapsible=False):
                        tb.add_slider("padding top  ",0,-60,60,1, callback= self.setPad)
                        tb.add_slider("padding bot  ",0,-60,60,1, callback= self.setPad)
                        tb.add_slider("padding left ",0,-60,60,1, callback= self.setPad)
                        tb.add_slider("padding right",0,-60,60,1, callback= self.setPad)
                        tb.add_button("auto padding", callback= self.autoPad)
                        
                    with tb.group("marker detect", horizontal=False, collapsible=False):
                        tb.add_slider("low thresh",16,0,100,1,callback= self.setMarkerDetect)
                        tb.add_slider("min area",30,0,200,1, callback= self.setMarkerDetect)
                    tb.add_stretch(1)

    
    def resetCam(self):
        self.sensor.resetCamera(0)


    # @profile
    def onTimeout(self):
        self.refreshSrc()


def main():

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    win = InferView()
    app.exec_()
    
    sys.exit()

if __name__ == '__main__':
    main()