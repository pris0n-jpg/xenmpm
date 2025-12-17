import time, os
import threading
from typing import Optional, Union
from dataclasses import asdict
import h5py
import numpy as np
import cv2
from . import PROJ_DIR, Path
from xensesdk.omni.cameraFactory import CameraFactory
from xensesdk.omni.rectifyConfig import RectifyConfig
from xensesdk.xenseInfer.markerTracker import FlowTracker
from xensesdk.xenseInfer.markerConfig import MarkerConfig
from xensesdk.utils.encrypt import encrypt_config
from xensesdk.ezgl.utils.video_utils import CV2VideoWriter
from xensesdk.omni.fem import OmniFemModel
from xensesdk.xenseInterface.scanXenseDevice import scan_xense_devices
from xensesdk.xenseInterface.configManager import ConfigManager
from xensesdk.xenseInterface.sensorEnum import OutputType, ConnectionType
from xensesdk.xenseInfer.inference import InferBase
from xensesdk.xenseInterface.DAGRunner import DAGManager
from xensesdk.xenseInterface.dataRecorder import DataRecorder


# # HACK: 由于很多传感器都没有序列号, 因此设置自动序列号为 serialnumber+camid
# os.environ["XENSE_AUTO_SERIAL"] = "1"


class Sensor:
    _real_camera = None
    _fem = None
    _infer_engine = None
    _flow_tracker = None
    _dag_runner = None

    OutputType = OutputType
    @classmethod
    def _camid_to_serial(cls, cam_id: int) -> str:
        """
        将相机编号转换为序列号, 若 cam_id 不存在则返回 None

        Parameters:
        - cam_id : int, 相机编号

        Returns:
        - serial_number, str
        """
        serial_number = None
        for key, value in cls._serial_number_map.items():
            if value == cam_id:
                serial_number = key
                break
        return serial_number
    
    @classmethod
    def extract_config(cls, config_path, serial_number):
        config_path = Path(config_path) / str(serial_number)
        if not config_path.exists():
            print(f"Cann't find config file for {serial_number}")
            return None
        return config_path

    @classmethod
    def flushSerialNumber(cls):
        cls._serial_number_map, cam_ids = scan_xense_devices()
        return cls._serial_number_map, cam_ids

    @classmethod
    def create(cls, cam_id=0, config_path=None, use_gpu=True, api=None, infer_size=(144, 240), check_serial=True, rectify_size=None, ip_address=None, video_path=None, raw_size=None, **kwargs) -> "Sensor":
        """
        创建 Sensor 实例

        Parameters:
        - cam_id : int/serial_numer, default: 0
        - config_path : str/Path, optional, default: None, 配置文件路径
        - use_gpu : bool, optional, default: True, 使用 GPU 进行推理
        - api : Enum, optional, default: None, 相机 API (Deprecated)
        - check_serial : bool, optional, default: True, 当直接使用 cam_id 创建时, 是否检查相机序列号, 不检查则可以打开无序列号的相机. 
        - infer_size : tuple, optional, default: (144, 240), (w, h) 推理尺寸
        - rectify_size : tuple, optional, default: None, (w, h) 矫正后的图像大小, 若为 None 则使用 grid_coord_size(标定时的矫正后尺寸) 代替
        - raw_size: tuple, optional, default: None, (w, h) 原始图像大小, 若为 None, 则使用各型号默认值
        - ip_address : str, optional, default: None, 当相机为远程连接时, 需要提供 IP 地址
        - video_path : str, optional, default: None, 当相机为视频文件时, 需要提供视频路径
        """
        _config_manager = ConfigManager(
            cam_id=cam_id,
            video_path=video_path,
            config_path=config_path,
            ip_address=ip_address,
            infer_size=infer_size,
            rectify_size=rectify_size,
            raw_size=raw_size,
            check_serial=check_serial,
            camera_api=api,
            **kwargs
        )

        return Sensor(_config_manager, use_gpu=use_gpu)
        
    def __init__(self, config_manager: ConfigManager, use_gpu=True, api=None) -> None:
        self._config_manager = config_manager
        self._initCamera() # set _real_camera
        self._initFlowTrackerAndFem()
        self._initInferEngine(use_gpu)
        self._initDAGRunner()  # must after _initCamera, _initInferEngine
        self._initDataRecorder()

    @property
    def infer_size(self):
        return ( self._config_manager.infer_config.width, self._config_manager.infer_config.height)
    
    @property
    def grid_coord_size(self):
        """
        返回定义 marker grid 的坐标系大小, 为 (marker_config.width, marker_config.height), 如 (400, 700)
        """
        return self._config_manager.marker_config.width, self._config_manager.marker_config.height
    
    @property
    def sensor_type(self):
        return self._config_manager.sensor_config.sensor_type

    @property
    def rectify_size(self):
        """
        返回 rectify 后图像的大小
        """
        return self._config_manager.rectify_config.width, self._config_manager.rectify_config.height 
        
    def getRectifyImage(self):
        '''
        image: image in [480,640,3] (h,w,c) uint8 (通过self.real_camera.get_frame()获取)
        return: image in [700,400,3] (h,w,c) uint8
        获取传感器图像,对原始图像进行(重映射)
        '''
        _, img = self._real_camera.get_frame()
        return img
    
    def getAllSensorInfo(self, image: Optional[np.ndarray]=None) -> dict:
        OT = Sensor.OutputType
        img, depth, diff, marker_2d, marker_3d, force = self.selectSensorInfo(
            OT.Rectify, OT.Depth, OT.Difference, OT.Marker2D, OT.Marker3D, OT.Force, image=image
        )
        return {
            "src_img": img,
            "diff_img": diff,
            "depth_map": depth,
            "marker": marker_2d,
            "marker_3d": marker_3d,
            "force": force,
        }
    def _initInferEngine(self, use_gpu):
        if self._config_manager.sensor_config.connection_type != ConnectionType.Remote:
            self._infer_engine = InferBase.create(self._config_manager, use_gpu)

    def _initFlowTrackerAndFem(self):
        sensor_type = self._config_manager.sensor_config.sensor_type
        self._flow_tracker = FlowTracker(sensor_type, self.grid_coord_size)
        self._fem = OmniFemModel.create(sensor_type, self._flow_tracker.grid_init, self.grid_coord_size)

    def _initCamera(self, cam_id=0):
        if self._real_camera is None:
            self._real_camera = CameraFactory.create(self._config_manager.sensor_config)
        else:
            self._real_camera.init(cam_id)
        rectify_params = self._config_manager.calc_rectify_params()
        self._real_camera.rectify.set(**rectify_params)
        self._real_camera.set_diff_enabled(False)

    def _initDAGRunner(self):
        self._dag_runner = DAGManager.create(self)
    
    def _initDataRecorder(self):
        self._data_recorder = DataRecorder(self)

    def selectSensorInfo(self, *args):
        return self._dag_runner.selectSensorInfo(*args)

    def drawMarkerMove(self, img):
        '''
        image: image in [700,400,3] (h,w,c) uint8
        return: image in [700,400,3] (h,w,c) uint8
        绘制描述marker位置变化的向量图像
        '''
        return self._flow_tracker.draw_move(img)

    def drawMarker(self, img, marker, color=(3, 253, 253), radius=2, thickness=2):
        return self._flow_tracker.draw_marker(img, marker, color, radius, thickness)

    def fetchRectifyConfig(self):
        return asdict(self._config_manager.rectify_config)
    
    def fetchRectifyMap(self):
        return self._config_manager.rectify_config.get_grid()
    
    def fetchMarkerConfig(self):
        return asdict(self._config_manager.marker_config)

    def resetReferenceImage(self):  
        self._dag_runner.resetRefernceImage()

    def resetRectifyConfig(self, rectify_config):
        self._config_manager.rectify_config = RectifyConfig(**rectify_config)
        rectify_params = self._config_manager.calc_rectify_params()
        self._real_camera.rectify.set(**rectify_params)
        self._real_camera.set_diff_enabled(False)

    def resetCamera(self, cam_id):
        self._initCamera(cam_id)

    def resetMarkerConfig(self, config_dict):
        self._config_manager.marker_config = MarkerConfig(**config_dict)

    def loadConfig(self, data: Union[dict, str], **kwargs):
        if isinstance(data, str):
            if not self._config_manager.readConfigFromFile(data):
                raise ValueError(f"cannot read from config file : {data}")
        
        # reset rectify config to camera
        self.resetRectifyConfig(asdict(self._config_manager.rectify_config))
        # init camera when first time
        params = {
            "cam_id": self._config_manager.sensor_config.camera_id
        }
        if "size" in kwargs:
            params["size"] = kwargs["size"]
        if "api" in kwargs:
            params["api"] = kwargs["api"]
        self._initCamera(**params)

    def saveConfig(self, dst_path):
        try:
            passcode = "Wz8mmWz2ALJ6X5Ic"
            self._config_manager.rectify_config.base_grid = self._config_manager.rectify_config.base_grid.tolist()
            data_to_write = {
                "SensorType": self.sensor_type.name,
                "MarkerConfig": asdict(self._config_manager.marker_config),
                "RectifyConfig": asdict(self._config_manager.rectify_config),
                "ExtraParams": self._config_manager.sensor_config.force_calibrate_param
            }
            encrypt_config(data_to_write, dst_path, passcode)
        except:
            print("Wrong save file path")
    
    def readConfigFlash(self):
        self.release()
        self._config_manager.readConfigFromFlash(self._real_camera.id)
        self._initCamera(self._real_camera.id)

    def saveConfigFlash(self):
        self.release()
        self._config_manager.writeConfigToFlash(self._real_camera.id)
        self._initCamera(self._real_camera.id)

    #-- 数据采集
    def startSaveSensorInfo(self, path, data_to_save):
        if self._data_recorder.isRecording():
            return
        else:
            self._recording = True
            if data_to_save == None:
                data_to_save = list(Sensor.OutputType)
            self._data_recorder.startRecord(path, data_to_save)
            
    def stopSaveSensorInfo(self):
        self._data_recorder.stopRecord()
    
    # 注册回调函数
    def registerCallback(self, callback):
        self._real_camera._callback_manager.register_callback(callback)
    
    def getCameraID(self):
        return self._real_camera.id
    
    def getNetworkStatus(self):
        return self._real_camera._connected
    
    def release(self):
        self.stopSaveSensorInfo()
        self._real_camera.release()
        self._real_camera.destroyFetchThread()
        # self._dag_runner._auto_refresher.release()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def __del__(self):
        self.release()
