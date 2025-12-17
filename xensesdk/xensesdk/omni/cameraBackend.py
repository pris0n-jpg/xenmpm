import av
import cv2
import time
import numpy as np
from typing import Tuple, Optional
from xensesdk.xenseInterface.sensorEnum import CameraSource, SensorType

class CameraBackendFactory:
    _registry = {}

    @classmethod
    def list_backends(cls):
        return list(cls._registry.keys())
    
    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def create(cls, api, cam_id, img_size, sensor_type):
        backend_class = cls._registry.get(api)
        if not backend_class:
            raise ValueError(f"Backend {api} is not registered.")
        instance = backend_class(cam_id, img_size, sensor_type)
        return instance

# ------------------------------------------------------------------------------
class CameraBackend:
    _property_dict = {
        "brightness":       cv2.CAP_PROP_BRIGHTNESS,
        "contrast":         cv2.CAP_PROP_CONTRAST,
        "hue":              cv2.CAP_PROP_HUE,
        "saturation":       cv2.CAP_PROP_SATURATION,
        "sharpness":        cv2.CAP_PROP_SHARPNESS,
        "white_balance":    cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
        "white_balance_t":  cv2.CAP_PROP_WB_TEMPERATURE,
        "exposure":         cv2.CAP_PROP_EXPOSURE,
        "auto_wb":          cv2.CAP_PROP_AUTO_WB,
        "auto_exposure":    cv2.CAP_PROP_AUTO_EXPOSURE
    }
    _param_presets = {
            "linux": {
                "default": {
                    "auto_exposure": 1,
                    "auto_wb": False,
                    "white_balance_t": 5550,
                    "exposure": 450,
                },
                "finger": {
                    "auto_exposure": 1,
                    "auto_wb": False,
                    "white_balance_t": 6500,
                    "exposure": 600,
                }
            },
            "win": {
                "default": {
                    "white_balance": 5550,
                    "brightness": 38,
                    "exposure": -5,
                },
                "finger": {
                    "white_balance": 4600,
                    "exposure": -4,
                }
            }
        }
    def __init__(self, cam_id, img_size, sensor_type):
        self._sensor_type = sensor_type
        self._cam_id = cam_id
        self._img_size = img_size
        self._timeOut = 0.5
        self._backend = None
        self.init()
    
    def init():
        raise NotImplementedError()
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        self.preRead()
        ret, frame = self._read()
        ret = self.postRead(ret)
        return ret, frame
   
    def _read(self):
        return NotImplementedError()
    
    def preRead(self):
        pass
    
    def postRead(self, ret: bool) -> bool:
        return ret
    
    def setProperty(self) -> bool:
        raise NotImplementedError()
    
    def getProperty(self):
        raise NotImplementedError()
    
    def release(self):
        raise NotImplementedError()

class CV2ApiBackend(CameraBackend):
    
    def _read(self):
        return self._backend.read()
    
    def setProperty(self, name, value):
        # -5
        self._backend.set(self._property_dict[name], float(value))
    
    def getProperty(self, name):
        if isinstance(name, int):
            name = list(self._property_dict.keys())[name]
        return self._backend.get(self._property_dict[name])
    
    def init(self):
        # check platform
        if self._api in {CameraSource.CV2_MSMF, CameraSource.CV2_DSHOW}:
            api_platform = "win"
            api_cap = cv2.CAP_DSHOW
        elif self._api in {CameraSource.CV2_V4L2, CameraSource.AV_V4L2}:
            api_platform = "linux"
            api_cap = cv2.CAP_V4L2

        if self._sensor_type == SensorType.Finger:
            param_mode = "finger"
        else:
            param_mode = "default"
        
        # prepare setting
        self._backend = cv2.VideoCapture(self._cam_id, api_cap,
                                [cv2.CAP_PROP_FRAME_WIDTH, self._img_size[0], 
                                cv2.CAP_PROP_FRAME_HEIGHT, self._img_size[1]])
        config_params = self._param_presets[api_platform].get(param_mode)
        for key, value in config_params.items():
            self.setProperty(key, value)
    
    def release(self):
        self._backend.release()

@CameraBackendFactory.register(CameraSource.CV2_DSHOW)
class WinCV2DSHOWBackend(CV2ApiBackend):
    _api = CameraSource.CV2_DSHOW
    
    def preRead(self):
        self.t0 = time.time()
    
    def postRead(self, ret: bool) -> bool:
        t1 = time.time()
        if t1 - self.t0 > self._timeOut:
            return False
        return ret

@CameraBackendFactory.register(CameraSource.CV2_MSMF)
class WinCV2MSMFBackend(WinCV2DSHOWBackend):
    _api = CameraSource.CV2_MSMF

    def init(self):
        super().init()
        # reset setup
        self.release()
        self._backend = cv2.VideoCapture(self._cam_id, cv2.CAP_MSMF,
                                [cv2.CAP_PROP_FRAME_WIDTH, self._img_size[0], 
                                cv2.CAP_PROP_FRAME_HEIGHT, self._img_size[1]])
    
    def postRead(self, ret):
        return ret

@CameraBackendFactory.register(CameraSource.CV2_V4L2)
class LinuxCV2V4L2Backend(CV2ApiBackend):
    _api = CameraSource.CV2_V4L2

@CameraBackendFactory.register(CameraSource.AV_V4L2)
class LinuxAVBackend(LinuxCV2V4L2Backend):
    _api = CameraSource.AV_V4L2
    def __init__(self, cam_id, img_size = (640,480)):
        super().__init__(cam_id, img_size)

    def _read(self):
        try:
            frame = next(self._backend.decode(video=0))  # Get the first decoded video frame
            frame = frame.to_ndarray(format='bgr24')
            ret = True
        except:
            frame = None
            ret = False
        return ret, frame
    
    def init(self):
        super().init()
        super().release()
        try:
            self._backend = av.open(
                f'/dev/video{self._cam_id}', 
                format='v4l2',  # Use Video4Linux2 format (Linux) or 'dshow' (Windows)
                options={
                    'input_format': 'mjpeg',  # Set MJPEG as the format
                    'video_size': f'{self._img_size[0]}x{self._img_size[1]}'  # Specify the image size (Width x Height)
                }
            )
            # stream_code = self._cam.streams[0].codec_context
            # print(f"cam open with format:{stream_code.codec_tag} size: {stream_code.width}x{stream_code.height}")

        except:
            print(f"cannot open cam use format:mjpeg, size:{self._img_size[0]}x{self._img_size[1]}, fallback to other method")
            self._backend = av.open(
                f'/dev/video{self._cam_id}', 
                format='v4l2',  # Use Video4Linux2 format (Linux) or 'dshow' (Windows)
            )
            stream_code = self._backend.streams[0].codec_context
            print(f"actual image format:{stream_code.codec_tag} size: {stream_code.width}x{stream_code.height}")

    def getProperty(self, name):
        raise Exception("please use debug mode to check property using cv2 backend")

    def release(self):
        self._backend.close()


