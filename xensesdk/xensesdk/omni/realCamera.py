import os
import av
import cv2
import time
import threading
import numpy as np
from xensesdk.omni import transforms as tf
from xensesdk.ezgl.functions import CircularBuffer
from typing import Tuple, Optional
from xensesdk.utils.callback_manager import CallbackManager
from .cameraBackend import CameraBackendFactory
from xensesdk.xenseInterface.sensorEnum import CameraSource, SensorType

if os.name == 'posix':
    PLATFORM = 'LINUX'
    DEFAULT_API = CameraSource.CV2_V4L2
elif os.name == 'nt':
    PLATFORM = 'WIN'
    DEFAULT_API = CameraSource.CV2_DSHOW
else:
    raise Exception("Unsupported OS")

if PLATFORM == 'WIN':
    import ctypes
    from ctypes.wintypes import LARGE_INTEGER

    kernel32 = ctypes.windll.kernel32
    INFINITE = 0xFFFFFFFF
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
    def accurate_sleep(seconds):
        """
        accurate sleep function for Windows.

        Parameters:
        - seconds : float
        """
        handle = kernel32.CreateWaitableTimerExW(
            None, None, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, 0x1F0003
        )
        res = kernel32.SetWaitableTimer(
            handle,
            ctypes.byref(LARGE_INTEGER(int(seconds * -10000000))),
            0, None, None, 0,
        )
        res = kernel32.WaitForSingleObject(handle, INFINITE)
        kernel32.CancelWaitableTimer(handle)
    
else:
    def accurate_sleep(seconds):
        time.sleep(seconds)

class RealCamera:

    camera_mode = 'default'
    def __init__(self, cam_id=0, size=(640, 480), api=DEFAULT_API, sensor_type=SensorType.Omni, disable_auto_wb=True):
        self._cam = None
        self._sensor_type = sensor_type
        self._cam_id = cam_id
        self._api = DEFAULT_API if api == None else api
        self._size = size
        self._disable_auto_wb = disable_auto_wb
        self._img_buff = CircularBuffer(5)
        self._fetch_img_thread = None
        self._connected = False
        self._callback_manager = CallbackManager()

        self.init(cam_id, self._size, self._api)

    @property
    def id(self):
        return self._cam_id

    def init(self, cam_id, size=None, api=None):
        self._cam_id = int(cam_id)
        self._api = api if api else self._api
        self._size = size if size else self._size

        self.release()
        self.selectBackend()
        
        ret, frame = self._cam.read()
        self._img_buff.put(frame)
        if not ret:
            raise Exception(f"[ERROR] Cannot open camera {cam_id}")
        
        if self._fetch_img_thread == None or not self._fetch_img_thread.is_alive():
            self._fetch_img_thread = threading.Thread(target=self._on_fetch_img, daemon=True)
            self._streaming = True
            self._fetch_img_thread.start()
    
    def selectBackend(self):
        self._cam = CameraBackendFactory.create(self._api, self._cam_id, self._size, self._sensor_type)

    def informUINetworkStatus(self):
        self._callback_manager._notify_callbacks()

    def _on_fetch_img(self):
        
        while self._streaming:
            t0 = time.time()
            if self._cam is not None:
                ret, frame = self._cam.read()
                if ret:
                    self._img_buff.put(frame)             

                if not ret:
                    if self._connected == True:
                        self._connected = False
                        print (f"In SDK: [Network] Camera {self._cam_id} disconnected")
                        self.informUINetworkStatus()
                else:
                    if self._connected == False:
                        self._connected = True
                        print (f"In SDK: [Network] Camera {self._cam_id} connected")                    
                        self.informUINetworkStatus()
                
            t1 = time.time()
            if t1 - t0 < 0.016: # 60 hz limit
                accurate_sleep(0.016 - (t1 - t0))

    def get_raw_frame(self):
        img = self._img_buff.get()
        ret = img is not None
        return ret, img

    def get_property(self, name):
        return self._cam.getProperty(name)

    def set_property(self, name, value):
        self._cam.setProperty(name, value)
        # print(f"Set {name} to {int(value)}")

    def set_properties(self, properties: dict):
        for name, value in properties.items():
            self.set_property(name, value)

    def get_properties(self) -> dict:
        properties = {}
        for name in self._cam._property_dict.keys():
            properties[name] = self.get_property(name)
        return properties

    def release(self):
        if self._cam is not None:
            self._cam.release()
            self._img_buff.ptr = -1   # clear()
            self._cam = None

    def destroyFetchThread(self):
        self._streaming = False
        self._fetch_img_thread.join()
    

class OmniCamera(RealCamera):
    diff = tf.Diff(scale=1.5)

    def __init__(self, cam_id=0, size=(640, 480), api=DEFAULT_API, sensor_type = SensorType.Omni, disable_auto_wb=True):
        try:
            super().__init__(cam_id, size, api, sensor_type, disable_auto_wb)
            self.rectify = tf.Rectify(
            12, 7, (400, 700),
            src_grid= np.flip(np.array([[[126.298, 146.471],
    [152.262, 119.215],
    [183.236, 106.345],
    [210.798,  99.250],
    [238.290,  93.153],
    [263.787,  86.916],
    [298.751,  74.325],
    [341.695,  62.291],
    [394.824,  47.962],
    [465.979,  33.891],
    [539.708,  26.015],
    [617.078,  23.406]],

    [[125.691, 169.484],
    [150.170, 149.142],
    [179.936, 139.195],
    [209.284, 135.232],
    [235.639, 131.060],
    [263.990, 127.028],
    [297.677, 118.357],
    [340.342, 110.313],
    [396.184, 100.184],
    [464.646,  95.950],
    [545.498,  86.567],
    [628.714,  86.371]],

    [[124.017, 193.426],
    [147.728, 184.057],
    [179.141, 179.237],
    [207.282, 178.197],
    [235.493, 176.160],
    [262.776, 173.056],
    [296.184, 168.375],
    [339.498, 165.389],
    [397.914, 161.454],
    [466.167, 160.212],
    [549.523, 158.021],
    [632.740, 157.826]],

    [[122.134, 220.360],
    [148.210, 220.179],
    [177.348, 219.209],
    [206.347, 220.234],
    [232.424, 220.053],
    [259.428, 220.939],
    [295.549, 220.457],
    [339.582, 221.531],
    [395.724, 221.447],
    [465.623, 225.333],
    [546.635, 227.990],
    [632.705, 229.999]],

    [[122.454, 244.441],
    [146.766, 255.163],
    [176.623, 258.253],
    [205.204, 265.264],
    [232.277, 265.152],
    [259.142, 268.033],
    [293.987, 271.473],
    [338.528, 279.599],
    [395.250, 285.571],
    [464.870, 293.446],
    [544.536, 301.022],
    [632.600, 303.170]],

    [[121.848, 267.455],
    [148.454, 288.362],
    [176.756, 299.363],
    [205.406, 305.376],
    [230.345, 307.120],
    [256.791, 315.986],
    [292.285, 324.483],
    [337.545, 336.670],
    [393.848, 348.626],
    [462.122, 361.420],
    [540.581, 371.919],
    [625.374, 377.848]],

    [[120.174, 291.396],
    [146.940, 324.344],
    [175.960, 339.405],
    [202.616, 345.279],
    [227.206, 352.010],
    [253.512, 362.872],
    [289.794, 374.431],
    [330.576, 393.322],
    [389.523, 410.475],
    [454.386, 429.045],
    [532.914, 438.546],
    [609.587, 445.913]]]).transpose(1, 0, 2), 0).copy(),
            )
            self._diff_enabled = False
        except:

            raise Exception(f"init camera {cam_id} fail")
    # @profile
    def get_frame(self):
        ret, frame = self.get_raw_frame()
        if not ret:
            return False, None

        frame = self.rectify(frame)

        if self._diff_enabled:
            frame = self.diff(frame)
        return True, frame

    def set_diff_enabled(self, enabled):
        if enabled and self.diff._ref_img is None:
            self.update_diff()
        self._diff_enabled = enabled

    def update_diff(self):
        ret, frame = self.get_raw_frame()
        frame = self.rectify(frame)
        if ret:
            self.diff.set(frame, self.diff._scale)
