import os
# from xensesdk.autoTest.simuLib import autoTestOn
# if autoTestOn.Enabled:
#     from xensesdk.autoTest.simuLib import simulateCVLib as cv2
# else:
#     import cv2
import h5py
import cv2
import time
from xensesdk.omni import transforms as tf
from threading import Lock
from pathlib import Path
if os.name == 'posix':
    PLATFORM = 'LINUX'
    DEFAULT_API = cv2.CAP_V4L2
elif os.name == 'nt':
    PLATFORM = 'WIN'
    DEFAULT_API = cv2.CAP_MSMF
else:
    raise Exception("Unsupported OS")


class VirtualRectify(tf.Rectify):
    
    def __call__(self, img):
        return img
        
    
class VirtualCamera:
    def __init__(self, data_path, size=(640, 480), api=DEFAULT_API, disable_auto_wb=True, **kwargs):
        if not isinstance(data_path, str):
            raise TypeError("Video file path must be a string.")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Video file {data_path} not found.")
        
        self.video_path = data_path
        if Path(data_path).suffix.lower() == '.h5':
            self.h5file = h5py.File(data_path, "r")
                # Print the structure of the file
            self._cam = self.h5file["rectify_data"]
        else:
            self._cam = cv2.VideoCapture(data_path)
        self.rectify = VirtualRectify(2, 2, (400, 700))
        self._cam_id = 0
        self._lock = Lock()
        self.count = 0

    def init(self, path, size=None, api=None):
        self.release()
        if Path(path).suffix.lower() == '.h5':
            with h5py.File(path, "r") as h5file:
                # Print the structure of the file
                self._cam = h5file["rectify_data"]
        else:
            self._cam = cv2.VideoCapture(path)

    def get_frame(self):
        return self.get_raw_frame()

    def get_raw_frame(self):        
        if isinstance(self._cam , cv2.VideoCapture):
            with self._lock:
                time.sleep(0.001)
                ret, frame = self._cam.read()
                if not ret:
                    self._cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cam.read()
                # decrypt
                return ret, frame
        else:
            frame = self._cam[self.count]
            temp = frame[:100].copy()
            frame[:100] = frame[100:200]
            frame[100:200] = temp
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            
            self.count += 1
            if self.count == len(self._cam):
                self.count = 0
            # frame = cv2.remap(frame, self.inv_mapx, self.inv_mapy, interpolation=cv2.INTER_LINEAR)
            return True, frame
    
    def set_diff_enabled(self, b):
        pass
    
    def destroyFetchThread(self):
        if isinstance(self._cam , cv2.VideoCapture):
            self._cam.release()
        else:
            self.h5file.close()
    
    def release(self):
        if isinstance(self._cam , cv2.VideoCapture):
            self._cam.release()
        else:
            self.h5file.close()