import numpy as np
import ctypes
from xensesdk import SYSTEM, MACHINE
from . import PROJ_DIR

class CamList(ctypes.Structure):
    _fields_ = [
        ("deviceName", ctypes.c_wchar * 1024),
        ("devicePath", ctypes.c_wchar * 1024),
        ("mSize", ctypes.c_long),
        ("idx", ctypes.c_byte)
    ]
    _pack_ = 8 
    def __init__(self):
        self.deviceName = u""
        self.devicePath = u""
        self.mSize = 0
        self.idx = 0

class RectifyConfigFlash(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("rot", ctypes.c_float),
        ("trans_x", ctypes.c_float),
        ("trans_y", ctypes.c_float),
        ("scale", ctypes.c_float),
        ("padding", ctypes.c_int32*4),
        ("rows", ctypes.c_int32),
        ("cols", ctypes.c_int32),
        ("base_grid", ctypes.POINTER(ctypes.c_int32))  # Pointer to 2D float array
    ]
            
    def print_fields(self):
        print(f"Rotation: {self.rot}")
        print(f"Translation: ({self.trans_x}, {self.trans_y})")
        print(f"Scale: {self.scale}")
        print(f"Rows: {self.rows}, Cols: {self.cols}")
        print(f"padding: {np.ctypeslib.as_array(self.padding, shape=(4,))}")
        
        grid = self.read_base_grid()
        if grid is not None:
            print("Base Grid Data:")
            print(grid)

    def read_base_grid(self):
        """ Convert the base_grid C pointer into a NumPy array. """
        if not self.base_grid:
            print("Error: base_grid is NULL")
            return None

        grid = np.ctypeslib.as_array(self.base_grid, shape=(self.cols*self.rows*2,)).reshape(self.cols, self.rows, 2)

        return grid
    
    def to_dict(self):
        # data transfer

        cols = self.cols
        rows = self.rows
        base_grid = np.ctypeslib.as_array(self.base_grid, shape=(cols*rows*2,))
        base_grid = base_grid.reshape((cols,rows,2))
        data_dict = {
            "rot": self.rot,
            "trans_x": self.trans_x,
            "trans_y": self.trans_y,
            "scale": self.scale,
            "padding": list(self.padding),
            "base_grid": base_grid
        }
        return data_dict


class MarkerConfigFlash(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("x0", ctypes.c_int32),
        ("y0", ctypes.c_int32),
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("theta", ctypes.c_float),
        ("ncol", ctypes.c_int32),
        ("nrow", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("lower_threshold", ctypes.c_int32),
        ("min_area", ctypes.c_int32),
    ]

    def print_fields(self):
        for field, _ in self._fields_:
            print(f"{field}: {getattr(self, field)}")
    
    def to_dict(self):
        config_dict = {}
        for field, _ in self._fields_:
            config_dict[field] = getattr(self, field)
        return config_dict

class XenseFlashManager:
    flash_manager = None

    def __init__(self):
        #TODO: ADD flash write in other platform
        try:
            if SYSTEM == "linux" and MACHINE == "x86_64":
                self.flash_manager = ctypes.CDLL(str(PROJ_DIR.parent / 'lib/linux_64/XenseWrapper.so'))
            elif SYSTEM == "linux" and MACHINE == "aarch64":
                self.flash_manager = ctypes.CDLL(str(PROJ_DIR.parent / 'lib/arm_64/XenseWrapper.so'))
            elif SYSTEM == "windows":
                # self.flash_manager = ctypes.CDLL(str(PROJ_DIR.parent / 'lib/win_x86/XenseWrapper.dll'))
                self.flash_manager = ctypes.CDLL(str(PROJ_DIR.parent / 'lib/win_x86/FRWLib.dll'))
                ctypes.CDLL(str(PROJ_DIR.parent / 'lib/win_x86/CLib.dll'))
                ################## for solving windows device number matching bug ##################  
                # self._check_device() 
                # self.flash_manager.getDeviceList.argtypes = [ctypes.POINTER(CamList), ctypes.POINTER(ctypes.c_long)]
                # self.flash_manager.getDeviceList.restype = ctypes.c_bool
                ##########################################################################################
            else:
                print(f"Do not support {SYSTEM}-{MACHINE} platform")
        except Exception as e:
            print("Failed to load flash reading function.")

        if self.flash_manager is not None:
            self.flash_manager.ReadMarkerConfig.argtypes = [ctypes.c_long, ctypes.POINTER(MarkerConfigFlash)]
            self.flash_manager.ReadMarkerConfig.restype = None

            self.flash_manager.SaveMarkerConfig.argtypes = [ctypes.c_long, ctypes.POINTER(MarkerConfigFlash)]
            self.flash_manager.SaveMarkerConfig.restype = None

            self.flash_manager.ReadRectifyConfig.argtypes = [ctypes.c_long, ctypes.POINTER(RectifyConfigFlash)]
            self.flash_manager.ReadRectifyConfig.restype = None

            self.flash_manager.SaveRectifyConfig.argtypes = [ctypes.c_long, ctypes.POINTER(RectifyConfigFlash)]
            self.flash_manager.SaveRectifyConfig.restype = None

            self.flash_manager.SetSensorType.argtypes = [ctypes.c_long, ctypes.c_char_p]
            self.flash_manager.SetSensorType.restype = None

            self.flash_manager.GetSensorType.argtypes = [ctypes.c_long, ctypes.c_char_p]
            self.flash_manager.GetSensorType.restype = None

            self.flash_manager.SaveExtraParams.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_float), ctypes.c_uint8]
            self.flash_manager.SaveExtraParams.restype = None

            self.flash_manager.ReadExtraParams.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_float), ctypes.c_uint8]
            self.flash_manager.ReadExtraParams.restype = None



    def readMarkerConfig(self, device_number):
        marker_config_flash = MarkerConfigFlash()
        self.flash_manager.ReadMarkerConfig(device_number, ctypes.byref(marker_config_flash))
        return marker_config_flash.to_dict()

    def saveMarkerConfig(self, device_number, data_dict):
        for field_name, field_type in MarkerConfigFlash._fields_:
            val = data_dict.get(field_name)
            if val is not None:
                # 自动转换类型
                if issubclass(field_type, ctypes.c_int):
                    data_dict[field_name] = ctypes.c_int32(int(val))
                elif issubclass(field_type, ctypes.c_float):
                    data_dict[field_name] = ctypes.c_float(float(val))

        marker_config_flash = MarkerConfigFlash(**data_dict)
        self.flash_manager.SaveMarkerConfig(device_number, ctypes.byref(marker_config_flash))

    def readRectifyConfig(self, device_number):
        rectify_config_read = RectifyConfigFlash()
        self.flash_manager.ReadRectifyConfig(device_number, ctypes.byref(rectify_config_read))
        return rectify_config_read.to_dict()

    def saveRectifyConfig(self, device_number, data_dict):
        rectify_config_write = RectifyConfigFlash(**data_dict)
        self.flash_manager.SaveRectifyConfig(device_number, ctypes.byref(rectify_config_write))

    def setSensorType(self, device_number, sensor_type):
        sensor_type_bytes = sensor_type.encode('utf-8') 
        self.flash_manager.SetSensorType(ctypes.c_long(device_number), ctypes.c_char_p(sensor_type_bytes))

    def getSensorType(self, device_number):
        sensor_type_buffer = ctypes.create_string_buffer(128)  
        self.flash_manager.GetSensorType(ctypes.c_long(device_number), sensor_type_buffer)
        sensor_type = sensor_type_buffer.value.decode('utf-8') 
        return sensor_type
    
    def saveExtraParams(self, device_number, float_data):
        float_array = (ctypes.c_float * len(float_data))(*float_data)
        self.flash_manager.SaveExtraParams(device_number, float_array, len(float_data))

    def readExtraParams(self, device_number, length=3):
        float_array = (ctypes.c_float * length)()    
        self.flash_manager.ReadExtraParams(device_number, float_array, length)
        ret = [float(x) for x in float_array]
        if np.isnan(ret[0]):
            return [1., 1., 1.]
        return ret
    
    # def _check_device(self):
    #     num_devices = ctypes.c_long(0)
    #     cam_list = (CamList * 10)()  # Allocate space for 5 devices
    #     success = self.flash_manager.getDeviceList(cam_list, ctypes.byref(num_devices))
    #     return success, cam_list, num_devices


xense_flash_manager = XenseFlashManager()