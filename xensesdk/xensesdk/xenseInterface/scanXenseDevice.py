import os
from . import PROJ_DIR
from distutils.util import strtobool

xense_id_lookup_table = [
    # vendor_id, product_id
    ("a000", "b111"),
    ("3938", "1299"),
    ("0BDA", "D561"),
]

if os.name == 'posix':
    import pyudev

    def scan_xense_devices():
        """
        Scan for Xense devices connected to the system.

        Returns:
        - dict: A dictionary mapping serial numbers to device IDs
        """
        # Create a context object to interact with udev
        context = pyudev.Context()
        xense_devices = {}
        auto_serial = strtobool(os.environ.get("XENSE_AUTO_SERIAL", '0'))

        # 查找所有video4linux设备
        for device in context.list_devices(subsystem='video4linux'):
            # 每个USB相机设备通常会创建两个video设备节点, 一个用于视频流 (MJPEG/YUYV等格式), 一个用于元数据或其他功能
            # 例如 /dev/video0 和 /dev/video1 可能同属于同一个相机设备, 两者的区别在于设备的功能
            v4l_capability = device.get('ID_V4L_CAPABILITIES')
            if v4l_capability != ":capture:":
                continue

            vendor_id = device.get('ID_VENDOR_ID')
            product_id = device.get('ID_MODEL_ID')
            serial_number = device.get('ID_SERIAL_SHORT')
            cam_id = int(device.device_node.split('video')[-1])

            if not auto_serial:
                if not (vendor_id, product_id) in xense_id_lookup_table:
                    continue

            if serial_number in xense_devices:
                serial_number = f"{serial_number}_{cam_id}"  # 若有重复的设备编号, 加上设备ID

            xense_devices[serial_number] = cam_id

        print(f"Found Xense devices: {xense_devices}")
        cam_ids = list(xense_devices.values())

        return xense_devices, cam_ids


elif os.name == 'nt':
    import wmi
    import ctypes
    import pythoncom  # 关键模块
    import threading

    # Initialize the WMI client
    parentPath = PROJ_DIR.parent
    dllPath = os.path.join(parentPath, "lib", "CvCameraIndex.dll")
    # 加载对应位数动态链接库
    _CvCameraIndex = ctypes.cdll.LoadLibrary(dllPath)
    _CvCameraIndex.getCameraIndex.argtypes = [ctypes.c_char_p]
    _CvCameraIndex.getCameraIndex.restype = ctypes.c_int

    def get_camera_id(hwid: str) -> int:
        """
        获取cv.VideoCapture(camera_index, cv2.CAP_DSHOW)参数camera_index

        Parameters:
        - hwid : str, 硬件标识, 不区分大小写, 如 vid_1234&pid_4321, 或 op000001

        Returns:
        - int, camera_index
        """
        return _CvCameraIndex.getCameraIndex(bytes(hwid, encoding='utf-8'))

    def scan_xense_devices():
        """
        Scan for Xense devices connected to the system.

        Returns:
        - dict: A dictionary mapping serial numbers to device IDs
        """
        auto_serial = strtobool(os.environ.get("XENSE_AUTO_SERIAL", '0'))
        if threading.current_thread() != threading.main_thread():
            pythoncom.CoInitialize()  # 初始化 COM
        wmi_clent = wmi.WMI()
        if auto_serial:
            devices_ids = wmi_clent.query(f"SELECT * FROM Win32_PnPEntity WHERE PNPClass = 'Camera'")
            devices_ids = [dev.DeviceID.split('\\')[1] for dev in devices_ids]
            # VID_0BDA&PID_3035&MI_00 删除最后一个 & 以及之后的字符
            devices_ids = ['&'.join(dev.split('&')[0:2]) for dev in devices_ids]
            devices_map = {}
            for dev in devices_ids:
                cam_id = get_camera_id(dev)
                devices_map[dev[-4:] + f"_{cam_id}"] = cam_id
        else:
            devices_ids = []
            for vid, pid in xense_id_lookup_table:
                res = wmi_clent.query(f"SELECT * FROM Win32_USBHub WHERE DeviceID LIKE '%VID_{vid.upper()}&PID_{pid.upper()}%'")
                if res:
                    devices_ids.extend(res)
            devices_ids = [dev.DeviceID.split('\\')[-1] for dev in devices_ids]
            devices_map = { dev: get_camera_id(dev) for dev in devices_ids }
                
        print(f"Found Xense devices: {devices_map}")
        cam_ids = list(devices_map.values())
        if threading.current_thread() != threading.main_thread():
            pythoncom.CoUninitialize()  # 清理 COM
        return devices_map, cam_ids


# if __name__ == '__main__':
#     import cv2
#     from ezgl.utils.QtTools import qtcv

#     devices = scan_xense_devices()

#     # 按次序读取 Xense 设备
#     for serial_number, cam_id in devices.items():
#         cap = cv2.VideoCapture(cam_id)

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             qtcv.imshow(f"Xense {serial_number}", frame)

#             if qtcv.waitKey(10) == qtcv.Key.Key_Escape:
#                 break
