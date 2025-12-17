import sys, os
from dataclasses import asdict, dataclass, field
from typing import Union, Optional, Set
import numpy as np

from . import PROJ_DIR, Path
from xensesdk.omni.rectifyConfig import RectifyConfig
from xensesdk.xenseInfer.markerConfig import MarkerConfig
from xensesdk.utils.encrypt import decrypt_config
from xensesdk.xenseInterface.scanXenseDevice import scan_xense_devices
from xensesdk.xenseInterface.sensorEnum import ConnectionType, SensorType, InferType, MachineType, DAGType, OutputType, CameraSource
from xensesdk.utils.flashRW import xense_flash_manager
from xensesdk import SYSTEM


@dataclass
class SensorConfig:
    # local
    serial_number: str = None
    camera_id: int = None
    # remote
    ip_address: str = None
    # virtual
    video_path: str = None
    connection_type: ConnectionType = None
    sensor_type: SensorType = None
    camera_source: CameraSource = None
    width:int = None
    height:int = None 
    default_raw_size: list = field(default_factory=lambda: [640, 480])  # 默认的原始图像大小
    force_calibrate_param:list = None
    
    def write_flash(self, device_id):
        xense_flash_manager.setSensorType(device_id, self.sensor_type.name)
        xense_flash_manager.saveExtraParams(device_id, self.force_calibrate_param)
        
    def set_data(self, **kwargs):
        """
        设置传感器类型, 并自动设置 default_raw_size, 设置 force_calibrate_param 的值
        """
        if "sensor_type" in kwargs:
            self.sensor_type = kwargs["sensor_type"]

            ###HACK: 对于finger或特殊传感器，有不同的打开相机默认参数和rectify比例，如果user输入就用user的，如果没输入，就用默认的 
            if self.sensor_type == SensorType.Finger:
                self.default_raw_size = [640, 360]
                if not self.width:
                    self.width = 640
                    self.height = 360
            else:
                self.default_raw_size = [640, 480]
                if not self.width:
                    self.width = 640
                    self.height = 480
        
        if "force_calibrate_param" in kwargs:
            self.force_calibrate_param = kwargs["force_calibrate_param"]
        
        if "serial_number" in kwargs:
            self.serial_number = kwargs["serial_number"]


@dataclass
class InferConfig:

    width: int = 144
    height: int = 240
    machine_type: MachineType = None
    infer_type: InferType = None
    dag_type: DAGType = DAGType.Split
    cache_from_remote: Set[OutputType] = field(default_factory=set)  # 缓存远程相机的输出类型

    def __post_init__(self):
        """
        自动检测 machine_type, infer_type, dag_type
        """
        self.detectMachineType()

        if self.machine_type in (MachineType.RK3588, MachineType.RK3576):
            self.infer_type = InferType.RKNN
            self.dag_type = DAGType.AllInOne

        elif self.machine_type == MachineType.Jetson:
            self.infer_type = InferType.Torch

        elif self.machine_type == MachineType.RDK_X5:
            raise Exception("RDK_X5 not support yet!")  # TODO

    def detectMachineType(self):
        self.machine_type = MachineType.X86

        if os.path.exists("/etc/nv_tegra_release"):
            self.machine_type = MachineType.Jetson
            print("Jetson detected")

        if not os.path.exists("/proc/device-tree/model"):
            return

        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            if "orange pi 5" in model:
                self.machine_type = MachineType.RK3588
            elif "rdk x5" in model:
                self.machine_type = MachineType.RDK_X5
            elif "3576" in model:
                self.machine_type = MachineType.RK3576


class ConfigManager:
    serial_number_map = dict()
    sensor_config: SensorConfig = None
    infer_config: InferConfig = None
    marker_config: MarkerConfig = None
    rectify_config: RectifyConfig = None

    def __init__(
        self,
        cam_id: Union[str, int] = None,
        config_path: Optional[Union[str, Path]] = None,
        video_path: Optional[Union[str, Path]] = None,
        ip_address: Optional[str] = None,
        infer_size = (144, 240),
        rectify_size = (400, 700),
        raw_size = None,
        check_serial = True,
        camera_api = None,
        **kwargs
    ):
        ConfigManager.flushSerialNumber()
        self.kwargs = kwargs
        rectify_size = (400, 700) if rectify_size is None else rectify_size
        infer_size = (144, 240) if infer_size is None else infer_size
        raw_size = (None, None) if raw_size is None else raw_size
        
        self.infer_config = InferConfig(width=infer_size[0], height=infer_size[1], infer_type=kwargs.get("infer_type", InferType.ONNX))
        self.sensor_config = self.initSensorConfig(cam_id, video_path, ip_address, raw_size, check_serial, camera_api=camera_api)
        self.marker_config = MarkerConfig(x0=20, y0=60, dx=45, dy=37, ncol=9, nrow=18)
        self.rectify_config = RectifyConfig(width=rectify_size[0], height=rectify_size[1])

        if self.sensor_config.connection_type == ConnectionType.Virtual:
            self.readConfigFromFile(config_path)

        elif self.sensor_config.connection_type == ConnectionType.Local:  # 先读配置文件, 再读 flash
            if not self.readConfigFromFile(config_path):
                self.readConfigFromFlash(self.sensor_config.camera_id)

        elif self.sensor_config.connection_type == ConnectionType.Remote:
            if not self.readConfigFromFile(config_path):
                self.readConfigFromRemote(cam_id, ip_address)

    def initSensorConfig(self, cam_id:  Union[str, int], video_path, ip_address, raw_size, check_serial, camera_api) -> SensorConfig:
        """
        初始化除了 sensor_type 的 sensor_config 属性
        """
        # connection type
        connection_type = None
        if ip_address is not None:
            connection_type = ConnectionType.Remote
        elif cam_id is not None:
            connection_type = ConnectionType.Local
        elif video_path is not None:
            connection_type = ConnectionType.Virtual

        # serial_number and camera_id
        real_cam_id = None
        serial_number = None
        if connection_type == ConnectionType.Local:
            if isinstance(cam_id, int):  # id
                real_cam_id = cam_id
                serial_number = ConfigManager.camidToSerial(cam_id)
                assert not check_serial or serial_number is not None, f"Bad cam_id: {cam_id}"

            elif isinstance(cam_id, str):  # serial_number
                serial_number = cam_id
                real_cam_id = ConfigManager.serial_number_map.get(serial_number, None)
                assert connection_type == ConnectionType.Remote or real_cam_id is not None, f"Bad serial number: {serial_number}"

        # get default camera source 
        if SYSTEM == "linux":
            camera_source = CameraSource.CV2_V4L2
        elif SYSTEM == "windows":
            if camera_api is not None:
                camera_source = camera_api
            else:   
                camera_source = CameraSource.CV2_DSHOW
        if self.infer_config.machine_type == MachineType.RK3588 or self.infer_config.machine_type == MachineType.RK3576:
            camera_source = CameraSource.AV_V4L2

        sensor_config = SensorConfig(
            connection_type=connection_type,
            serial_number=serial_number,
            camera_id=real_cam_id,
            ip_address=ip_address,
            video_path=video_path,
            sensor_type=None,  # NOTE : this will be set in readConfigXXX method
            camera_source=camera_source,
            width=raw_size[0],
            height=raw_size[1],
            force_calibrate_param=[1,1,1]
        )        
        sensor_config.set_data(sensor_type=SensorType.Omni)  # 设置默认值为 Omni, 同时设置 default_raw_size, readConfig 成功时会被覆盖
        return sensor_config        

    def readConfigFromFile(self, config_path=None) -> bool:
        config_path = self.extractConfig(config_path, self.sensor_config.serial_number)
        if config_path is None:
            return False
        if config_path.suffix == ".yaml":
            return self._readConfigFromYaml(config_path)
        
        try:
            passcode = "Wz8mmWz2ALJ6X5Ic"
            # 尝试解密配置文件
            data = decrypt_config(config_path, passcode)
            assert data is not None and "SensorType" in data, f"Wrong config file path! file do not exist."
        except Exception as e:
            print(f"Read config from {config_path} failed!")
            return False

        self.sensor_config.set_data(
            sensor_type = SensorType[data["SensorType"]],
            force_calibrate_param = data.get("ExtraParams", [1.,1.,1.])
        )
        self.rectify_config.read_config(data["RectifyConfig"])  # 不会覆盖 rectify_size
        self.marker_config = MarkerConfig(**data["MarkerConfig"])
        print(f"Read config from {config_path} success!")
        return True
    
    def _readConfigFromYaml(self, config_path) -> bool:
        try:
            import yaml
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            assert data is not None and "SensorType" in data, f"Wrong config file path! file do not exist."
        except Exception as e:
            print(f"Read config from {config_path} failed!")
            return False

        self.sensor_config.set_data(
            sensor_type = SensorType[data["SensorType"]],
            force_calibrate_param = data.get("ExtraParams", [1.,1.,1.])
        )
        self.rectify_config.read_config(data["RectifyConfig"])  # 不会覆盖 rectify_size
        self.marker_config = MarkerConfig(**data["MarkerConfig"])
        print(f"Read config from {config_path} success!")
        return True

    def readConfigFromFlash(self, cam_id=None) -> bool:
        if cam_id is None:
            return False
        try:
            self.sensor_config.set_data(
                sensor_type = SensorType[xense_flash_manager.getSensorType(cam_id)],
                force_calibrate_param = xense_flash_manager.readExtraParams(cam_id)
            )
            self.rectify_config.read_flash(cam_id)
            self.marker_config.read_flash(cam_id)
            print(f"Read config from cam_id_{cam_id} success!")
            return True
        except:
            print(f"Read config from cam_id_{cam_id} failed!")
            return False

    def readConfigFromRemote(self, camera_serial, ip_address) -> bool:
        from xensesdk.zeroros import Node
        # get config from node and destory
        node = Node(ip_address)
        client = node.create_client(camera_serial)
        remote_config:ConfigManager = client.config()['ret']
        node.destroy_node()
        
        # decode config and save it 
        self.sensor_config.set_data(
            sensor_type = remote_config.sensor_config.sensor_type,
            force_calibrate_param = remote_config.sensor_config.force_calibrate_param,
            serial_number = camera_serial,
        )
        self.marker_config = remote_config.marker_config
        # TODO: 现在是继承本地 rectify_size, 但本地和远程冲突的情况没有考虑
        rec = remote_config.rectify_config
        rec.width = self.rectify_config.width
        rec.height = self.rectify_config.height
        self.rectify_config = rec
        
        return True
    
    def writeConfigToFlash(self, cam_id):
        try:
            self.marker_config.write_flash(cam_id)
            self.rectify_config.write_flash(cam_id)
            self.sensor_config.write_flash(cam_id)
            return True
        except Exception as e:
            print(f"write flash error: {e}")

    @classmethod
    def camidToSerial(cls, cam_id: int) -> str:
        """
        将相机编号转换为序列号, 若 cam_id 不存在则返回 None

        Parameters:
        - cam_id : int, 相机编号

        Returns:
        - serial_number, str
        """
        serial_number = None
        for key, value in cls.serial_number_map.items():
            if value == cam_id:
                serial_number = key
                break
        return serial_number

    @classmethod
    def flushSerialNumber(cls):
        cls.serial_number_map, cam_ids = scan_xense_devices()
        return cls.serial_number_map, cam_ids

    @classmethod
    def extractConfig(cls, config_path, serial_number):
        """
        valid config_path, 如果是 None/dir, 自动补全为 config_path/serial_number
        """
        config_path: Path = Path(".") if config_path is None else Path(config_path)

        if config_path.is_file():
            return config_path
        
        if not config_path.is_dir():
            return None

        if serial_number is None:
            return None
        else:
            config_path = config_path / str(serial_number)
            if config_path.exists():
                return config_path
            else:
                return None
        
    def calc_rectify_params(self):
        """
        计算更新相机 rectify transform 的参数, src_grid, img_sz, pad
        """
        if self.sensor_config.width is None:
            base_grid_scale = 1
        else:
            base_grid_scale = np.array([self.sensor_config.width, self.sensor_config.height]) / self.sensor_config.default_raw_size
            
        src_grid = self.rectify_config.get_grid() * base_grid_scale

        rectify_size = np.array([self.rectify_config.width, self.rectify_config.height])
        pad_scale = rectify_size / (self.marker_config.width, self.marker_config.height)
        pad = self.rectify_config.padding
        pad = [pad[0] * pad_scale[1], pad[1] * pad_scale[1], pad[2] * pad_scale[0], pad[3] * pad_scale[0]]

        return {
            "src_grid": src_grid,
            "img_sz": rectify_size,
            "pad": pad
        }
