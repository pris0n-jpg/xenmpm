from xensesdk.xenseInterface.configManager import SensorConfig, ConfigManager
from xensesdk.xenseInterface.sensorEnum import SensorType, ConnectionType, MachineType
from xensesdk.omni.realCamera import OmniCamera
from xensesdk.omni.virtualCamera import VirtualCamera
from xensesdk.omni.remoteCamera import RemoteCamera


class CameraFactory:
    @classmethod
    def create(cls, sensor_config: SensorConfig, **kwargs):
        # dynamic params 
        params = {}

        params["api"] = sensor_config.camera_source
        params["sensor_type"] = sensor_config.sensor_type
        if sensor_config.height:
            params["size"] = (sensor_config.width, sensor_config.height)

        if sensor_config.connection_type == ConnectionType.Local:
            return OmniCamera(sensor_config.camera_id, **params)
        elif sensor_config.connection_type == ConnectionType.Virtual:
            return VirtualCamera(sensor_config.video_path, **params)
        elif sensor_config.connection_type == ConnectionType.Remote:
            return RemoteCamera(sensor_config.serial_number, sensor_config.ip_address, **params)