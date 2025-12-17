import numpy as np
from enum import Enum
from typing import Optional, List, Union, Tuple
from pathlib import Path

class Sensor:
    class OutputType(Enum):
        Rectify = 1
        Difference = 2  
        Depth = 3
        Marker2D = 4
        Marker3D = 5
        Marker3DFlow = 6
        Marker3DInit = 7
        MarkerUnorder = 8
        Force = 9
        ForceResultant = 16
        ForceNorm = 10
        Mesh3D = 11
        Mesh3DFlow = 12
        Mesh3DInit = 13
        Marker2DInit = 14
        Marker2DFlip = 15

    @classmethod
    def create(
        cls,
        cam_id: Union[int, str] = 0,
        use_gpu: bool = True,
        config_path: Optional[Union[str, Path]] = None,
        api: Optional[object] = None,  # 若有具体 Enum 类型，建议替换 object
        infer_size: Tuple[int, int] = (144, 240),
        check_serial: bool = True,
        rectify_size: Optional[Tuple[int, int]] = None,
        ip_address: Optional[str] = None,
        video_path: Optional[str] = None
    ) -> "Sensor":
        """
        创建 Sensor 实例。

        Args:
            cam_id (int | str, optional): 相机 ID、序列号或视频路径。默认值为 0。
            use_gpu (bool, optional): 是否使用 GPU 进行推理。默认值为 True。
            config_path (str | Path, optional): 配置文件路径。默认 None。
            api (Enum, optional): 相机 API 类型（如 OpenCV 后端）。默认 None。
            check_serial (bool, optional): 是否检查相机序列号。默认 True。
                - 若设置环境变量 `XENSE_AUTO_SERIAL=1`，所有相机设备将使用 PID+camid 作为序列号（仅用于调试）。
            rectify_size (tuple[int, int], optional): 矫正后图像尺寸，用于图像重映射。默认 None。
            ip_address (str, optional): 相机的网络地址（用于远程连接）。默认 None。
            video_path (str, optional): 视频文件路径（用于离线模拟）。默认 None。

        Returns:
            Sensor: 创建的传感器实例。
        """

    def getRectifyImage(self) -> np.ndarray:
        """
        获取传感器图像，并对原始图像进行重映射。

        Returns:
            np.ndarray: 图像数据，形状为 (700, 400, 3)，类型为 uint8
        """

    def resetReferenceImage(self) -> None:
        """
        重置传感器的参考图像。
        """
        ...

    def startSaveSensorInfo(self, path: str, data_to_save: Optional[List[Sensor.OutputType]] = None) -> None:
        """
        开始保存传感器信息。

        Args:
            path (str): 保存数据的路径。
            data_to_save (Optional[List[Sensor.OutputType]]): 指定要保存的数据类型列表, None则为所有可保存数据。
        """
        ...

    def stopSaveSensorInfo(self) -> None:
        """
        停止保存传感器信息。
        """
        ...

    def getCameraID(self) -> int:
        """
        获取相机 ID。

        Returns:
            int: 相机的唯一标识符。
        """
        ...

    def release(self) -> None:
        """
        安全退出传感器，释放资源。
        """

    def selectSensorInfo(self, *args: Sensor.OutputType):
        """
        选择需要输出的传感器数据。

        Parameters:
            *args: Sensor.OutputType 实例，指定选择的传感器输出类型。
        """
    def drawMarkerMove(self, img):
        """
        绘制描述marker位置变化的向量图像

        Parameters:
            img: image in [700,400,3] (h,w,c) uint8
        Returns:
            img: image in [700,400,3] (h,w,c) uint8
        """
    
    def drawMarker(self, img, marker, color=(3, 253, 253), radius=2, thickness=2):
                """
        在原图上标注出已检测出的marker位置

        Parameters:
            img: image in [700,400,3] (h,w,c) uint8
            marker: np.array 
        Returns:
            img: image in [700,400,3] (h,w,c) uint8
        """

class ExampleView:
    def create2d(self, *args) -> View2d:
        """
        
        """
    def setDepth(self, depth):
        """
        
        """

    def setForceFlow(self, force, res_force, mesh_init):
        """
        
        """

    def setCallback(self, function):
        """
        
        """

    def show(self):
        """
        
        """

    def setMarkerUnorder(self, marker_unordered):
        """
        
        """

    class View2d():
        def setData(self, name, img):
            """
            
            """
