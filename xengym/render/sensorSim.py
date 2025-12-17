from typing import Generic, TypeVar

from .sensorScene import SensorScene
from .. import PROJ_DIR


class VecTouchSim(SensorScene):

    def __init__(
        self,
        depth_size: tuple,
        fem_file: str = str(PROJ_DIR / "assets/data/fem_data_gel_2035.npz"),
        # fem_file: str = str(PROJ_DIR / "assets/data/fem_data_vec4070.npz"),
        visible: bool = False,
        title = "VecTouchSim"
    ):
        """
        单个 VecTouch 触觉传感器仿真场景, 需要外部提供 depth, 物体和相机坐标系的姿态用于 step

        Parameters:
        - fem_file : str or Path, FEM 文件,
        - depth_size : tuple, (width, height), 深度图尺寸
        - visible : bool, optional, default: False
        - title : str, optional, default: "XenseSim"
        """
        self.gel_size_mm = (17.3, 29.15)
        # self.gel_size_mm = (19.4, 30.8)
        # self.gel_size_mm = (17.2, 28.5)
        self.marker_row_col = (20, 11)
        # self.marker_dx_dy_mm = (1.48, 1.38)
        self.marker_dx_dy_mm = (1.31, 1.31)
        super().__init__(
            fem_file,
            depth_size,
            self.gel_size_mm,
            marker_row_col = self.marker_row_col,
            marker_dx_dy_mm = self.marker_dx_dy_mm,
            visible = visible,
            title = title
        )



SensorType = TypeVar("SensorType")


class MultiSensorSim(Generic[SensorType]):

    def __init__(
        self,
        names: list[str],
        depth_size: tuple,
        fem_file: str = None,
        visible: bool = True
    ):
        self.sensors = dict()

        for i, name in enumerate(names):
            kwargs = dict(fem_file=fem_file) if fem_file else dict()
            self.sensors[name] = SensorType(depth_size=depth_size, name=name, visible=visible, **kwargs)

    def __getitem__(self, key) -> SensorType:
        return self.sensors[key]


MultiVecTouchSim = MultiSensorSim[VecTouchSim]

