from dataclasses import dataclass
import numpy as np
import ctypes
from . import PROJ_DIR
from xensesdk import SYSTEM, MACHINE
from xensesdk.utils.flashRW import xense_flash_manager

def rotate_matrix_around_point(x, y, theta_deg):
    """
    返回绕点 (x, y) 逆时针旋转 theta_deg 度的 3x3 变换矩阵
    """
    theta_rad = np.radians(theta_deg)

    # 旋转矩阵
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # 3x3 齐次变换矩阵
    transform_matrix = np.array([
        [cos_theta, -sin_theta, x - x * cos_theta + y * sin_theta],
        [sin_theta,  cos_theta, y - x * sin_theta - y * cos_theta],
        [0,          0,         1]
    ])

    return transform_matrix

def rotate_around_point(points, x, y, theta_deg):
    """
    将给定 points 绕点 (x, y) 逆时针旋转 theta_deg 度

    Parameters:
    - points : np.ndarray of shape (..., 3) or (..., 2), 待旋转点云
    - x : float, center x
    - y : float, center y
    - theta_deg : float, 旋转角度
    """
    tf = rotate_matrix_around_point(x, y, theta_deg)

    if points.shape[-1] == 3:
        return points @ tf.T
    elif points.shape[-1] == 2:
        return points @ tf[:2, :2].T + tf[:2, 2]
    else:
        raise ValueError("Invalid shape of points, should be (..., 3) or (..., 2)")

class MarkerConfigFlash(ctypes.Structure):
    _fields_ = [
        ("x0", ctypes.c_int32),
        ("y0", ctypes.c_int32),
        ("dx", ctypes.c_float),
        ("dy", ctypes.c_float),
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

@dataclass
class MarkerConfig:
    x0: int = 27
    y0: int = 59
    dx: int = 43
    dy: int = 36
    theta: float = 0.0
    ncol: int = 9
    nrow: int = 18
    width: int = 400
    height: int = 700
    lower_threshold: int = 16
    min_area: int = 60

    def __post_init__(self):
        if not isinstance(self.x0, int):
            raise TypeError(f"Expected x0 to be an int, got {type(self.x0)}")
        if not isinstance(self.y0, int):
            raise TypeError(f"Expected y0 to be an int, got {type(self.y0)}")
        # if not isinstance(self.dy, int):
        #     raise TypeError(f"Expected dy to be an int, got {type(self.dy)}")
        # if not isinstance(self.dx, int):
        #     raise TypeError(f"Expected dx to be an int, got {type(self.dx)}")
        if not isinstance(self.ncol, int):
            raise TypeError(f"Expected ncol to be an int, got {type(self.ncol)}")
        if not isinstance(self.nrow, int):
            raise TypeError(f"Expected nrow to be an int, got {type(self.nrow)}")
        if not isinstance(self.width, int):
            raise TypeError(f"Expected width to be an int, got {type(self.width)}")
        if not isinstance(self.height, int):
            raise TypeError(f"Expected height to be an int, got {type(self.height)}")
    

    def get_grid(self, padx=0, pady=0) -> np.ndarray:
        X = np.linspace(self.x0 - padx * self.dx, self.x0 + self.dx * (self.ncol - 1 + padx), self.ncol + padx*2)
        Y = np.linspace(self.y0 - pady * self.dy, self.y0 + self.dy * (self.nrow - 1 + pady), self.nrow + pady*2)
        grid = list(np.meshgrid(X, Y, indexing="xy"))
        grid.append(np.ones_like(grid[0]))
        grid = np.stack(grid, axis=-1)  # x, y, 1 齐次坐标

        # 绕中心点旋转
        tf = rotate_matrix_around_point(self.width//2, self.height//2, self.theta)
        grid = grid @ tf.T
        return grid  # x, y, 3

    def get_direct(self):
        theta_rad = np.radians(self.theta)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        rot = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        direct = np.array([
            [self.dx, 0],  # 右
            [-self.dx, 0], # 左
            [0, self.dy],  # 下
            [0, -self.dy], # 上
        ]) @ rot.T

        return direct

    def write_flash(self, device_number):
        aquire_feild_names = [field for field,_ in MarkerConfigFlash._fields_]
        field_values = {field: getattr(self, field) for field in aquire_feild_names}
        # field_values['dx'], field_values['dy'] = int(field_values['dx']), int(field_values['dy'])
        xense_flash_manager.saveMarkerConfig(device_number, field_values)

    def read_flash(self, device_number):
        data_dict = xense_flash_manager.readMarkerConfig(device_number)
        self.dx = data_dict['dx']
        self.dy = data_dict['dy']
        self.theta = data_dict['theta']
        self.x0 = data_dict['x0']
        self.y0 = data_dict['y0']
        self.ncol = data_dict['ncol']
        self.nrow = data_dict['nrow']
        self.height = data_dict['height']
        self.width = data_dict['width']
        self.lower_threshold = data_dict['lower_threshold']
        self.min_area = data_dict['min_area']