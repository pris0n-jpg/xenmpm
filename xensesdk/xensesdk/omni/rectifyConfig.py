from dataclasses import dataclass, field
import numpy as np
from typing import Union
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

def scale_around_point(points, x, y, scale):
    """
    将给定 points 绕点 (x, y) 缩放 scale 倍

    Parameters:
    - points : np.ndarray of shape (..., 3) or (..., 2), 待缩放点云
    - x : float, center x
    - y : float, center y
    - scale : float, 缩放比例
    """
    tf = np.array([
        [scale, 0, x - scale * x],
        [0, scale, y - scale * y],
        [0, 0, 1]
    ])

    if points.shape[-1] == 3:
        return points @ tf.T
    elif points.shape[-1] == 2:
        return points @ tf[:2, :2].T + tf[:2, 2]
    else:
        raise ValueError("Invalid shape of points, should be (..., 3) or (..., 2)")

def rot_trans_grid(grid, angle, x, y):
    grid = np.array(grid, dtype=np.float32)
    center_x, center_y = grid[3][5] + grid[3][6]
    center_x += x
    center_y += y
    angle = np.deg2rad(angle)  # 度转换为弧度
    # 旋转矩阵
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    # 旋转每个点

    rotated_grid = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # 计算相对中心点的坐标
            relative_pos = grid[i, j]  - np.array([center_x, center_y])
            # 应用旋转矩阵
            rotated_pos = rotation_matrix @ relative_pos
            # 计算旋转后的实际位置
            rotated_grid[i, j] = rotated_pos + np.array([center_x, center_y])

    return rotated_grid

def scale_around_point(points, x, y, scale):
    """
    将给定 points 绕点 (x, y) 缩放 scale 倍

    Parameters:
    - points : np.ndarray of shape (..., 3) or (..., 2), 待缩放点云
    - x : float, center x
    - y : float, center y
    - scale : float, 缩放比例
    """
    tf = np.array([
        [scale, 0, x - scale * x],
        [0, scale, y - scale * y],
        [0, 0, 1]
    ])

    if points.shape[-1] == 3:
        return points @ tf.T
    elif points.shape[-1] == 2:
        return points @ tf[:2, :2].T + tf[:2, 2]
    else:
        raise ValueError("Invalid shape of points, should be (..., 3) or (..., 2)")

@dataclass
class RectifyConfig:
    width: int = 400
    height: int = 700  # not saved to config
    rot: float = 0.0
    trans_x: int = 0
    trans_y: int = 0
    base_grid: Union[list, np.ndarray] = field(default_factory= lambda :[[[126.298, 146.471],
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
  [609.587, 445.913]]])
    scale: float = 1.0
    padding: list = field(default_factory= lambda :[0,0,0,0]) # top bot left right

    def __post_init__(self):
        if not isinstance(self.scale, float):
            raise TypeError(f"Expected scale to be an float, got {type(self.scale)}")
        if not isinstance(self.trans_x, int):
            raise TypeError(f"Expected trans_x to be an int, got {type(self.trans_x)}")
        if not isinstance(self.trans_y, int):
            raise TypeError(f"Expected trans_y to be an int, got {type(self.trans_y)}")
        if not isinstance(self.base_grid, list) and  not isinstance(self.base_grid, np.ndarray):
            raise TypeError(f"Expected base_grid to be an list, got {type(self.base_grid)}")
        if not isinstance(self.rot, float):
            raise TypeError(f"Expected rot to be an float, got {type(self.rot)}")
        if not isinstance(self.padding, list):
            raise TypeError(f"Expected padding to be a list, got {type(self.padding)}")

    def get_grid(self) -> np.ndarray:
        grid = rotate_around_point(np.array(self.base_grid), 200, 350, self.rot) + (self.trans_y, self.trans_x)
        grid = scale_around_point(grid, 200, 350, self.scale).transpose(1,0,2).copy()
        return grid
    
    def read_config(self, config_data):
        self.rot = config_data['rot']
        self.trans_x = int(config_data['trans_x'])
        self.trans_y = int(config_data['trans_y'])
        self.scale = config_data['scale']
        self.padding = config_data['padding']
        self.base_grid = config_data['base_grid']

    def read_flash(self, device_number):
        config_data = xense_flash_manager.readRectifyConfig(device_number)
        self.read_config(config_data)

    def write_flash(self, device_number):
        np_grid = np.array(self.base_grid)
        cols, rows, _ = np_grid.shape # cols rows, 2
        base_grid = (ctypes.c_int32 * (rows * cols * 2))(*np_grid.flatten().astype(int).tolist())
        padding = (ctypes.c_int32 * 4)(*self.padding)
        data_dict = {
            "rot":self.rot,
            "trans_x":self.trans_x,
            "trans_y":self.trans_y,
            "scale":self.scale,
            "padding":padding,
            "rows":rows,
            "cols":cols,
            "base_grid":base_grid
        }
        xense_flash_manager.saveRectifyConfig(device_number, data_dict)