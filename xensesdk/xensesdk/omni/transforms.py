"""
Description: This file contains the implementation of the image transformation.

Author: Jin Liu
Date: 2024/09/30
"""

import numpy as np
import cv2
from xensesdk.ezgl.experimental import cuda_interp, cuda_inverse_map

def init_marker_based_rectify_raw_py(marker_grids, w, h, down_sample_rate, pad):
    if isinstance(marker_grids, list):
        marker_grids = np.array(marker_grids)

    row_n, col_n = marker_grids.shape[:2]
    w /= down_sample_rate
    h /= down_sample_rate
    x_l, x_h = 0.0, w
    y_l, y_h = 0.0, h
    ratio = w / h

    x_max = x_h + pad - 1
    x_min = x_l - pad
    y_max = y_h + pad / ratio - 1
    y_min = y_l - pad / ratio

    x_list = np.linspace(x_min, x_max, col_n, dtype=np.float32)
    y_list = np.linspace(y_min, y_max, row_n, dtype=np.float32)

    mapxy = np.zeros((int(h), int(w), 2), dtype=np.float32)

    x_interval = (x_max - x_min) / (col_n - 1)
    y_interval = (y_max - y_min) / (row_n - 1)

    for y in range(int(h)):
        for x in range(int(w)):
            x_idx = int((x - x_min) / x_interval)
            y_idx = int((y - y_min) / y_interval)

            x_idx = min(max(x_idx, 0), col_n - 2)
            y_idx = min(max(y_idx, 0), row_n - 2)

            P11 = marker_grids[y_idx, x_idx]
            P12 = marker_grids[y_idx, x_idx + 1]
            P21 = marker_grids[y_idx + 1, x_idx]
            P22 = marker_grids[y_idx + 1, x_idx + 1]

            x1 = x_list[x_idx]
            x2 = x_list[x_idx + 1]
            y1 = y_list[y_idx]
            y2 = y_list[y_idx + 1]

            u1 = x - x1
            u2 = x2 - x
            v1 = y - y1
            v2 = y2 - y

            ret = (P11 * u2 * v2 + P12 * u1 * v2 + P21 * u2 * v1 + P22 * u1 * v1) / ((x2 - x1) * (y2 - y1))
            mapxy[y, x] = ret

    # split mapxy into mapx and mapy
    mapx, mapy = mapxy[:, :, 0], mapxy[:, :, 1]

    return mapx, mapy, x_list, y_list

def init_marker_based_rectify_ti(marker_grids, w, h, down_sample_rate, pad):
    """taichi version"""
    if isinstance(marker_grids, list):
        marker_grids = np.array(marker_grids)

    row_n, col_n = marker_grids.shape[:2]
    w /= down_sample_rate
    h /= down_sample_rate
    x_l, x_h = 0.0, w
    y_l, y_h = 0.0, h
    ratio = (col_n - 1) / (row_n - 1)  # x / y

    x_max = x_h + pad[3] / down_sample_rate - 1
    x_min = x_l - pad[2] / down_sample_rate
    y_max = y_h + pad[1] / ratio / down_sample_rate - 1
    y_min = y_l - pad[0] / ratio / down_sample_rate

    x_list = np.linspace(x_min, x_max, col_n, dtype=np.float32)
    y_list = np.linspace(y_min, y_max, row_n, dtype=np.float32)

    mapxy = np.zeros((int(h), int(w), 2), dtype=np.float32)

    x_interval = (x_max - x_min) / (col_n - 1)
    y_interval = (y_max - y_min) / (row_n - 1)

    cuda_interp(mapxy, marker_grids, x_list, y_list, x_interval, y_interval)

    # split mapxy into mapx and mapy
    mapx, mapy = mapxy[:, :, 0], mapxy[:, :, 1]

    return mapx, mapy, x_list, y_list


def get_regular_grid(row=3, col=3, imgsz=(640, 480)):
    w, h = imgsz
    X = np.linspace(0, w, col, dtype=np.float32)
    Y = np.linspace(0, h, row, dtype=np.float32)
    grid = np.stack(np.meshgrid(X, Y, indexing="xy"), axis=2)
    return grid  # shape: (row, col, 2)



class TransformBase:

    def __call__(self, x):
        return x


class Rectify(TransformBase):
    """
    图片校正
    """
    def __init__(self, nrow, ncol, img_sz, src_grid=None, downsample=1, pad=(0,0,0,0), border_value=(0, 0, 0)):
        """
        将非规则的 src_grid 映射到规则的网格上

        Parameters:
        - nrow : int, 默认 src_grid 的行数, 只在 src_grid 为 None 时有效
        - ncol : int, 默认 src_grid 的列数
        - img_sz : tuple(w, h), 目标图像大小
        - src_grid : np.ndarray(nrow, ncol, 2), optional, default: None, 每个点的坐标, xy 顺序
        - downsample : int, optional, default: 1, _description_
        - pad : tuple[int, int, int, int], optional, default: [0,0,0,0], 规则网格距离边界的距离
        - border_value : tuple, optional, default: (0, 0, 0), 边界填充值
        """
        self.nrow = nrow
        self.ncol = ncol
        self.img_sz = img_sz
        self.downsample = downsample
        if src_grid is None:
            self.src_grid = get_regular_grid(nrow, ncol, img_sz)
        else:
            self.src_grid = np.array(src_grid, np.float32)
        self.pad = pad
        self.border_value = border_value
        self.set()

    def set(self, src_grid=None, pad=None, img_sz=None):
        if img_sz is not None:
            self.img_sz = img_sz
        if src_grid is not None:
            self.src_grid = np.array(src_grid, np.float32)
        if pad is not None:
            self.pad = pad

        self._mapx, self._mapy, self._x_list, self._y_list = init_marker_based_rectify_ti(
            self.src_grid, self.img_sz[0], self.img_sz[1], self.downsample, self.pad
        )

        self.dst_grid = np.stack(np.meshgrid(self._x_list, self._y_list, indexing="xy"), axis=2)

    def set_point(self, pos):
        """
        查找最近的点并替换
        """
        dist = np.sum((self.src_grid - pos) ** 2, axis=2)
        min_idx = np.argmin(dist)
        self.src_grid[min_idx // self.ncol, min_idx % self.ncol] = pos
        self.set()

    def __call__(self, img):
        return cv2.remap(img, self._mapx, self._mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.border_value)


class Rectify2():
    """
    Rectify: 将 src 中定义的不规则 src_grid 变换到 dst 中定义的规则网格(regular_dst_grid)上, regular_dst_grid 由 src_grid 的行列数和 pad 自动生成。dst 的图像大小与 src 不必一致
    RectifyInverse: 进行反向操作
    """
    def __init__(self, src_size, src_grid, dst_size, pad=(0,0,0,0), border_value=(0, 0, 0), is_inverse=True, super_resolution=1):
        """
        Parameters:
        - src_size : tuple(w1, h1), 原始图像尺寸
        - src_grid : np.ndarray(nrow, ncol, 2), 目标网格点坐标
        - dst_size : tuple(w2, h2), 目标图像尺寸
        - pad : tuple[int, int, int, int], optional, default: [0,0,0,0], 规则网格距离边界的距离
        - border_value : tuple, optional, default: (0, 0, 0), 填充值
        - is_inverse : bool, optional, default: True, 是否进行逆向映射
        - super_resolution : int, optional, default: 1, 超分辨率倍数, 用于解决 inverse 时空缺值的问题, 越大越精确
        """
        self.src_size = src_size
        self.src_grid = src_grid
        self.dst_size = dst_size
        self.pad = pad
        self.border_value = border_value
        self.super_resolution = super_resolution
        self.is_inverse = is_inverse
        self.set()
    
    def set(self, src_grid=None, pad=None):
        if src_grid is not None:
            self.src_grid = np.array(src_grid, np.float32)
        if pad is not None:
            self.pad = pad
        # 正向映射, sizeof(_mapx, _mapy) == dst_size
        self._mapx, self._mapy, self._x_list, self._y_list = init_marker_based_rectify_ti(
            self.src_grid, self.dst_size[0], self.dst_size[1], 1, self.pad
        )
        self.dst_grid = np.stack(np.meshgrid(self._x_list, self._y_list, indexing="xy"), axis=2)

        # 逆向映射, sizeof(_imapx, _imapy) == src_size
        if self.is_inverse:
            _mapx, _mapy = self._mapx / self.super_resolution, self._mapy / self.super_resolution
            imap_shape = (self.src_size[1] // self.super_resolution, self.src_size[0] // self.super_resolution)
            _imapx, _imapy = np.full(imap_shape, -1, dtype=self._mapx.dtype), np.full(imap_shape, -1, dtype=self._mapx.dtype)
            cuda_inverse_map(_mapx, _mapy, _imapx, _imapy)
            if self.super_resolution != 1:
                self._imapx = cv2.resize(_imapx, self.src_size, interpolation=cv2.INTER_LINEAR)
                self._imapy = cv2.resize(_imapy, self.src_size, interpolation=cv2.INTER_LINEAR)
            else:
                self._imapx, self._imapy = _imapx, _imapy
    
    def set_point(self, pos):
        """
        查找最近的点并替换
        """
        dist = np.sum((self.src_grid - pos) ** 2, axis=2)
        min_idx = np.argmin(dist)
        self.src_grid[min_idx // self.src_grid.shape[1], min_idx % self.src_grid.shape[1]] = pos
        self.set()
        
    def __call__(self, img, border_value=None, inverse=True):
        border_value = border_value if border_value is not None else self.border_value
        if self.is_inverse and inverse:  # dst -> src
            if img.shape[0] != self.dst_size[1] or img.shape[1] != self.dst_size[0]:
                img = cv2.resize(img, self.dst_size, interpolation=cv2.INTER_LINEAR)
            return cv2.remap(img, self._imapx, self._imapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        else:  # src -> dst
            if img.shape[0] != self.src_size[1] or img.shape[1] != self.src_size[0]:
                img = cv2.resize(img, self.src_size, interpolation=cv2.INTER_LINEAR)
            return cv2.remap(img, self._mapx, self._mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    
class Diff(TransformBase):
    """
    图片差分
    """
    def __init__(self, ref_img=None, scale=1.5):
        self._ref_img = ref_img.astype(np.float32) if isinstance(ref_img, np.ndarray) else None
        self._scale = scale

    def __call__(self, img):
        if self._ref_img is None:
            return img
        return np.clip((img - self._ref_img) * self._scale + 110, 0, 255).astype(np.uint8)

    def set(self, ref_img=None, scale=1.5):
        self._scale = scale
        if isinstance(ref_img, np.ndarray):
            self._ref_img = ref_img.astype(np.float32)

class GaussianBlur(TransformBase):
    """
    高斯模糊
    """
    def __init__(self, ksize=(41, 41), sigma=20, times=1):
        self.ksize = ksize
        self.sigma = sigma
        self.times = times

    def __call__(self, img):
        for _ in range(self.times):
            img = cv2.GaussianBlur(img, self.ksize, self.sigma)
        return img
