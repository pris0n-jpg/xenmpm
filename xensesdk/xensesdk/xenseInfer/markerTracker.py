"""
Description: FlowTracker, 用于跟踪光流运动

Author: Jin Liu
Date: 2025/05/13
"""

import numpy as np
import cv2
from xensesdk.xenseInterface.sensorEnum import SensorType


class FlowTracker:
    def __init__(self, sensor_type: SensorType, grid_coord_size, border_size=2):
        """
        Marker Tracker, 用于跟踪斑点的运动
        border_size: 边界大小, 丢弃的边缘数
        """
        if sensor_type in [SensorType.Omni, SensorType.VecTouch, SensorType.OmniB, SensorType.Finger]:
            self.grid_shape = (30, 18)  # 包括边缘
        self._width = grid_coord_size[0]
        self._height = grid_coord_size[1]
        
        X = np.linspace(0, self._width-1, self.grid_shape[1], dtype=np.float32)
        Y = np.linspace(0, self._height-1, self.grid_shape[0], dtype=np.float32)
        grid = np.meshgrid(X, Y, indexing="xy")
        grid = np.stack(grid, axis=-1)  # 30x18x2
        self.border_size = border_size  # 边界大小, 丢弃的边缘数
        self._grid_init = grid[border_size:-border_size, border_size:-border_size].copy().astype(np.int32)  # 第一帧网格点像素坐标
        self.reset()

    @property
    def grid_init(self):
        return self._grid_init  # 第一帧网格点像素坐标

    @property
    def grid_curr(self):
        return self._grid_curr
    
    def reset(self):
        self._grid_curr = self._grid_init.copy()  # 当前网格点像素坐标

    def update(self, flow, max_flow=60):
        """
        更新当前网格点像素坐标
        :param flow: 流场，[60, 36, 2]
        """
        flow = cv2.resize(flow, (self.grid_shape[1], self.grid_shape[0]), interpolation=cv2.INTER_LINEAR) * max_flow
        self._grid_curr = self._grid_init + flow[self.border_size:-self.border_size, self.border_size:-self.border_size].astype(np.int32)

        return self._grid_curr
    
    def draw_marker(self, img, grid, color=(3, 253, 253), radius=2, thickness=2):
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        else:
            img = img.copy()

        grid = grid.reshape(-1, 2)
        
        if img.shape[1] != self._width or img.shape[0] != self._height:
            img = cv2.resize(img, (self._width, self._height))

        for dot in grid:
            cv2.circle(img, (int(dot[0]), int(dot[1])),
                    radius=radius, color=color, thickness=thickness, lineType=cv2.LINE_AA)  # 黄点
        return img
    
    def draw_move(self, img, alpha=0.8, draw_arrow=True, show_text=False):
        """绘制运动轨迹, alpha 箭头的不透明度

        :param img: 输入图片
        """
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        else:
            img = img.copy()

        if img.shape[1] != self._width or img.shape[0] != self._height:
            img = cv2.resize(img, (self._width, self._height))

        if alpha < 1:
            img_base = img.copy()

        arrow_color= (3, 253, 253)
        circle_color= (3, 253, 253)
        
        grid = self._grid_curr.astype(np.int32)
        for i, row in enumerate(grid):  # 丢弃最外围点
            for j, dot in enumerate(row):
                cv2.circle(img, dot, radius=2 , color=circle_color, thickness=2, lineType=cv2.LINE_AA)  # 黄点

                if show_text:
                    cv2.putText(img, str(f'{i},{j}'), dot+(0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 1)

                if not draw_arrow:
                    continue

                arrow_len = np.linalg.norm(self._grid_init[i, j] - self._grid_curr[i, j], axis=0)
                arrow_len = max(0.1, arrow_len)
                cv2.arrowedLine(
                    img, self._grid_init[i, j], self._grid_curr[i, j],
                    arrow_color, 2, tipLength=0.3/arrow_len*10,
                    line_type=cv2.LINE_AA
                )

        if alpha < 1:
            img = cv2.addWeighted(img, alpha, img_base, 1-alpha, 0)

        return img

# Visualization
def draw_marker_track_config(img, row, col, x0, y0, dx, dy, win_size, **kwargs):
    xs = np.linspace(x0, x0+dx*(col-1), col, dtype=int)
    ys = np.linspace(y0, y0+dy*(row-1), row, dtype=int)
    for i in range(row):
        for j in range(col):
            cv2.ellipse(img, (xs[j], ys[i]), (3,3) ,0 ,0 ,360, (3, 3, 253), -1)
            cv2.circle(img, (xs[j], ys[i]), win_size//2 , (3, 253, 253),  1)
    return img


def draw_grid(img, grid_cnt, **kwargs):
    """绘制nxn规则网格"""
    if grid_cnt == 0:
        return img

    color = (255,255,255)
    thickness = 1

    h, w = img.shape[:2]
    dy, dx = h / (grid_cnt+1), w / (grid_cnt+1)

    # draw vertical lines
    for i in range(1, grid_cnt+1):
        x = int(i * dx)
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for i in range(1, grid_cnt+1):
        y = int(i * dy)
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img


def draw_marker(img, grid, color=(3, 253, 253), radius=2, thickness=1):
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    else:
        img = img.copy()

    grid = grid.reshape(-1, 2)
    for dot in grid:
        cv2.circle(img, (int(dot[0]), int(dot[1])),
                   radius=radius, color=color, thickness=thickness, lineType=cv2.LINE_AA)  # 黄点
    return img


def draw_arrow(img, start_grid, end_grid, color=(0, 0, 255), draw_start=True):
    start_grid = start_grid.astype(np.int32).reshape(-1, 2)
    end_grid = end_grid.astype(np.int32).reshape(-1, 2)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    else:
        img = img.copy()

    for i in range(start_grid.shape[0]):
        if draw_start:
            cv2.circle(img, start_grid[i], 2, (0, 220, 220), 2)
        cv2.arrowedLine(img, start_grid[i], end_grid[i], color, 2, tipLength=0.2, line_type=cv2.LINE_AA)

    return img

