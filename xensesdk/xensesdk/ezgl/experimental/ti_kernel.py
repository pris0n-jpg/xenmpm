import os
import sys
import numpy as np
from contextlib import contextmanager

os.environ['ENABLE_TAICHI_HEADER_PRINT'] = '0'
import taichi as ti

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

with suppress_stdout():
    ti.init(arch=ti.cuda, log_level=ti.ERROR, verbose=False)


#-- kernel code --

@ti.kernel
def _ti_compute_normals(
    vertices: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),  # type: ignore
    indices: ti.types.ndarray(dtype=ti.i32, ndim=2),  # type: ignore
    normals: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),  # type: ignore
):
    # 累加面法线到相应的顶点法线
    for i in range(indices.shape[0]):
        idx1 = indices[i, 0]
        idx2 = indices[i, 1]
        idx3 = indices[i, 2]

        v1 = vertices[idx2] - vertices[idx1]
        v2 = vertices[idx3] - vertices[idx1]

        # 计算法线（未归一化）
        normal = v1.cross(v2)

        normals[idx1] += normal
        normals[idx2] += normal
        normals[idx3] += normal

    # 归一化法线
    for i in range(normals.shape[0]):
        normals[i] = normals[i].normalized()

def ti_compute_normals(vertices, indices):
    normals = np.zeros_like(vertices, dtype=np.float32)
    _ti_compute_normals(vertices, indices, normals)
    return normals

@ti.kernel
def _ti_compute_normals_quad(
    vertices: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
    indices: ti.types.ndarray(dtype=ti.i32, ndim=2),
    normals: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
):
    for i in range(indices.shape[0]):
        idx1 = indices[i, 0]
        idx2 = indices[i, 1]
        idx3 = indices[i, 2]
        idx4 = indices[i, 3]

        v1 = vertices[idx2] - vertices[idx1]
        v2 = vertices[idx3] - vertices[idx1]

        # 计算法线（未归一化）
        normal1 = v1.cross(v2)

        v1 = vertices[idx3] - vertices[idx2]
        v2 = vertices[idx4] - vertices[idx2]

        # 计算法线（未归一化）
        normal2 = v1.cross(v2)

        normals[idx1] += normal1
        normals[idx2] += normal1
        normals[idx3] += normal1

        normals[idx2] += normal2
        normals[idx3] += normal2
        normals[idx4] += normal2

    # 归一化法线
    for i in range(normals.shape[0]):
        normals[i] = normals[i].normalized()

def ti_compute_normals_quad(vertices, indices):
    normals = np.zeros_like(vertices, dtype=np.float32)
    _ti_compute_normals_quad(vertices, indices, normals)
    return normals


# 根据双线性插值计算映射
@ti.kernel
def ti_interp(
    mapxy: ti.types.ndarray(dtype=ti.math.vec2, ndim=2),  # type: ignore
    marker_grids: ti.types.ndarray(dtype=ti.math.vec2, ndim=2),  # type: ignore
    x_list: ti.types.ndarray(dtype=ti.f32, ndim=1),  # type: ignore
    y_list: ti.types.ndarray(dtype=ti.f32, ndim=1),  # type: ignore
    x_interval: ti.f32,  # type: ignore
    y_interval: ti.f32,  # type: ignore
):
    col_n = x_list.shape[0]
    row_n = y_list.shape[0]
    x_min = x_list[0]
    y_min = y_list[0]

    for y, x in mapxy:
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


# @ti.kernel
# def ti_inverse_map(
#     mapx: ti.types.ndarray(dtype=ti.f32, ndim=2),  # type: ignore
#     mapy: ti.types.ndarray(dtype=ti.f32, ndim=2),  # type: ignore
#     imapx: ti.types.ndarray(dtype=ti.f32, ndim=2),  # type: ignore
#     imapy: ti.types.ndarray(dtype=ti.f32, ndim=2),  # type: ignore
# ):
#     height, width = mapx.shape
#     ih, iw = imapx.shape
#     for y, x in ti.ndrange(height, width):  # 遍历 mapx, mapy 的每个像素
#         src_x = int(mapx[y, x])  # 源点 x 坐标
#         src_y = int(mapy[y, x])  # 源点 y 坐标
#         if 0 <= src_x < iw and 0 <= src_y < ih:  # 确保索引在范围内
#             imapx[src_y, src_x] = x
#             imapy[src_y, src_x] = y