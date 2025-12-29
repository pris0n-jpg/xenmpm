"""
MPM vs FEM Sensor RGB Comparison

This script compares MPM and FEM simulations using the visuotactile sensor
RGB rendering approach. Both methods render to the same sensor image format
for direct visual comparison.

Features:
- FEM path: Uses existing VecTouchSim rendering pipeline
- MPM path: Extracts top-surface height field and renders with matching style
- Side-by-side visualization with raw/diff modes
- Configurable press + slide trajectory

Usage:
    python example/mpm_fem_rgb_compare.py --mode raw
    python example/mpm_fem_rgb_compare.py --mode diff --press-mm 1.5
    python example/mpm_fem_rgb_compare.py --save-dir output/rgb_compare

    # Recommended baseline (stable, auditable output):
    python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/baseline

Output (when --save-dir is set):
    - fem_XXXX.png / mpm_XXXX.png
    - run_manifest.json (resolved params + frame->phase mapping)

Requirements:
    - xengym conda environment with Taichi installed
    - FEM data file (default: assets/data/fem_data_gel_2035.npz)
"""
import argparse
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import sys
import ast
import re
import datetime
import struct

# Add project root to path for standalone execution
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project imports
try:
    from xengym import PROJ_DIR
except ImportError:
    PROJ_DIR = _PROJECT_ROOT / "xengym"

# Attempt optional imports
try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False
    print("Warning: Taichi not available, MPM mode disabled")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Attempt ezgl/render imports
try:
    from xensesdk.ezgl import tb, Matrix4x4
    from xensesdk.ezgl.items.scene import Scene
    from xensesdk.ezgl.experimental.GLSurfMeshItem import GLSurfMeshItem
    from xensesdk.ezgl.items import (
        GLModelItem,
        GLBoxItem,
        GLMeshItem,
        DepthCamera,
        RGBCamera,
        PointLight,
        LineLight,
        GLAxisItem,
        Texture2D,
        Material,
    )
    from xensesdk.ezgl.items.MeshData import sphere as ezgl_mesh_sphere
    from xengym.render import VecTouchSim
    from xengym import ASSET_DIR
    HAS_EZGL = True
except Exception as e:
    HAS_EZGL = False
    # Define placeholder classes when ezgl is not available
    class Scene:
        """Placeholder Scene class when ezgl is not available"""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ezgl not available")
    class Matrix4x4:
        """Placeholder Matrix4x4 class"""
        pass
    class GLSurfMeshItem:
        """Placeholder GLSurfMeshItem class"""
        pass
    class GLModelItem:
        """Placeholder GLModelItem class"""
        pass
    class GLBoxItem:
        """Placeholder GLBoxItem class"""
        pass
    class GLMeshItem:
        """Placeholder GLMeshItem class"""
        pass
    class DepthCamera:
        """Placeholder DepthCamera class"""
        pass
    class RGBCamera:
        """Placeholder RGBCamera class"""
        pass
    class PointLight:
        """Placeholder PointLight class"""
        pass
    class LineLight:
        """Placeholder LineLight class"""
        pass
    class Texture2D:
        """Placeholder Texture2D class"""
        pass
    class Material:
        """Placeholder Material class"""
        pass
    tb = None
    ASSET_DIR = Path(".")
    print(f"Warning: ezgl/xengym render not available ({e})")


# ==============================================================================
# Scene Parameters
# ==============================================================================
SCENE_PARAMS = {
    # Gel geometry (match VecTouchSim defaults)
    'gel_size_mm': (17.3, 29.15),       # width (x), height (y) in mm
    'gel_thickness_mm': 5.0,            # depth (z) in mm

    # Height field grid resolution (matches SensorScene)
    'height_grid_shape': (140, 80),     # (n_row, n_col)

    # Height-field postprocess (MPM -> render)
    'mpm_height_fill_holes': True,      # fill empty cells via diffusion (recommended to avoid hard-edge artifacts)
    'mpm_height_fill_holes_iters': 10,  # iterations for hole filling
    'mpm_height_smooth': True,          # apply box smoothing before rendering
    'mpm_height_smooth_iters': 2,       # matches previous hardcoded behavior
    'mpm_height_reference_edge': True,  # use edge-region baseline alignment (preserve slide motion)
    # Height-field outlier suppression (MPM -> render)
    # NOTE: 仅作为“最后一道防线”，避免 footprint 外的异常深值把整块区域渲染成暗盘/彩虹 halo。
    # 默认关闭以保持基线行为不变；建议与 fill_holes 联用。
    'mpm_height_clip_outliers': False,
    'mpm_height_clip_outliers_min_mm': 5.0,  # 超过该负向深度（mm）的值会被视作离群并置为 NaN
    # IMPORTANT: MPM 接触采用 penalty 形式时，压头可能“穿透”粒子表面并导致高度场过深，
    # 从而出现非物理的“整块发黑/暗盘”。该开关会把 height_field 限制在“不低于压头表面”。
    'mpm_height_clamp_indenter': True,

    # Trajectory parameters
    # NOTE: 默认压深建议保持在 FEM 数据覆盖范围内（通常 <= 1mm），否则 MPM 侧容易出现“广域下陷”
    # 进而放大渲染伪影（暗块/halo），导致 MPM vs FEM 观感不可直接对比。
    'press_depth_mm': 1.0,              # target indentation depth (mm)
    'slide_distance_mm': 3.0,           # tangential travel (x direction)
    'press_steps': 150,                 # steps to reach press depth
    'slide_steps': 240,                 # steps for sliding phase
    'hold_steps': 40,                   # steps to hold at end

    # MPM simulation parameters
    'mpm_dt': 2e-5,                     # Reduced for stability with higher stiffness
    'mpm_grid_dx_mm': 0.4,              # grid spacing in mm
    'mpm_particles_per_cell': 2,        # particles per cell per dimension      

    # MPM grid padding (cells) to keep particles away from sticky boundaries.
    # MPMSolver.grid_op clamps grid velocities when I[d] < 3 or I[d] >= grid_size[d]-3.
    # Using >=6 padding cells provides a safety margin for contact/friction.
    'mpm_grid_padding_cells_xy': 6,
    'mpm_grid_padding_cells_z_bottom': 6,
    'mpm_grid_padding_cells_z_top': 20,

    # Material (soft gel)
    'density': 1000.0,                  # kg/m³
    'ogden_mu': [2500.0],               # Pa
    'ogden_alpha': [2.0],
    # NOTE: kappa 控制体积压缩性（kappa >> mu 时近似不可压缩）。
    # 早期的 2.5e4 Pa 会导致明显“广域下陷带”，MPM vs FEM 无法直接对比。
    # 这里默认对齐到 xengym MPM demo 常用量级（~3e5 Pa），并保留 CLI 覆盖入口。
    'ogden_kappa': 300000.0,            # Pa

    # Indenter (sphere)
    'indenter_radius_mm': 4.0,
    'indenter_start_gap_mm': 0.5,       # initial clearance above gel
    # Indenter (sphere/cylinder/box)
    # - cylinder: flat round pad, matches circle_r4.STL (tip face) better than box
    'indenter_type': 'cylinder',        # 'sphere' | 'cylinder' | 'box'
    'indenter_cylinder_half_height_mm': None,  # Optional half height for cylinder; default uses radius
    'indenter_half_extents_mm': None,   # Optional (x,y,z) for box mode; overrides indenter_radius_mm

    # Contact / friction (explicit defaults for auditability)
    'fem_fric_coef': 0.4,               # FEM fric_coef (single coefficient)
    'mpm_mu_s': 2.0,                    # MPM static friction (mu_s)
    'mpm_mu_k': 1.5,                    # MPM kinetic friction (mu_k)     
    'mpm_contact_stiffness_normal': 8e2,
    'mpm_contact_stiffness_tangent': 4e2,

    # Optional Kelvin-Voigt bulk viscosity (damping) for MPM
    'mpm_enable_bulk_viscosity': False,
    'mpm_bulk_viscosity': 0.0,          # Pa·s

    # Depth camera settings (for FEM path)
    'depth_img_size': (100, 175),       # matches demo_simple_sensor
    # IMPORTANT: For FEM depth->gel mapping, the depth camera ortho view should match gel_size_mm.
    # Keep these consistent to avoid implicit scaling mismatch between FEM and MPM.
    'cam_view_width_m': 0.0173,         # 17.3 mm (match gel_size_mm[0])
    'cam_view_height_m': 0.02915,       # 29.15 mm (match gel_size_mm[1])

    # Debug settings
    'debug_verbose': False,             # Enable verbose per-frame logging
}


def _infer_square_size_mm_from_stl_path(object_file: Optional[str]) -> Optional[float]:
    """
    从资产命名中推断正方形压头边长（mm）。

    约定：xengym/assets/obj/square_d6.STL -> d=6mm。
    """
    if not object_file:
        return None
    name = Path(object_file).name.lower()
    m = re.search(r"square_d(\d+)", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _analyze_binary_stl_endfaces_mm(stl_path: Path) -> Optional[Dict[str, object]]:
    """
    Analyze a binary STL and return simple end-face extents (mm) at y_min / y_max.

    This is used as a lightweight verification tool for assets like circle_r4.STL
    whose bottom (y_min) and top (y_max) faces may have very different contact footprints.
    """
    try:
        with stl_path.open("rb") as f:
            header = f.read(80)
            if len(header) != 80:
                return None
            tri_count_bytes = f.read(4)
            if len(tri_count_bytes) != 4:
                return None
            tri_count = struct.unpack("<I", tri_count_bytes)[0]
            if tri_count <= 0:
                return None
            raw = f.read(50 * tri_count)
            if len(raw) != 50 * tri_count:
                return None

        tri_dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("v1", "<f4", (3,)),
                ("v2", "<f4", (3,)),
                ("v3", "<f4", (3,)),
                ("attr", "<u2"),
            ]
        )
        if tri_dtype.itemsize != 50:
            return None

        tris = np.frombuffer(raw, dtype=tri_dtype, count=tri_count)
        verts = np.concatenate([tris["v1"], tris["v2"], tris["v3"]], axis=0).astype(np.float64)
        if verts.size == 0:
            return None

        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        y_min = float(bbox_min[1])
        y_max = float(bbox_max[1])
        height = max(y_max - y_min, 0.0)

        tol = max(1e-6, height * 1e-4)
        mask_min = verts[:, 1] <= (y_min + tol)
        mask_max = verts[:, 1] >= (y_max - tol)
        if not mask_min.any() or not mask_max.any():
            return None

        vmin = verts[mask_min]
        vmax = verts[mask_max]

        def _xz_extents_mm(v: np.ndarray) -> Dict[str, float]:
            x_min_m = float(v[:, 0].min())
            x_max_m = float(v[:, 0].max())
            z_min_m = float(v[:, 2].min())
            z_max_m = float(v[:, 2].max())
            return {
                "x_min_mm": x_min_m * 1000.0,
                "x_max_mm": x_max_m * 1000.0,
                "z_min_mm": z_min_m * 1000.0,
                "z_max_mm": z_max_m * 1000.0,
                "size_x_mm": (x_max_m - x_min_m) * 1000.0,
                "size_z_mm": (z_max_m - z_min_m) * 1000.0,
            }

        return {
            "path": str(stl_path).replace("\\", "/"),
            "triangles": int(tri_count),
            "bbox_min_mm": (bbox_min * 1000.0).tolist(),
            "bbox_max_mm": (bbox_max * 1000.0).tolist(),
            "y_min_mm": y_min * 1000.0,
            "y_max_mm": y_max * 1000.0,
            "height_mm": height * 1000.0,
            "endfaces_mm": {
                "y_min": _xz_extents_mm(vmin),
                "y_max": _xz_extents_mm(vmax),
            },
        }
    except Exception:
        return None


def _box_blur_2d_xy(values: np.ndarray, iterations: int = 1) -> np.ndarray:
    """对 (H,W,2) 的位移场做轻量 3x3 box blur。"""
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("values must be (H,W,2)")
    result = values.astype(np.float32, copy=True)
    for _ in range(max(iterations, 0)):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        result = (
            padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
            padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, 0:-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    return result


def _fill_uv_holes(uv: np.ndarray, valid_mask: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """
    使用扩散法填充 UV 场中的空洞（无粒子覆盖的网格单元）。

    Args:
        uv: (H,W,2) UV 位移场
        valid_mask: (H,W) bool，True 表示该单元有有效数据
        max_iterations: 最大扩散迭代次数

    Returns:
        填充后的 UV 场 (H,W,2)
    """
    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError("uv must be (H,W,2)")
    if valid_mask.shape != uv.shape[:2]:
        raise ValueError("valid_mask shape must match uv[:,:,0]")

    result = uv.astype(np.float32, copy=True)
    filled = valid_mask.copy()

    for _ in range(max_iterations):
        if filled.all():
            break  # 全部填充完成

        # 找到未填充但有邻居已填充的单元
        padded_filled = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        neighbor_count = (
            padded_filled[0:-2, 0:-2] + padded_filled[0:-2, 1:-1] + padded_filled[0:-2, 2:] +
            padded_filled[1:-1, 0:-2] +                            padded_filled[1:-1, 2:] +
            padded_filled[2:, 0:-2]   + padded_filled[2:, 1:-1]   + padded_filled[2:, 2:]
        )

        # 可填充的单元：未填充 且 至少有一个邻居已填充
        can_fill = (~filled) & (neighbor_count > 0)

        if not can_fill.any():
            break

        # 计算邻居的加权平均
        padded_uv = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0)
        padded_mask = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)

        neighbor_sum = np.zeros_like(result)
        neighbor_weight = np.zeros(result.shape[:2], dtype=np.float32)

        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            i_slice = slice(1 + di, result.shape[0] + 1 + di)
            j_slice = slice(1 + dj, result.shape[1] + 1 + dj)
            w = padded_mask[i_slice, j_slice]
            neighbor_sum += padded_uv[i_slice, j_slice] * w[..., None]
            neighbor_weight += w

        # 填充
        fill_mask = can_fill & (neighbor_weight > 0)
        result[fill_mask] = neighbor_sum[fill_mask] / neighbor_weight[fill_mask][..., None]
        filled[fill_mask] = True

    return result


def _fill_height_holes(height_mm: np.ndarray, valid_mask: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """
    使用扩散法填充高度场中的空洞（无粒子覆盖的网格单元）。

    Args:
        height_mm: (H,W) 高度场（mm，<=0 表示压入）
        valid_mask: (H,W) bool，True 表示该单元有有效数据
        max_iterations: 最大扩散迭代次数

    Returns:
        填充后的高度场 (H,W)
    """
    if height_mm.ndim != 2:
        raise ValueError("height_mm must be (H,W)")
    if valid_mask.shape != height_mm.shape:
        raise ValueError("valid_mask shape must match height_mm")

    # NOTE: height_mm may contain NaN for missing cells; keep NaN but ensure
    # they do not pollute neighbor aggregation (use nan_to_num when padding).
    result = height_mm.astype(np.float32, copy=True)
    filled = valid_mask.copy()

    for _ in range(max(max_iterations, 0)):
        if filled.all():
            break

        padded_filled = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)
        neighbor_count = (
            padded_filled[0:-2, 0:-2] + padded_filled[0:-2, 1:-1] + padded_filled[0:-2, 2:] +
            padded_filled[1:-1, 0:-2] +                            padded_filled[1:-1, 2:] +
            padded_filled[2:, 0:-2]   + padded_filled[2:, 1:-1]   + padded_filled[2:, 2:]
        )

        can_fill = (~filled) & (neighbor_count > 0)
        if not can_fill.any():
            break

        padded_h = np.pad(
            np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0),
            ((1, 1), (1, 1)),
            mode="constant",
            constant_values=0,
        )
        padded_mask = np.pad(filled.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0)

        neighbor_sum = np.zeros_like(result)
        neighbor_weight = np.zeros_like(result, dtype=np.float32)

        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            i_slice = slice(1 + di, result.shape[0] + 1 + di)
            j_slice = slice(1 + dj, result.shape[1] + 1 + dj)
            w = padded_mask[i_slice, j_slice]
            neighbor_sum += padded_h[i_slice, j_slice] * w
            neighbor_weight += w

        fill_mask = can_fill & (neighbor_weight > 0)
        result[fill_mask] = neighbor_sum[fill_mask] / neighbor_weight[fill_mask]
        filled[fill_mask] = True

    # Remaining holes (if any) are set to 0mm (flat), consistent with previous behavior.
    result[~filled] = 0.0
    return result


def warp_marker_texture(
    base_tex: np.ndarray,
    uv_disp_mm: np.ndarray,
    gel_size_mm: Tuple[float, float],
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    """
    使用面内位移场对 marker 纹理做非均匀 warp，使点阵体现拉伸/压缩/剪切。

    - uv_disp_mm: (Ny,Nx,2)，单位 mm，u=+x 向右，v=+y 向上（以“传感器平面坐标”约定）
    - 采用逆向映射：输出像素 (x,y) 从输入 base_tex 采样 (x - dx, y - dy)
    - flip_x/flip_y 用于处理 texcoords / mesh 的翻转约定差异
    """
    if base_tex.ndim != 3 or base_tex.shape[2] != 3:
        raise ValueError("base_tex must be (H,W,3)")
    if uv_disp_mm.ndim != 3 or uv_disp_mm.shape[2] != 2:
        raise ValueError("uv_disp_mm must be (Ny,Nx,2)")

    tex_h, tex_w = base_tex.shape[0], base_tex.shape[1]
    gel_w_mm, gel_h_mm = gel_size_mm

    # Upsample uv field to texture resolution using bilinear interpolation
    # (Previously used nearest-neighbor which caused blocky displacement artifacts)
    src_h, src_w = uv_disp_mm.shape[0], uv_disp_mm.shape[1]
    if HAS_CV2:
        # Use cv2.resize for smooth bilinear interpolation
        uv_up = cv2.resize(uv_disp_mm, (tex_w, tex_h), interpolation=cv2.INTER_LINEAR)
    else:
        # Numpy fallback: bilinear upsampling
        row_scale = (src_h - 1) / max(tex_h - 1, 1)
        col_scale = (src_w - 1) / max(tex_w - 1, 1)
        row_coords = np.arange(tex_h) * row_scale
        col_coords = np.arange(tex_w) * col_scale
        r0 = np.floor(row_coords).astype(np.int32)
        c0 = np.floor(col_coords).astype(np.int32)
        r1 = np.clip(r0 + 1, 0, src_h - 1)
        c1 = np.clip(c0 + 1, 0, src_w - 1)
        wr = (row_coords - r0).astype(np.float32)
        wc = (col_coords - c0).astype(np.float32)
        # Bilinear interpolation for each channel
        uv_up = np.zeros((tex_h, tex_w, 2), dtype=np.float32)
        for ch in range(2):
            src = uv_disp_mm[..., ch]
            top = src[r0[:, None], c0[None, :]] * (1 - wc[None, :]) + src[r0[:, None], c1[None, :]] * wc[None, :]
            bot = src[r1[:, None], c0[None, :]] * (1 - wc[None, :]) + src[r1[:, None], c1[None, :]] * wc[None, :]
            uv_up[..., ch] = top * (1 - wr[:, None]) + bot * wr[:, None]

    # mm -> px
    dx_px = (uv_up[..., 0] / max(gel_w_mm, 1e-6)) * tex_w
    dy_px = (uv_up[..., 1] / max(gel_h_mm, 1e-6)) * tex_h

    # Mesh/texcoords 翻转修正
    if flip_x:
        dx_px = -dx_px
    if flip_y:
        dy_px = -dy_px

    xx, yy = np.meshgrid(np.arange(tex_w, dtype=np.float32), np.arange(tex_h, dtype=np.float32))
    map_x = xx - dx_px.astype(np.float32)
    map_y = yy - dy_px.astype(np.float32)

    if HAS_CV2:
        # 与 numpy fallback 的 clip 行为保持一致，避免 remap 出界触发反射边界
        # （典型现象：边缘 marker 被抻成短线/拖影，干扰方向/量级归因）。
        map_x = np.clip(map_x, 0.0, tex_w - 1.001).astype(np.float32, copy=False)
        map_y = np.clip(map_y, 0.0, tex_h - 1.001).astype(np.float32, copy=False)
        return cv2.remap(
            base_tex,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    # Numpy fallback: bilinear sampling (slow but keeps functionality)
    map_x = np.clip(map_x, 0.0, tex_w - 1.001)
    map_y = np.clip(map_y, 0.0, tex_h - 1.001)
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, tex_w - 1)
    y1 = np.clip(y0 + 1, 0, tex_h - 1)
    wx = (map_x - x0).astype(np.float32)[..., None]
    wy = (map_y - y0).astype(np.float32)[..., None]

    Ia = base_tex[y0, x0].astype(np.float32)
    Ib = base_tex[y0, x1].astype(np.float32)
    Ic = base_tex[y1, x0].astype(np.float32)
    Id = base_tex[y1, x1].astype(np.float32)
    out = (Ia * (1 - wx) * (1 - wy) +
           Ib * wx * (1 - wy) +
           Ic * (1 - wx) * wy +
           Id * wx * wy)
    return np.clip(out, 0, 255).astype(np.uint8)


def _mpm_flip_x_field(field: np.ndarray) -> np.ndarray:
    """Apply MPM->render horizontal flip (x axis) to match mesh x_range convention."""
    return field[:, ::-1]


def _mpm_flip_x_mm(x_mm: float) -> float:
    """Apply MPM->render horizontal flip (x axis) for scalar coordinates (mm)."""
    return -float(x_mm)


def _compute_rgb_diff_metrics(a_rgb: np.ndarray, b_rgb: np.ndarray) -> Dict[str, float]:
    """
    Compute simple per-frame RGB difference metrics for audit/regression.

    Returns:
        dict with keys: mae, mae_r, mae_g, mae_b, max_abs, p50, p90, p99.
    """
    if a_rgb is None or b_rgb is None:
        raise ValueError("RGB inputs must not be None")
    if a_rgb.shape != b_rgb.shape:
        raise ValueError(f"RGB shape mismatch: {a_rgb.shape} vs {b_rgb.shape}")

    a16 = a_rgb.astype(np.int16, copy=False)
    b16 = b_rgb.astype(np.int16, copy=False)
    abs_diff = np.abs(a16 - b16).astype(np.float32, copy=False)

    p50, p90, p99 = [float(x) for x in np.percentile(abs_diff, [50, 90, 99]).tolist()]
    return {
        "mae": float(abs_diff.mean()),
        "mae_r": float(abs_diff[..., 0].mean()),
        "mae_g": float(abs_diff[..., 1].mean()),
        "mae_b": float(abs_diff[..., 2].mean()),
        "max_abs": float(abs_diff.max()),
        "p50": p50,
        "p90": p90,
        "p99": p99,
    }


def _sanitize_run_context_for_manifest(run_context: Dict[str, object]) -> Dict[str, object]:
    """
    Keep run_manifest diff-friendly by removing volatile/sensitive fields from run_context.

    Notes:
    - argv is already recorded in run_manifest.json; avoid duplicating save_dir or other
      environment-dependent paths in run_context.args.
    - Keep resolved.* for audit / alignment checks.
    """
    if not isinstance(run_context, dict):
        return {}
    sanitized: Dict[str, object] = {}
    resolved = run_context.get("resolved")
    if isinstance(resolved, dict):
        sanitized["resolved"] = resolved
    args = run_context.get("args")
    if isinstance(args, dict):
        args_copy = dict(args)
        args_copy.pop("save_dir", None)
        sanitized["args"] = args_copy
    return sanitized


def _write_tuning_notes(
    save_dir: Path,
    *,
    record_interval: int,
    total_frames: int,
    run_context: Dict[str, object],
    overwrite: bool = False,
    reason: Optional[str] = None,
) -> None:
    """
    Write a stable, human-editable tuning_notes.md alongside run_manifest.json.

    The file is intended to be diff-friendly (avoid timestamps) and can be edited
    by humans after the run without being clobbered by default.
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    notes_path = save_dir / "tuning_notes.md"
    if notes_path.exists() and not overwrite:
        return

    press_steps = int(SCENE_PARAMS["press_steps"])
    slide_steps = int(SCENE_PARAMS["slide_steps"])
    hold_steps = int(SCENE_PARAMS["hold_steps"])

    def _phase_for_step(step: int) -> str:
        if step < press_steps:
            return "press"
        if step < press_steps + slide_steps:
            return "slide"
        return "hold"

    frame_to_step = [int(i * int(record_interval)) for i in range(int(total_frames))]
    frame_to_phase = [_phase_for_step(step) for step in frame_to_step]

    phase_ranges: Dict[str, Dict[str, int]] = {}
    for i, phase in enumerate(frame_to_phase):
        if phase not in phase_ranges:
            phase_ranges[phase] = {"start_frame": i, "end_frame": i}
        else:
            phase_ranges[phase]["end_frame"] = i

    def _pick_mid(start: int, end: int) -> int:
        return int((int(start) + int(end)) // 2)

    key_frames: Dict[str, Optional[int]] = {
        "press_end": None,
        "slide_mid": None,
        "hold_end": None,
    }
    if "press" in phase_ranges:
        key_frames["press_end"] = int(phase_ranges["press"]["end_frame"])
    if "slide" in phase_ranges:
        key_frames["slide_mid"] = _pick_mid(phase_ranges["slide"]["start_frame"], phase_ranges["slide"]["end_frame"])
    if "hold" in phase_ranges:
        key_frames["hold_end"] = int(phase_ranges["hold"]["end_frame"])

    resolved = run_context.get("resolved") if isinstance(run_context, dict) else None
    resolved_dict: Dict[str, object] = resolved if isinstance(resolved, dict) else {}

    def _dump(obj: object) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

    lines: List[str] = []
    lines.append("# Tuning Notes (mpm_fem_rgb_compare)")
    lines.append("")
    lines.append("## Baseline command")
    lines.append("```json")
    lines.append(_dump(list(sys.argv)))
    lines.append("```")
    if reason:
        lines.append(f"- reason: `{str(reason)}`")
    lines.append("")
    lines.append("## Key frames (frame_id)")
    for k in ["press_end", "slide_mid", "hold_end"]:
        lines.append(f"- {k}: `{key_frames.get(k)}`")
    lines.append("")
    lines.append("## Resolved (for diff/audit)")
    for section in ["friction", "scale", "indenter", "conventions", "render", "export", "contact"]:
        if section in resolved_dict:
            lines.append(f"### {section}")
            lines.append("```json")
            lines.append(_dump(resolved_dict.get(section)))
            lines.append("```")
            lines.append("")
    lines.append("## Conclusion")
    lines.append("- (fill in) What changed, what improved, what to try next.")
    lines.append("")
    lines.append("## Repro / analysis")
    lines.append(f"- Run intermediate analysis: `python example/analyze_rgb_compare_intermediate.py --save-dir {save_dir}`")
    lines.append("")

    try:
        notes_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        print(f"Warning: failed to write tuning_notes.md: {e}")


def _write_preflight_run_manifest(
    save_dir: Path,
    record_interval: int,
    total_frames: int,
    run_context: Dict[str, object],
    *,
    reason: Optional[str] = None,
) -> None:
    """
    Write a minimal run_manifest.json before entering the heavy render/sim path.

    This keeps outputs auditable even when optional dependencies (ezgl/taichi)
    are missing in the current environment.
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    press_steps = int(SCENE_PARAMS["press_steps"])
    slide_steps = int(SCENE_PARAMS["slide_steps"])
    hold_steps = int(SCENE_PARAMS["hold_steps"])
    total_steps = press_steps + slide_steps + hold_steps

    def _phase_for_step(step: int) -> str:
        if step < press_steps:
            return "press"
        if step < press_steps + slide_steps:
            return "slide"
        return "hold"

    frame_to_step = [int(i * record_interval) for i in range(int(total_frames))]
    frame_to_phase = [_phase_for_step(step) for step in frame_to_step]
    phase_ranges: Dict[str, Dict[str, int]] = {}
    for i, phase in enumerate(frame_to_phase):
        if phase not in phase_ranges:
            phase_ranges[phase] = {"start_frame": i, "end_frame": i}
        else:
            phase_ranges[phase]["end_frame"] = i

    manifest: Dict[str, object] = {
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "argv": list(sys.argv),
        "run_context": _sanitize_run_context_for_manifest(run_context),
        "scene_params": dict(SCENE_PARAMS),
        "deps": {
            "has_taichi": bool(HAS_TAICHI),
            "has_ezgl": bool(HAS_EZGL),
            "has_cv2": bool(HAS_CV2),
        },
        "execution": {
            "stage": "preflight",
            "note": "Written before running; may be overwritten by the runtime manifest.",
            "reason": str(reason) if reason else None,
        },
        "trajectory": {
            "press_steps": press_steps,
            "slide_steps": slide_steps,
            "hold_steps": hold_steps,
            "total_steps": total_steps,
            "record_interval": int(record_interval),
            "total_frames": int(total_frames),
            "phase_ranges_frames": phase_ranges,
            "frame_to_step": frame_to_step,
            "frame_to_phase": frame_to_phase,
            "frame_controls": None,
        },
        "outputs": {
            "frames_glob": {
                "fem": "fem_*.png",
                "mpm": "mpm_*.png",
            },
            "run_manifest": "run_manifest.json",
            "metrics": {
                "csv": "metrics.csv",
                "json": "metrics.json",
            },
            "intermediate": {
                "dir": "intermediate",
                "frames_glob": "intermediate/frame_*.npz",
            },
        },
    }

    manifest_path = save_dir / "run_manifest.json"
    try:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"Warning: failed to write preflight run manifest: {e}")
        return

    try:
        _write_tuning_notes(
            save_dir,
            record_interval=int(record_interval),
            total_frames=int(total_frames),
            run_context=manifest.get("run_context") if isinstance(manifest, dict) else {},
            overwrite=False,
            reason=reason,
        )
    except Exception as e:
        print(f"Warning: failed to write tuning notes: {e}")


# ==============================================================================
# FEM RGB Renderer (Reuses VecTouchSim)
# ==============================================================================
class FEMRGBRenderer:
    """FEM sensor RGB rendering using existing VecTouchSim pipeline"""

    def __init__(
        self,
        fem_file: str,
        object_file: Optional[str] = None,
        visible: bool = False,
        indenter_face: str = "tip",
        indenter_geom: str = "stl",
    ):
        if not HAS_EZGL:
            raise RuntimeError("ezgl not available for FEM rendering")   

        self.fem_file = Path(fem_file)
        self.object_file = object_file
        self.visible = visible
        self.indenter_face = indenter_face
        self.indenter_geom = indenter_geom

        # Create depth scene for object rendering
        self.depth_scene = self._create_depth_scene()

        # Create VecTouchSim for FEM sensor rendering
        self.sensor_sim = VecTouchSim(
            depth_size=SCENE_PARAMS['depth_img_size'],
            fem_file=str(fem_file),
            visible=visible,
            title="FEM Sensor"
        )
        self.sensor_sim.set_friction_coefficient(float(SCENE_PARAMS.get("fem_fric_coef", 0.4)))

        # Object pose (indenter position)
        self._object_y = 0.02  # initial y position (m)
        self._object_z = 0.0   # will be set by trajectory

    def _create_depth_scene(self) -> 'DepthRenderScene':
        """Create depth rendering scene for the indenter"""
        return DepthRenderScene(
            object_file=self.object_file,
            visible=self.visible,
            indenter_face=self.indenter_face,
            indenter_geom=self.indenter_geom,
        )

    def set_indenter_pose(self, x_mm: float, y_mm: float, z_mm: float):
        """Set indenter position in mm (sensor coordinates)"""
        # Convert to meters for depth scene
        x_m = x_mm * 1e-3
        y_m = y_mm * 1e-3
        z_m = z_mm * 1e-3
        self.depth_scene.set_object_pose(x_m, y_m, z_m)

    def step(self) -> np.ndarray:
        """Run one step and return RGB image"""
        # Update depth scene to render object
        self.depth_scene.update()

        # Render depth map
        depth = self.depth_scene.get_depth()

        # Get poses
        sensor_pose = self.depth_scene.get_sensor_pose()
        object_pose = self.depth_scene.get_object_pose()

        # Step FEM simulation
        self.sensor_sim.step(object_pose, sensor_pose, depth)

        # Update sensor scene rendering
        self.sensor_sim.update()

        return self.get_image()

    def get_image(self) -> np.ndarray:
        """Get current RGB image (H, W, 3) uint8"""
        return self.sensor_sim.get_image()

    def get_diff_image(self) -> np.ndarray:
        """Get diff image relative to reference"""
        return self.sensor_sim.get_diff_image()

    def update(self):
        """Update visualization windows"""
        if self.visible:
            self.depth_scene.update()
            self.sensor_sim.update()


class DepthRenderScene(Scene):
    """Simple depth rendering scene for indenter object"""

    def __init__(
        self,
        object_file: Optional[str] = None,
        visible: bool = True,
        indenter_face: str = "tip",
        indenter_geom: str = "stl",
    ):
        # Note: visible=True is required for proper OpenGL context initialization
        super().__init__(600, 400, visible=visible, title="Depth Render")
        self.cameraLookAt((0.1, 0.1, 0.1), (0, 0, 0), (0, 1, 0))

        # Camera view parameters
        self.cam_view_width = SCENE_PARAMS['cam_view_width_m']
        self.cam_view_height = SCENE_PARAMS['cam_view_height_m']

        self._indenter_geom = str(indenter_geom).strip().lower()
        if self._indenter_geom not in {"stl", "box", "sphere"}:
            self._indenter_geom = "stl"

        # Create indenter object
        stl_path: Optional[Path] = None
        if self._indenter_geom == "box":
            half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
            if half_extents_mm is not None:
                hx_mm, hy_mm, hz_mm = half_extents_mm
            else:
                r_mm = float(SCENE_PARAMS["indenter_radius_mm"])
                hx_mm = hy_mm = hz_mm = r_mm
            size_m = (float(hx_mm) * 2e-3, float(hy_mm) * 2e-3, float(hz_mm) * 2e-3)
            self.object = GLBoxItem(size=size_m, glOptions="translucent")
        elif self._indenter_geom == "sphere":
            r_m = float(SCENE_PARAMS["indenter_radius_mm"]) * 1e-3
            verts, faces = ezgl_mesh_sphere(radius=r_m, rows=24, cols=24)
            self.object = GLMeshItem(
                vertexes=verts,
                indices=faces,
                lights=PointLight(),
                glOptions="translucent",
            )
        else:
            # STL path: prefer explicit object_file if provided
            if object_file and Path(object_file).exists():
                stl_path = Path(object_file)
            else:
                stl_path = ASSET_DIR / "obj/circle_r4.STL"
            self.object = GLModelItem(
                str(stl_path),
                glOptions="translucent",
                lights=PointLight(),
            )

        # Depth camera
        self.depth_cam = DepthCamera(
            self,
            eye=(0, 0, 0),
            center=(0, 1, 0),
            up=(0, 0, 1),
            img_size=SCENE_PARAMS['depth_img_size'],
            proj_type="ortho",
            ortho_space=(
                -self.cam_view_width / 2, self.cam_view_width / 2,
                -self.cam_view_height / 2, self.cam_view_height / 2,
                -0.005, 0.1
            ),
            frustum_visible=True,
            actual_depth=True
        )

        # IMPORTANT: setTransform() will overwrite any constructor-time rotate/translate.
        # 所以“固定旋转/偏移”必须显式地与每帧 pose 合成，再一次性 setTransform()，避免隐式状态。
        self._indenter_face = indenter_face
        self._stl_endfaces_mm: Optional[Dict[str, object]] = None
        self._stl_height_m = 0.0
        if self._indenter_geom == "stl" and stl_path and stl_path.suffix.lower() == ".stl" and stl_path.exists():
            stats = _analyze_binary_stl_endfaces_mm(stl_path)
            if stats is not None:
                self._stl_endfaces_mm = stats
                height_mm = float(stats.get("height_mm", 0.0))
                self._stl_height_m = max(height_mm * 1e-3, 0.0)

        self._object_fixed_tf = Matrix4x4()     # local->parent 固定变换（轴对齐/朝向/模型原点偏移等）
        self._object_pose_raw = Matrix4x4()     # 每帧输入 pose（仅平移/旋转控制量）
        self._object_pose = Matrix4x4()         # 最终用于渲染+FEM 的世界变换（raw * fixed）
        self._apply_indenter_face()

    def _apply_indenter_face(self) -> None:
        if self._indenter_geom != "stl":
            self._object_fixed_tf = Matrix4x4()
            return
        face = str(self._indenter_face).strip().lower()
        if face not in {"base", "tip"}:
            face = "base"
        self._indenter_face = face

        self._object_fixed_tf = Matrix4x4()
        if face == "tip":
            # 目标：把 STL 的 y_max 端面翻到 y_min 方向，确保“tip”端面可用于接触对齐验证。
            # 使用 y 轴翻转（绕 X 轴 180°），并用 STL 高度做一次平移以保持 tip 端面仍位于局部 y≈0 平面。
            if self._stl_height_m > 0:
                self._object_fixed_tf.translate(0, float(self._stl_height_m), 0)
            self._object_fixed_tf.rotate(180, 1, 0, 0)

    def set_object_pose(self, x: float, y: float, z: float):
        """Set object position in meters"""
        self._object_pose_raw = Matrix4x4.fromVector6d(x, y, z, 0, 0, 0)
        final_tf = Matrix4x4(self._object_pose_raw) * self._object_fixed_tf
        self._object_pose = Matrix4x4(final_tf)
        self.object.setTransform(self._object_pose)

        if SCENE_PARAMS.get("debug_verbose", False):
            try:
                raw_xyz = self._object_pose_raw.xyz.tolist()
                fixed_xyz = self._object_fixed_tf.xyz.tolist()
                final_xyz = self._object_pose.xyz.tolist()
                raw_euler = getattr(self._object_pose_raw, "euler", None)
                fixed_euler = getattr(self._object_fixed_tf, "euler", None)
                final_euler = getattr(self._object_pose, "euler", None)
                print(
                    "[DepthRenderScene] "
                    f"raw_xyz={raw_xyz}, fixed_xyz={fixed_xyz}, final_xyz={final_xyz}; "
                    f"raw_euler={raw_euler}, fixed_euler={fixed_euler}, final_euler={final_euler}"
                )
            except Exception:
                # debug 模式下也不应因为日志失败而影响渲染
                pass

    def get_object_pose(self) -> Matrix4x4:
        return self._object_pose

    def get_object_pose_raw(self) -> Matrix4x4:
        return self._object_pose_raw

    def get_sensor_pose(self) -> Matrix4x4:
        return self.depth_cam.transform(local=False)

    def get_depth(self) -> np.ndarray:
        depth = self.depth_cam.render()

        # CRITICAL: Filter out background (far plane) values
        # Background has depth = far = 0.1m
        # FEM contact detection: z_gel < p_gel[:, 2]
        # After FEM processing: depth_map = depth * 0.4 * 1000 (mm)
        # Background 0.1m -> 40mm, which may still trigger contact if gel z < 40mm
        # Solution: Set background to a very large value (e.g., 1000mm) so it never triggers contact
        far_value = 0.1
        background_mask = depth >= (far_value - 0.001)  # Pixels at or near far plane
        depth_filtered = depth.copy()
        depth_filtered[background_mask] = 10.0  # 10m = 10000mm after processing, definitely no contact

        return depth_filtered


# ==============================================================================
# MPM Height Field Renderer
# ==============================================================================
class MPMHeightFieldRenderer:
    """Renders MPM particle data as sensor RGB images via height field extraction"""

    def __init__(self, visible: bool = False):
        if not HAS_EZGL:
            raise RuntimeError("ezgl not available for MPM rendering")

        self.visible = visible
        self.gel_size_mm = SCENE_PARAMS['gel_size_mm']
        self.grid_shape = SCENE_PARAMS['height_grid_shape']

        # Create rendering scene
        self.scene = self._create_render_scene()

        # Reference height field (flat surface)
        self._ref_height = np.zeros(self.grid_shape, dtype=np.float32)

        # Cached mapping derived from initial particle positions (stable across frames)
        self._is_configured = False
        self._x_center_mm = 0.0
        self._y_min_mm = 0.0
        self._surface_indices: Optional[np.ndarray] = None
        self._initial_positions_m: Optional[np.ndarray] = None
        self._last_height_reference_z_mm = 0.0

    def _create_render_scene(self) -> 'MPMSensorScene':
        """Create rendering scene matching SensorScene style"""
        return MPMSensorScene(
            gel_size_mm=self.gel_size_mm,
            grid_shape=self.grid_shape,
            visible=self.visible
        )

    def configure_from_initial_positions(self, initial_positions_m: np.ndarray, initial_top_z_m: float) -> None:
        """
        Cache a stable mapping from MPM solver coordinates to sensor grid coordinates.

        - Avoids per-frame recentring (which cancels slide motion and causes jitter)
        - Derives a thin top-surface particle mask to produce a proper height field
        """
        pos_mm = initial_positions_m * 1000.0
        self._x_center_mm = float((pos_mm[:, 0].min() + pos_mm[:, 0].max()) / 2.0)
        self._y_min_mm = float(pos_mm[:, 1].min())

        dx_m = float(SCENE_PARAMS['mpm_grid_dx_mm']) * 1e-3
        particles_per_cell = float(SCENE_PARAMS['mpm_particles_per_cell'])
        particle_spacing_m = dx_m / max(particles_per_cell, 1.0)
        surface_band_m = 2.0 * particle_spacing_m

        z_threshold_m = float(initial_top_z_m) - surface_band_m
        surface_mask = initial_positions_m[:, 2] >= z_threshold_m
        self._surface_indices = np.nonzero(surface_mask)[0].astype(np.int32)
        self._initial_positions_m = initial_positions_m.copy()
        self._is_configured = True

    def extract_height_field(
        self,
        positions_m: np.ndarray,
        initial_top_z_m: float,
        indenter_center_m: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """
        Extract top-surface height field from MPM particles

        Args:
            positions_m: Particle positions (N, 3) in meters
            initial_top_z_m: Initial top surface z coordinate in meters  
            indenter_center_m: Optional indenter center (x,y,z) in meters, used to clamp
                height_field not below indenter surface (avoid penalty penetration artifacts).

        Returns:
            height_field_mm: (n_row, n_col) array, negative = indentation
        """
        n_row, n_col = self.grid_shape
        gel_w_mm, gel_h_mm = self.gel_size_mm

        if not self._is_configured:
            self.configure_from_initial_positions(positions_m, initial_top_z_m)

        # NOTE: 这里不能只取“初始顶面一层粒子”。在平底压头（cylinder/box）接触中，
        # 初始顶面粒子可能会被挤压下沉并被更深层粒子“顶替”成为新表面；
        # 若仅追踪初始顶面索引，会把“已下沉的旧表面”误当成当前表面，导致高度场过深，
        # 进而在渲染中出现非物理的“整块变暗/发脏”。
        pos_mm = positions_m * 1000.0
        z_top_init_mm = initial_top_z_m * 1000.0

        # Map to sensor grid using cached reference frame:
        # x ∈ [-gel_w/2, gel_w/2], y ∈ [0, gel_h]
        pos_sensor = pos_mm.copy()
        pos_sensor[:, 0] -= self._x_center_mm
        pos_sensor[:, 1] -= self._y_min_mm

        # Grid cell dimensions
        cell_w = gel_w_mm / n_col
        cell_h = gel_h_mm / n_row

        # Build height field using a small neighborhood splat to reduce holes.
        # The render grid resolution is close to particle spacing, so strict per-cell binning
        # creates pepper noise (empty cells) which becomes hard edges after shading.
        x_mm = pos_sensor[:, 0].astype(np.float32, copy=False)
        y_mm = pos_sensor[:, 1].astype(np.float32, copy=False)
        z_mm = pos_sensor[:, 2].astype(np.float32, copy=False)
        z_disp = (z_mm - np.float32(z_top_init_mm)).astype(np.float32, copy=False)  # <= 0

        col_f = (x_mm + np.float32(gel_w_mm / 2.0)) / np.float32(cell_w) - np.float32(0.5)
        row_f = y_mm / np.float32(cell_h) - np.float32(0.5)
        col0 = np.floor(col_f).astype(np.int32)
        row0 = np.floor(row_f).astype(np.int32)

        # Initialize with -inf so we can take max z_disp per cell (top surface).
        height_field = np.full((n_row, n_col), -np.inf, dtype=np.float32)
        for di in (0, 1):
            rr = row0 + di
            for dj in (0, 1):
                cc = col0 + dj
                m = (rr >= 0) & (rr < n_row) & (cc >= 0) & (cc < n_col)
                if m.any():
                    np.maximum.at(height_field, (rr[m], cc[m]), z_disp[m])

        height_field[~np.isfinite(height_field)] = np.nan
        valid_mask = np.isfinite(height_field)

        reference_z = 0.0
        if bool(SCENE_PARAMS.get("mpm_height_reference_edge", True)):    
            # CRITICAL: Use EDGE regions as fixed spatial reference for better contrast.
            # This preserves slide motion (unlike global percentile which cancels it).
            # Use original valid cells for baseline to avoid bias from hole filling.
            edge_margin = max(3, n_row // 20)  # ~5% margin or at least 3 rows/cols
            edge_mask = np.zeros_like(valid_mask, dtype=bool)
            edge_mask[:edge_margin, :] = True
            edge_mask[-edge_margin:, :] = True
            if edge_margin * 2 < n_row and edge_margin * 2 < n_col:
                edge_mask[edge_margin:-edge_margin, :edge_margin] = True
                edge_mask[edge_margin:-edge_margin, -edge_margin:] = True

            edge_values = height_field[edge_mask & valid_mask]
            edge_valid = edge_values[np.isfinite(edge_values) & (edge_values > -10)]  # filter outliers
            if edge_valid.size > 0:
                reference_z = float(np.median(edge_valid))  # Median is robust to outliers
                height_field = height_field - reference_z  # Now center region depression is negative

        # Cache the per-run reference used by extract_surface_fields() so UV selection is consistent.
        self._last_height_reference_z_mm = float(reference_z)

        # IMPORTANT: Clamp to indenter surface to suppress over-penetration artifacts.
        if (
            indenter_center_m is not None
            and bool(SCENE_PARAMS.get("mpm_height_clamp_indenter", True))
        ):
            try:
                cx_mm = float(indenter_center_m[0]) * 1000.0 - float(self._x_center_mm)
                cy_mm = float(indenter_center_m[1]) * 1000.0 - float(self._y_min_mm)
                cz_mm = float(indenter_center_m[2]) * 1000.0

                x_centers = (np.arange(n_col, dtype=np.float32) + 0.5) * np.float32(cell_w) - np.float32(gel_w_mm / 2.0)
                y_centers = (np.arange(n_row, dtype=np.float32) + 0.5) * np.float32(cell_h)
                xx, yy = np.meshgrid(x_centers, y_centers)

                clamp_field = np.full((n_row, n_col), np.nan, dtype=np.float32)
                indenter_type = str(SCENE_PARAMS.get("indenter_type", "box")).lower().strip()

                if indenter_type == "sphere":
                    r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                    rr = np.sqrt((xx - np.float32(cx_mm)) ** 2 + (yy - np.float32(cy_mm)) ** 2)
                    inside = rr <= np.float32(r_mm)
                    # Sphere lower hemisphere: z = cz - sqrt(R^2 - r^2)
                    dz = np.sqrt(np.maximum(np.float32(r_mm) ** 2 - rr ** 2, 0.0))
                    z_surface_mm = np.float32(cz_mm) - dz
                    surface_disp = (z_surface_mm - np.float32(z_top_init_mm)) - np.float32(reference_z)
                    clamp_field[inside] = surface_disp[inside].astype(np.float32, copy=False)
                    clamp_field[~(clamp_field < 0.0)] = np.nan
                elif indenter_type == "cylinder":
                    r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                    half_h_mm = SCENE_PARAMS.get("indenter_cylinder_half_height_mm", None)
                    half_h_mm = float(r_mm if half_h_mm is None else float(half_h_mm))
                    inside = ((xx - np.float32(cx_mm)) ** 2 + (yy - np.float32(cy_mm)) ** 2) <= np.float32(r_mm) ** 2
                    surface_disp = (np.float32(cz_mm) - np.float32(half_h_mm) - np.float32(z_top_init_mm)) - np.float32(reference_z)
                    if float(surface_disp) < 0.0:
                        clamp_field[inside] = np.float32(surface_disp)
                else:
                    half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
                    if half_extents_mm is not None and len(half_extents_mm) == 3:
                        hx_mm, hy_mm, hz_mm = [float(v) for v in half_extents_mm]
                    else:
                        r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                        hx_mm = hy_mm = hz_mm = float(r_mm)
                    inside = (np.abs(xx - np.float32(cx_mm)) <= np.float32(hx_mm)) & (np.abs(yy - np.float32(cy_mm)) <= np.float32(hy_mm))
                    surface_disp = (np.float32(cz_mm) - np.float32(hz_mm) - np.float32(z_top_init_mm)) - np.float32(reference_z)
                    if float(surface_disp) < 0.0:
                        clamp_field[inside] = np.float32(surface_disp)

                # Use fmax to keep NaNs outside indenter footprint.
                height_field = np.fmax(height_field, clamp_field)
                valid_mask = np.isfinite(height_field)
            except Exception as e:
                if SCENE_PARAMS.get("debug_verbose", False):
                    print(f"[MPM HEIGHT] clamp_to_indenter failed: {e}")

        # 可选：footprint 外离群深值裁剪（避免“深坑”把整块区域渲染成暗盘/彩虹 halo）
        if bool(SCENE_PARAMS.get("mpm_height_clip_outliers", False)):
            clip_min = float(SCENE_PARAMS.get("mpm_height_clip_outliers_min_mm", 0.0))
            if clip_min > 0.0:
                floor_mm = -abs(clip_min)
                footprint_mask = None
                if indenter_center_m is not None:
                    try:
                        cx_mm = float(indenter_center_m[0]) * 1000.0 - float(self._x_center_mm)
                        cy_mm = float(indenter_center_m[1]) * 1000.0 - float(self._y_min_mm)

                        x_centers = (np.arange(n_col, dtype=np.float32) + 0.5) * np.float32(cell_w) - np.float32(gel_w_mm / 2.0)
                        y_centers = (np.arange(n_row, dtype=np.float32) + 0.5) * np.float32(cell_h)
                        xx, yy = np.meshgrid(x_centers, y_centers)

                        indenter_type = str(SCENE_PARAMS.get("indenter_type", "box")).lower().strip()
                        if indenter_type in {"sphere", "cylinder"}:
                            r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                            footprint_mask = ((xx - np.float32(cx_mm)) ** 2 + (yy - np.float32(cy_mm)) ** 2) <= np.float32(r_mm) ** 2
                        else:
                            half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
                            if half_extents_mm is not None and len(half_extents_mm) == 3:
                                hx_mm, hy_mm, _ = [float(v) for v in half_extents_mm]
                            else:
                                r_mm = float(SCENE_PARAMS.get("indenter_radius_mm", 4.0))
                                hx_mm = hy_mm = float(r_mm)
                            footprint_mask = (np.abs(xx - np.float32(cx_mm)) <= np.float32(hx_mm)) & (np.abs(yy - np.float32(cy_mm)) <= np.float32(hy_mm))
                    except Exception as e:
                        footprint_mask = None
                        if SCENE_PARAMS.get("debug_verbose", False):
                            print(f"[MPM HEIGHT] footprint_mask failed: {e}")

                try:
                    if footprint_mask is not None and isinstance(footprint_mask, np.ndarray):
                        outliers = (~footprint_mask) & np.isfinite(height_field) & (height_field < floor_mm)
                    else:
                        outliers = np.isfinite(height_field) & (height_field < floor_mm)
                    if outliers.any():
                        height_field[outliers] = np.nan
                        valid_mask = np.isfinite(height_field)
                except Exception as e:
                    if SCENE_PARAMS.get("debug_verbose", False):
                        print(f"[MPM HEIGHT] clip_outliers failed: {e}")

        # Fill holes after baseline alignment to avoid converting holes into small positive bumps
        # (which get clamped to 0 and create hard edges / rainbow halos after shading).
        if bool(SCENE_PARAMS.get("mpm_height_fill_holes", False)):
            iters = int(SCENE_PARAMS.get("mpm_height_fill_holes_iters", 10))
            if iters > 0:
                try:
                    height_field = _fill_height_holes(height_field, valid_mask, max_iterations=iters)
                except Exception:
                    height_field = np.nan_to_num(height_field, nan=0.0)
            else:
                height_field = np.nan_to_num(height_field, nan=0.0)
        else:
            height_field = np.nan_to_num(height_field, nan=0.0)

        # Debug: check height field statistics
        valid_mask = height_field < -0.05  # Only count significant depressions
        if valid_mask.any():
            # Find center of mass of the depression
            rows, cols = np.where(valid_mask)
            if len(rows) > 0:
                center_row = np.average(rows, weights=-height_field[rows, cols])
                center_col = np.average(cols, weights=-height_field[rows, cols])
                center_x_mm = (center_col - n_col/2) * cell_w
                center_y_mm = center_row * cell_h

                # Debug: show actual surface particle x positions
                x_positions = pos_sensor[:, 0]
                if SCENE_PARAMS.get('debug_verbose', False):
                    print(f"[MPM HEIGHT] min={height_field.min():.2f}mm, cells={valid_mask.sum()}, "
                          f"center=({center_x_mm:.1f}, {center_y_mm:.1f})mm | "
                          f"particles: x_range=[{x_positions.min():.1f}, {x_positions.max():.1f}]mm")
        else:
            if SCENE_PARAMS.get('debug_verbose', False):
                print(f"[MPM HEIGHT] No significant deformation (threshold=0.05mm), particles={len(pos_sensor)}")

        return height_field

    def extract_surface_fields(
        self,
        positions_m: np.ndarray,
        initial_top_z_m: float,
        indenter_center_m: Optional[Tuple[float, float, float]] = None,
        smooth_uv: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时提取顶面高度场与面内位移场（u,v）。

        Returns:
            height_field_mm: (Ny,Nx), <= 0 表示压入
            uv_disp_mm: (Ny,Nx,2), 单位 mm
        """
        height_field = self.extract_height_field(
            positions_m,
            initial_top_z_m,
            indenter_center_m=indenter_center_m,
        )

        if not self._is_configured:
            self.configure_from_initial_positions(positions_m, initial_top_z_m)
        if self._initial_positions_m is None:
            uv = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=np.float32)
            return height_field, uv

        n_row, n_col = self.grid_shape
        gel_w_mm, gel_h_mm = self.gel_size_mm

        pos_mm_all = positions_m * 1000.0
        init_mm_all = self._initial_positions_m * 1000.0

        # NOTE: 同 extract_height_field()，这里也必须使用“当前顶面”而不是“初始顶面一层”。
        # 否则在平底压头接触/滑移时，UV 会接近 0，marker 看起来像贴在屏幕上不动。

        pos_sensor = pos_mm_all.copy()
        init_sensor = init_mm_all.copy()
        pos_sensor[:, 0] -= self._x_center_mm
        pos_sensor[:, 1] -= self._y_min_mm
        init_sensor[:, 0] -= self._x_center_mm
        init_sensor[:, 1] -= self._y_min_mm

        cell_w = gel_w_mm / n_col
        cell_h = gel_h_mm / n_row

        uv_sum = np.zeros((n_row, n_col, 2), dtype=np.float32)
        uv_cnt = np.zeros((n_row, n_col), dtype=np.int32)

        # Vectorized per-cell surface displacement:
        # - reuse the same 4-neighbor splat as height extraction
        # - only keep particles near each cell's top surface (within `surface_band_mm`)
        x_mm = pos_sensor[:, 0].astype(np.float32, copy=False)
        y_mm = pos_sensor[:, 1].astype(np.float32, copy=False)
        z_mm = pos_sensor[:, 2].astype(np.float32, copy=False)
        z_top_init_mm = np.float32(initial_top_z_m * 1000.0)
        z_disp = (z_mm - z_top_init_mm).astype(np.float32, copy=False)
        z_disp = z_disp - np.float32(getattr(self, "_last_height_reference_z_mm", 0.0))

        disp_x = (pos_sensor[:, 0] - init_sensor[:, 0]).astype(np.float32, copy=False)
        disp_y = (pos_sensor[:, 1] - init_sensor[:, 1]).astype(np.float32, copy=False)

        col_f = (x_mm + np.float32(gel_w_mm / 2.0)) / np.float32(cell_w) - np.float32(0.5)
        row_f = y_mm / np.float32(cell_h) - np.float32(0.5)
        col0 = np.floor(col_f).astype(np.int32)
        row0 = np.floor(row_f).astype(np.int32)

        dx_m = float(SCENE_PARAMS['mpm_grid_dx_mm']) * 1e-3
        particles_per_cell = float(SCENE_PARAMS['mpm_particles_per_cell'])
        particle_spacing_m = dx_m / max(particles_per_cell, 1.0)
        surface_band_mm = np.float32(2.0 * particle_spacing_m * 1000.0)

        for di in (0, 1):
            rr = row0 + di
            for dj in (0, 1):
                cc = col0 + dj
                m = (rr >= 0) & (rr < n_row) & (cc >= 0) & (cc < n_col)
                if not m.any():
                    continue

                rr_m = rr[m]
                cc_m = cc[m]
                ref = height_field[rr_m, cc_m].astype(np.float32, copy=False)
                z_m = z_disp[m]
                top = np.isfinite(ref) & (z_m >= (ref - surface_band_mm))
                if not top.any():
                    continue

                rr_t = rr_m[top]
                cc_t = cc_m[top]
                np.add.at(uv_sum[..., 0], (rr_t, cc_t), disp_x[m][top])
                np.add.at(uv_sum[..., 1], (rr_t, cc_t), disp_y[m][top])
                np.add.at(uv_cnt, (rr_t, cc_t), 1)

        uv = np.zeros((n_row, n_col, 2), dtype=np.float32)
        nonzero = uv_cnt > 0
        uv[nonzero] = uv_sum[nonzero] / uv_cnt[nonzero][..., None]

        # Fill holes in UV field using diffusion from neighboring cells
        # This prevents discontinuities in areas without particle coverage
        hole_count = (~nonzero).sum()
        if hole_count > 0:
            uv = _fill_uv_holes(uv, nonzero, max_iterations=10)

        if smooth_uv:
            uv = _box_blur_2d_xy(uv, iterations=1)

        return height_field, uv

    def render(self, height_field_mm: np.ndarray) -> np.ndarray:
        """
        Render height field to RGB image

        Args:
            height_field_mm: (n_row, n_col) height displacement in mm

        Returns:
            rgb_image: (H, W, 3) uint8
        """
        # Debug: verify height field values before rendering
        neg_mask = height_field_mm < 0
        if SCENE_PARAMS.get('debug_verbose', False):
            if neg_mask.any():
                print(f"[MPM RENDER] height_field: min={height_field_mm.min():.2f}mm, "
                      f"shape={height_field_mm.shape}, negative_cells={neg_mask.sum()}")
            else:
                print(f"[MPM RENDER] WARNING: No negative values in height_field! "
                      f"range=[{height_field_mm.min():.4f}, {height_field_mm.max():.4f}]")
        smooth = bool(SCENE_PARAMS.get("mpm_height_smooth", True))
        self.scene.set_height_field(height_field_mm, smooth=smooth)
        return self.scene.get_image()

    def get_diff_image(self, height_field_mm: np.ndarray) -> np.ndarray:
        """Get diff image relative to flat reference"""
        smooth = bool(SCENE_PARAMS.get("mpm_height_smooth", True))
        self.scene.set_height_field(height_field_mm, smooth=smooth)
        return self.scene.get_diff_image()

    def update(self):
        if self.visible:
            self.scene.update()


class MPMSensorScene(Scene):
    """Minimal rendering scene matching SensorScene visual style"""

    def __init__(
        self,
        gel_size_mm: Tuple[float, float],
        grid_shape: Tuple[int, int],
        visible: bool = False
    ):
        super().__init__(win_height=630, win_width=375, visible=visible, title="MPM Sensor")

        self.gel_width_mm, self.gel_height_mm = gel_size_mm
        self.n_row, self.n_col = grid_shape

        self.align_view()

        # Scale factor to match SensorScene
        scale_ratio = 4 / self.gel_width_mm
        base_tf = (Matrix4x4.fromScale(scale_ratio, scale_ratio, scale_ratio)
                   .translate(0, -self.gel_height_mm / 2, 0, True)
                   .rotate(180, 0, 1, 0, True))

        # Lights matching SensorScene
        self.light_white = PointLight(
            pos=(0, 0, 1), ambient=(0.1, 0.1, 0.1), diffuse=(0.1, 0.1, 0.1),
            specular=(0, 0, 0), visible=True, directional=False, render_shadow=False
        )
        self.light_r = LineLight(
            pos=np.array([2, -3.0, 1.5]), pos2=np.array([2, 3.0, 1.3]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )
        self.light_g = LineLight(
            pos=np.array([-2, -3.2, 1.5]), pos2=np.array([-2, 3.2, 1.3]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )
        self.light_b = LineLight(
            pos=np.array([-2, -3.2, 1]), pos2=np.array([2, -3.2, 1]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )
        self.light_b2 = LineLight(
            pos=np.array([-1.7, 3.2, 1]), pos2=np.array([1.7, 3.2, 1]),
            render_shadow=True, visible=True, light_frustum_visible=False
        )
        lights = [self.light_white, self.light_r, self.light_g, self.light_b, self.light_b2]

        # Load light configuration if available
        light_file = ASSET_DIR / "data/light.txt"
        if light_file.exists():
            self.loadLight(str(light_file))

        # Create marker texture (static grid pattern matching SensorScene style)
        self.marker_tex_np = self._make_marker_texture(tex_size=(320, 560), marker_radius=3)
        self.white_tex_np = np.full((560, 320, 3), 255, dtype=np.uint8)
        self.marker_tex = Texture2D(self.marker_tex_np)
        self._show_marker = True
        self._marker_mode = "static"  # off|static|warp
        self._uv_disp_mm: Optional[np.ndarray] = None
        self._cached_warped_tex: Optional[np.ndarray] = None  # Cache to avoid double remap per frame
        self._depth_tint_enabled = True
        # NOTE: SensorScene 的 texcoords u_range=(1,0), v_range=(1,0)；MPM 侧跟随此约定。
        self._warp_flip_x = True
        self._warp_flip_y = True

        # Surface mesh
        self.surf_mesh = GLSurfMeshItem(
            (self.n_row, self.n_col),
            x_range=(self.gel_width_mm / 2, -self.gel_width_mm / 2),
            y_range=(self.gel_height_mm, 0),
            lights=lights,
            material=Material(
                ambient=(1, 1, 1), diffuse=(1, 1, 1), specular=(1, 1, 1),
                textures=[self.marker_tex]
            ),
        )
        self.surf_mesh.applyTransform(base_tf)

        # Set texture coordinates (CRITICAL for marker display!)
        texcoords = self._gen_texcoords(self.n_row, self.n_col, v_range=(1, 0))
        self.surf_mesh.mesh_item.setData(texcoords=texcoords)

        # RGB camera
        w = (self.gel_width_mm - 1) / 2 * scale_ratio
        h = (self.gel_height_mm - 1) / 2 * scale_ratio
        self.rgb_camera = RGBCamera(
            self, img_size=(400, 700), eye=(0, 0, 10 * scale_ratio), up=(0, 1, 0),
            ortho_space=(-w, w, -h, h, 0, 10),
            frustum_visible=False
        )
        self.rgb_camera.render_group.update(self.surf_mesh)

        # Reference image for diff mode
        self._ref_image = None
        self._indenter_overlay_enabled = False
        self._indenter_center_mm = (0.0, 0.0)
        self._indenter_square_size_mm: Optional[float] = None
        self._debug_overlay = "off"  # off|uv|warp

    def align_view(self):
        self.cameraLookAt([0, 0, 8.15], [0, 0, 0], [0, 1, 0])

    @staticmethod
    def _gen_texcoords(n_row: int, n_col: int, u_range: Tuple[float, float] = (0, 1),
                       v_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Generate texture coordinates for mesh grid"""
        tex_u = np.linspace(*u_range, n_col)
        tex_v = np.linspace(*v_range, n_row)
        return np.stack(np.meshgrid(tex_u, tex_v), axis=-1).reshape(-1, 2)

    def loadLight(self, file_path: str):
        """Load light configuration from file"""
        try:
            with open(file_path, "r") as f:
                self.light_white.loadDict(ast.literal_eval(f.readline()))
                self.light_r.loadDict(ast.literal_eval(f.readline()))
                self.light_g.loadDict(ast.literal_eval(f.readline()))
                self.light_b.loadDict(ast.literal_eval(f.readline()))
                self.light_b2.loadDict(ast.literal_eval(f.readline()))
        except Exception:
            pass

    def _make_marker_texture(self, tex_size: Tuple[int, int], marker_radius: int = 3) -> np.ndarray:
        """
        Create static marker texture with uniform grid pattern

        Args:
            tex_size: (width, height) of texture
            marker_radius: radius of marker dots

        Returns:
            Texture array (H, W, 3) uint8
        """
        tex_w, tex_h = tex_size
        tex = np.full((tex_h, tex_w, 3), 255, dtype=np.uint8)

        # Create uniform marker grid matching typical sensor pattern
        # Approximately 14 columns x 20 rows of markers
        n_cols, n_rows = 14, 20
        margin_x, margin_y = 20, 20

        for row in range(n_rows):
            for col in range(n_cols):
                x = int(margin_x + col * (tex_w - 2 * margin_x) / (n_cols - 1))
                y = int(margin_y + row * (tex_h - 2 * margin_y) / (n_rows - 1))
                if HAS_CV2:
                    cv2.ellipse(tex, (x, y), (marker_radius, marker_radius), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
                else:
                    # Fallback: draw filled circle with numpy
                    yy, xx = np.ogrid[:tex_h, :tex_w]
                    mask = (xx - x)**2 + (yy - y)**2 <= marker_radius**2
                    tex[mask] = 0

        return tex

    def set_show_marker(self, show: bool):
        """Toggle marker visibility"""
        self._show_marker = show
        self._update_marker_texture()

    def set_marker_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in ("off", "static", "warp"):
            raise ValueError("marker mode must be one of: off|static|warp")
        self._marker_mode = mode
        self._update_marker_texture()

    def set_depth_tint_enabled(self, enabled: bool) -> None:
        """Toggle depth tint overlay on the marker/white texture."""
        self._depth_tint_enabled = bool(enabled)
        self._update_marker_texture()

    def set_uv_displacement(self, uv_disp_mm: Optional[np.ndarray]) -> None:
        """设置当前帧的面内位移场 (Ny,Nx,2)，单位 mm。"""
        if uv_disp_mm is None:
            self._uv_disp_mm = None
        else:
            # CRITICAL: Apply same horizontal flip as height_field to match mesh x_range convention
            # height_field is flipped with [:, ::-1], so UV must be too
            uv_flipped = _mpm_flip_x_field(uv_disp_mm).copy()
            # NOTE: u 分量的“方向反转”由 warp 的 flip_x 统一处理，避免同一轴被多处重复修正。
            self._uv_disp_mm = uv_flipped.astype(np.float32, copy=False)
        # 在 warp 模式下，每帧都需要更新纹理
        if self._marker_mode == "warp":
            self._update_marker_texture()

    def set_indenter_overlay(self, enabled: bool, square_size_mm: Optional[float] = None) -> None:
        self._indenter_overlay_enabled = bool(enabled)
        self._indenter_square_size_mm = square_size_mm

    def set_indenter_center(self, x_mm: float, y_mm: float) -> None:
        self._indenter_center_mm = (float(x_mm), float(y_mm))

    def set_debug_overlay(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in ("off", "uv", "warp"):
            raise ValueError("debug overlay must be one of: off|uv|warp")
        self._debug_overlay = mode

    def _update_marker_texture(self) -> None:
        if not self._show_marker or self._marker_mode == "off":
            self.marker_tex.setTexture(self.white_tex_np)
            self._cached_warped_tex = None
            return
        if self._marker_mode == "static" or self._uv_disp_mm is None:
            self.marker_tex.setTexture(self.marker_tex_np)
            self._cached_warped_tex = None
            return
        warped = warp_marker_texture(
            self.marker_tex_np,
            self._uv_disp_mm,
            gel_size_mm=(self.gel_width_mm, self.gel_height_mm),
            flip_x=self._warp_flip_x,
            flip_y=self._warp_flip_y,
        )
        self._cached_warped_tex = warped  # Cache for reuse in _update_depth_tint_texture
        self.marker_tex.setTexture(warped)

    @staticmethod
    def _box_blur_2d(values: np.ndarray, iterations: int = 1) -> np.ndarray:
        """轻量平滑：纯 numpy 3x3 box blur，避免引入 SciPy 依赖。"""
        result = values.astype(np.float32, copy=True)
        for _ in range(max(iterations, 0)):
            padded = np.pad(result, ((1, 1), (1, 1)), mode="edge")
            result = (
                padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
                padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, 0:-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
        return result

    def _update_depth_tint_texture(self, depth_mm: np.ndarray) -> None:
        """
        MPM 表层按压深度着色：压得越深越红，增强反馈可见性。

        做法：在 marker/white 底图上叠加红色热度（不改变 marker warp）。
        """
        if not self._show_marker or self._marker_mode == "off":
            base = self.white_tex_np
        elif self._marker_mode == "static" or self._uv_disp_mm is None:
            base = self.marker_tex_np
        else:
            # Use cached warped texture to avoid double remap per frame
            if self._cached_warped_tex is not None:
                base = self._cached_warped_tex
            else:
                # Fallback: compute if cache is missing (shouldn't happen in normal flow)
                base = warp_marker_texture(
                    self.marker_tex_np,
                    self._uv_disp_mm,
                    gel_size_mm=(self.gel_width_mm, self.gel_height_mm),
                    flip_x=self._warp_flip_x,
                    flip_y=self._warp_flip_y,
                )

        depth_pos = np.clip(-depth_mm, 0.0, None)  # mm, >=0
        if depth_pos.max() <= 1e-6:
            self.marker_tex.setTexture(base)
            return

        depth_norm = depth_pos / (depth_pos.max() + 1e-6)
        tex_h, tex_w = base.shape[0], base.shape[1]
        src_h, src_w = depth_norm.shape
        row_idx = (np.linspace(0, src_h - 1, tex_h)).astype(np.int32)
        col_idx = (np.linspace(0, src_w - 1, tex_w)).astype(np.int32)
        upsampled = depth_norm[row_idx][:, col_idx]

        tinted = base.astype(np.float32)
        tinted[..., 0] = np.clip(tinted[..., 0] + 150.0 * upsampled, 0, 255)
        tinted[..., 1] = np.clip(tinted[..., 1] * (1.0 - 0.45 * upsampled), 0, 255)
        tinted[..., 2] = np.clip(tinted[..., 2] * (1.0 - 0.45 * upsampled), 0, 255)

        self.marker_tex.setTexture(tinted.astype(np.uint8))

    def set_height_field(self, height_field_mm: np.ndarray, smooth: bool = True):
        """Update surface mesh with height field data"""
        if smooth:
            iters = int(SCENE_PARAMS.get("mpm_height_smooth_iters", 2))
            if iters > 0:
                height_field_mm = self._box_blur_2d(height_field_mm, iterations=iters)

        # CRITICAL: Flip horizontally to match mesh x_range convention
        # Height field: col=0 is x=-gel_w/2 (left)
        # Mesh x_range: (gel_w/2, -gel_w/2) means col=0 is x=+gel_w/2 (right)
        height_field_mm = _mpm_flip_x_field(height_field_mm)

        # Ensure negative values for indentation (SensorScene convention)
        depth = np.minimum(height_field_mm, 0)

        # Debug: verify depth values being sent to mesh
        neg_count = (depth < -0.01).sum()
        if neg_count > 0 and SCENE_PARAMS.get('debug_verbose', False):
            print(f"[MESH UPDATE] depth: min={depth.min():.2f}mm, neg_cells(>0.01mm)={neg_count}")

        self.surf_mesh.setData(depth, smooth)
        self._update_marker_texture()
        if self._depth_tint_enabled:
            self._update_depth_tint_texture(depth)

    def get_image(self) -> np.ndarray:
        """Render and return RGB image"""
        image = (self.rgb_camera.render() * 255).astype(np.uint8)

        if self._debug_overlay in ("uv", "warp") and self._uv_disp_mm is not None:
            if self._debug_overlay == "uv":
                field = np.sqrt(np.sum(self._uv_disp_mm**2, axis=-1))  # mm
            else:
                # Approximate warp magnitude in pixels at texture resolution
                tex_h, tex_w = self.marker_tex_np.shape[0], self.marker_tex_np.shape[1]
                gel_w_mm = max(self.gel_width_mm, 1e-6)
                gel_h_mm = max(self.gel_height_mm, 1e-6)
                dx_px = (self._uv_disp_mm[..., 0] / gel_w_mm) * tex_w
                dy_px = (self._uv_disp_mm[..., 1] / gel_h_mm) * tex_h
                field = np.sqrt(dx_px**2 + dy_px**2)

            if field.max() > 1e-6:
                norm = field / (field.max() + 1e-6)
                h, w = image.shape[0], image.shape[1]
                src_h, src_w = norm.shape
                row_idx = (np.linspace(0, src_h - 1, h)).astype(np.int32)
                col_idx = (np.linspace(0, src_w - 1, w)).astype(np.int32)
                ov = norm[row_idx][:, col_idx]
                image = image.copy()
                image[..., 0] = np.clip(image[..., 0] + 120.0 * ov, 0, 255)
                image[..., 1] = np.clip(image[..., 1] * (1.0 - 0.35 * ov), 0, 255)
                image[..., 2] = np.clip(image[..., 2] * (1.0 - 0.35 * ov), 0, 255)

        if self._indenter_overlay_enabled:
            x_mm, y_mm = self._indenter_center_mm
            cell_w = self.gel_width_mm / self.n_col
            cell_h = self.gel_height_mm / self.n_row
            # Keep overlay consistent with MPM render flip convention (see _mpm_flip_x_field).
            x_mm = _mpm_flip_x_mm(x_mm)
            col = int((x_mm + self.gel_width_mm / 2.0) / cell_w)
            row = int(y_mm / cell_h)
            col = int(np.clip(col, 0, self.n_col - 1))
            row = int(np.clip(row, 0, self.n_row - 1))

            px = int(col / max(self.n_col - 1, 1) * (image.shape[1] - 1))
            py = int(row / max(self.n_row - 1, 1) * (image.shape[0] - 1))

            size_mm = self._indenter_square_size_mm if self._indenter_square_size_mm is not None else 6.0
            half_cols = int((size_mm / 2.0) / cell_w)
            half_rows = int((size_mm / 2.0) / cell_h)
            px0 = int(np.clip((col - half_cols) / max(self.n_col - 1, 1) * (image.shape[1] - 1), 0, image.shape[1] - 1))
            px1 = int(np.clip((col + half_cols) / max(self.n_col - 1, 1) * (image.shape[1] - 1), 0, image.shape[1] - 1))
            py0 = int(np.clip((row - half_rows) / max(self.n_row - 1, 1) * (image.shape[0] - 1), 0, image.shape[0] - 1))
            py1 = int(np.clip((row + half_rows) / max(self.n_row - 1, 1) * (image.shape[0] - 1), 0, image.shape[0] - 1))

            if HAS_CV2:
                cv2.rectangle(image, (px0, py0), (px1, py1), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(image, (px, py), 3, (255, 255, 0), -1, cv2.LINE_AA)
            else:
                image[py0:py0+2, px0:px1] = (255, 255, 0)
                image[py1-2:py1, px0:px1] = (255, 255, 0)
                image[py0:py1, px0:px0+2] = (255, 255, 0)
                image[py0:py1, px1-2:px1] = (255, 255, 0)
                image[max(py-1,0):py+2, max(px-1,0):px+2] = (255, 255, 0)

        return image

    def get_diff_image(self) -> np.ndarray:
        """Get difference image relative to reference"""
        if self._ref_image is None:
            # Capture reference on first call
            self.set_height_field(np.zeros((self.n_row, self.n_col)))
            self._ref_image = self.get_image()

        cur_image = self.get_image()
        return np.clip((cur_image.astype(np.int16) - self._ref_image.astype(np.int16)) + 110, 0, 255).astype(np.uint8)


# ==============================================================================
# MPM Simulation Adapter
# ==============================================================================
class MPMSimulationAdapter:
    """Adapter for running MPM simulation with press+slide trajectory"""

    def __init__(self):
        if not HAS_TAICHI:
            raise RuntimeError("Taichi not available for MPM simulation")

        self.solver = None
        self.positions_history: List[np.ndarray] = []
        self.frame_controls: List[Tuple[float, float]] = []  # [(press_amount_m, slide_amount_m)]
        self.frame_indenter_centers_m: List[Tuple[float, float, float]] = []  # [(x,y,z)] at recorded frames
        self.initial_top_z_m = 0.0
        self.initial_positions_m: Optional[np.ndarray] = None
        self._base_indices_np: Optional[np.ndarray] = None
        self._base_fixer = None

    def setup(self):
        """Initialize MPM solver"""
        ti.init(arch=ti.cpu)

        from xengym.mpm import (
            MPMConfig, GridConfig, TimeConfig, OgdenConfig,
            MaterialConfig, ContactConfig, OutputConfig, MPMSolver, SDFConfig
        )

        # Gel dimensions in meters
        gel_w_m = SCENE_PARAMS['gel_size_mm'][0] * 1e-3
        gel_h_m = SCENE_PARAMS['gel_size_mm'][1] * 1e-3
        gel_t_m = SCENE_PARAMS['gel_thickness_mm'] * 1e-3

        dx = SCENE_PARAMS['mpm_grid_dx_mm'] * 1e-3

        # Grid size (with padding)
        pad_xy = int(SCENE_PARAMS.get("mpm_grid_padding_cells_xy", 6))
        pad_z_bottom = int(SCENE_PARAMS.get("mpm_grid_padding_cells_z_bottom", 6))
        pad_z_top = int(SCENE_PARAMS.get("mpm_grid_padding_cells_z_top", 20))
        grid_extent = [
            int(np.ceil(gel_w_m / dx)) + pad_xy * 2,
            int(np.ceil(gel_h_m / dx)) + pad_xy * 2,
            int(np.ceil(gel_t_m / dx)) + pad_z_bottom + pad_z_top,
        ]

        # Create particles
        n_particles = self._create_particles(gel_w_m, gel_h_m, gel_t_m, dx)

        # Indenter setup
        indenter_r = SCENE_PARAMS['indenter_radius_mm'] * 1e-3
        indenter_gap = SCENE_PARAMS['indenter_start_gap_mm'] * 1e-3
        indenter_type = SCENE_PARAMS.get('indenter_type', 'box')

        # Determine the effective z half-height for indenter placement
        # - sphere: use radius
        # - cylinder: use half_height (defaults to radius)
        # - box: use half_extents[2]
        if indenter_type == 'sphere':
            indenter_z_half = indenter_r
            half_extents = (indenter_r, 0, 0)  # sphere uses radius in first component
        elif indenter_type == 'cylinder':
            half_h_mm = SCENE_PARAMS.get("indenter_cylinder_half_height_mm", None)
            if half_h_mm is None:
                half_h = float(indenter_r)
            else:
                half_h = float(half_h_mm) * 1e-3
            indenter_z_half = float(half_h)
            half_extents = (indenter_r, indenter_r, float(half_h))  # (radius, radius, half_height)
        else:
            # Box (flat bottom): use half_extents for z calculation
            half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
            if half_extents_mm is not None:
                hx, hy, hz = half_extents_mm
                half_extents = (float(hx) * 1e-3, float(hy) * 1e-3, float(hz) * 1e-3)
            else:
                half_extents = (indenter_r, indenter_r, indenter_r)
            indenter_z_half = half_extents[2]  # Use actual z half-height for box

        # Calculate indenter center with correct z based on indenter type
        indenter_center = (
            float(self._particle_center[0]),
            float(self._particle_center[1]),
            float(self.initial_top_z_m + indenter_z_half + indenter_gap),
        )

        obstacles = [
            SDFConfig(sdf_type="plane", center=(0, 0, 0), normal=(0, 0, 1)),
        ]

        # Add indenter based on type
        if indenter_type == 'sphere':
            obstacles.append(SDFConfig(
                sdf_type="sphere",
                center=indenter_center,
                half_extents=half_extents
            ))
        elif indenter_type == 'cylinder':
            obstacles.append(SDFConfig(
                sdf_type="cylinder",
                center=indenter_center,
                half_extents=half_extents
            ))
        else:
            obstacles.append(SDFConfig(
                sdf_type="box",
                center=indenter_center,
                half_extents=half_extents
            ))

        print(f"[MPM] Indenter: type={indenter_type}, z_half={indenter_z_half*1000:.1f}mm, "
              f"center=({indenter_center[0]*1000:.1f}, {indenter_center[1]*1000:.1f}, {indenter_center[2]*1000:.1f})mm")

        mpm_mu_s = float(SCENE_PARAMS.get("mpm_mu_s", 2.0))
        mpm_mu_k = float(SCENE_PARAMS.get("mpm_mu_k", 1.5))
        k_n = float(SCENE_PARAMS.get("mpm_contact_stiffness_normal", 8e2))
        k_t = float(SCENE_PARAMS.get("mpm_contact_stiffness_tangent", 4e2))
        print(f"[MPM] Contact: k_n={k_n:.3g}, k_t={k_t:.3g}, mu_s={mpm_mu_s:.3g}, mu_k={mpm_mu_k:.3g}")

        enable_bulk_visc = bool(SCENE_PARAMS.get("mpm_enable_bulk_viscosity", False))
        eta_bulk = float(SCENE_PARAMS.get("mpm_bulk_viscosity", 0.0))
        print(
            f"[MPM] Material: ogden_mu={SCENE_PARAMS.get('ogden_mu')}, "
            f"ogden_alpha={SCENE_PARAMS.get('ogden_alpha')}, "
            f"ogden_kappa={SCENE_PARAMS.get('ogden_kappa')}, "
            f"bulk_viscosity={'on' if enable_bulk_visc else 'off'} (eta={eta_bulk:g})"
        )

        config = MPMConfig(
            grid=GridConfig(grid_size=grid_extent, dx=dx),
            time=TimeConfig(dt=SCENE_PARAMS['mpm_dt'], num_steps=1),      
            material=MaterialConfig(
                density=SCENE_PARAMS['density'],
                ogden=OgdenConfig(
                    mu=SCENE_PARAMS['ogden_mu'],
                    alpha=SCENE_PARAMS['ogden_alpha'],
                    kappa=SCENE_PARAMS['ogden_kappa']
                ),
                maxwell_branches=[],
                enable_bulk_viscosity=enable_bulk_visc,
                bulk_viscosity=eta_bulk,
            ),
            contact=ContactConfig(
                enable_contact=True,
                contact_stiffness_normal=k_n,
                contact_stiffness_tangent=k_t,
                mu_s=mpm_mu_s,  # static friction
                mu_k=mpm_mu_k,  # kinetic friction
                obstacles=obstacles,
            ),
            output=OutputConfig()
        )

        self.solver = MPMSolver(config, n_particles)
        self._indenter_center0 = np.array(indenter_center, dtype=np.float32)

        # Disable gravity to better match the quasi-static FEM use case in this demo.
        try:
            self.solver.gravity = ti.Vector([0.0, 0.0, 0.0])
        except Exception:
            pass

        self._setup_base_fixer()

        print(f"MPM solver initialized: {n_particles} particles")

    def _setup_base_fixer(self) -> None:
        """Fix a thin bottom layer of particles to emulate gel bonded to a rigid sensor base."""
        if self._base_indices_np is None or len(self._base_indices_np) == 0:
            self._base_fixer = None
            return
        if self.solver is None:
            self._base_fixer = None
            return

        base_indices = ti.field(dtype=ti.i32, shape=int(self._base_indices_np.shape[0]))
        base_indices.from_numpy(self._base_indices_np.astype(np.int32))

        base_init_pos = ti.Vector.field(3, dtype=ti.f32, shape=int(self._base_indices_np.shape[0]))
        base_init_pos.from_numpy(self._initial_positions[self._base_indices_np].astype(np.float32))

        @ti.kernel
        def apply_fix():
            for k in range(base_indices.shape[0]):
                p = base_indices[k]
                self.solver.fields.x[p] = base_init_pos[k]
                self.solver.fields.v[p] = ti.Vector([0.0, 0.0, 0.0])
                self.solver.fields.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                self.solver.fields.F[p] = ti.Matrix.identity(ti.f32, 3)

        self._base_fixer = apply_fix

    def _create_particles(self, gel_w: float, gel_h: float, gel_t: float, dx: float) -> int:
        """Create particle positions filling gel volume"""
        spacing = dx / SCENE_PARAMS['mpm_particles_per_cell']

        nx = int(np.ceil(gel_w / spacing))
        ny = int(np.ceil(gel_h / spacing))
        nz = int(np.ceil(gel_t / spacing))

        x = np.linspace(-gel_w / 2, gel_w / 2, nx)
        y = np.linspace(0, gel_h, ny)
        z = np.linspace(0, gel_t, nz)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

        # Shift to positive domain with padding
        # MPM solver has sticky boundary at I[d] < 3 or I[d] >= grid_size[d]-3.
        # Keep particles away from boundary nodes to avoid artificial clamping.
        pad_xy = int(SCENE_PARAMS.get("mpm_grid_padding_cells_xy", 6))
        pad_z_bottom = int(SCENE_PARAMS.get("mpm_grid_padding_cells_z_bottom", 6))
        padding_vec = np.array([pad_xy * dx, pad_xy * dx, pad_z_bottom * dx], dtype=np.float32)
        min_pos = positions.min(axis=0)
        positions += (padding_vec - min_pos)

        self._initial_positions = positions.copy()
        self.initial_positions_m = self._initial_positions.copy()
        self._particle_center = positions.mean(axis=0)
        self.initial_top_z_m = float(positions[:, 2].max())

        # Bottom fixation indices (2*dx thick layer)
        z_min = float(positions[:, 2].min())
        base_thickness = 2.0 * float(dx)
        base_mask = positions[:, 2] <= (z_min + base_thickness)
        self._base_indices_np = np.nonzero(base_mask)[0].astype(np.int32)

        return len(positions)

    def run_trajectory(self, record_interval: int = 10) -> List[np.ndarray]:
        """
        Run press + slide trajectory and record particle positions

        Args:
            record_interval: Record positions every N steps

        Returns:
            List of position arrays
        """
        if self.solver is None:
            self.setup()

        # Initialize particles
        velocities = np.zeros_like(self._initial_positions, dtype=np.float32)
        self.solver.initialize_particles(self._initial_positions, velocities)

        # Trajectory parameters
        press_steps = SCENE_PARAMS['press_steps']
        slide_steps = SCENE_PARAMS['slide_steps']
        hold_steps = SCENE_PARAMS['hold_steps']
        total_steps = press_steps + slide_steps + hold_steps

        press_depth_m = SCENE_PARAMS['press_depth_mm'] * 1e-3
        slide_dist_m = SCENE_PARAMS['slide_distance_mm'] * 1e-3

        self.positions_history = []
        self.frame_controls = []
        self.frame_indenter_centers_m = []

        print(f"Running MPM trajectory: {total_steps} steps")
        start_time = time.time()

        for step in range(total_steps):
            # Compute indenter position
            if step < press_steps:
                # Press phase
                t = step / max(press_steps, 1)
                dz = press_depth_m * t
                dx_slide = 0.0
            elif step < press_steps + slide_steps:
                # Slide phase
                t = (step - press_steps) / max(slide_steps, 1)
                dz = press_depth_m
                dx_slide = slide_dist_m * t
            else:
                # Hold phase
                dz = press_depth_m
                dx_slide = slide_dist_m

            # Update indenter position
            new_center = np.array([
                float(self._indenter_center0[0] + dx_slide),
                float(self._indenter_center0[1]),
                float(self._indenter_center0[2] - dz),
            ], dtype=np.float32)

            # Use numpy interface for reliable Taichi field update
            centers_np = self.solver.obstacle_centers.to_numpy()
            centers_np[1] = new_center
            self.solver.obstacle_centers.from_numpy(centers_np)

            # Verify update took effect
            if step % 50 == 0 and SCENE_PARAMS.get('debug_verbose', False):
                actual = self.solver.obstacle_centers[1]
                print(f"[INDENTER UPDATE] step={step}: target_x={new_center[0]*1000:.2f}mm, actual_x={actual[0]*1000:.2f}mm")

            # Step simulation
            self.solver.step()
            if self._base_fixer is not None:
                try:
                    self._base_fixer()
                except Exception:
                    pass

            # Record positions
            if step % record_interval == 0:
                pos = self.solver.get_particle_data()['x'].copy()        
                self.positions_history.append(pos)
                self.frame_controls.append((float(dz), float(dx_slide)))
                self.frame_indenter_centers_m.append((float(new_center[0]), float(new_center[1]), float(new_center[2])))

                # Debug: check particle displacement
                z_displacements = pos[:, 2] - self._initial_positions[:, 2]
                x_displacements = pos[:, 0] - self._initial_positions[:, 0]
                max_indent = -z_displacements.min()
                max_x_slide = x_displacements.max()

                # Check for ground penetration
                min_z = pos[:, 2].min()
                below_ground = (pos[:, 2] < 0).sum()

                # Verify actual indenter position from solver
                actual_center = self.solver.obstacle_centers[1]
                if SCENE_PARAMS.get('debug_verbose', False):
                    print(f"[MPM SIM] Step {step}: dz={dz*1000:.1f}mm, dx={dx_slide*1000:.1f}mm | "
                          f"indent={max_indent*1000:.1f}mm, x_slide={max_x_slide*1000:.1f}mm | "
                          f"min_z={min_z*1000:.1f}mm, below_ground={below_ground}")

        elapsed = time.time() - start_time
        print(f"MPM trajectory complete: {elapsed:.2f}s, {len(self.positions_history)} frames")

        return self.positions_history


# ==============================================================================
# Comparison Engine
# ==============================================================================
class RGBComparisonEngine:
    """Engine for comparing FEM and MPM sensor RGB outputs"""

    def __init__(
        self,
        fem_file: str,
        object_file: Optional[str] = None,
        mode: str = 'raw',
        visible: bool = True,
        save_dir: Optional[str] = None,
        fem_indenter_face: str = "tip",
        fem_indenter_geom: str = "auto",
    ):
        self.mode = mode
        self.visible = visible
        self.save_dir = Path(save_dir) if save_dir else None
        self.run_context: Dict[str, object] = {}
        self.fem_indenter_face = fem_indenter_face
        self.fem_indenter_geom = fem_indenter_geom

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize renderers
        self.fem_renderer = None
        self.mpm_renderer = None
        self.mpm_sim = None

        if HAS_EZGL:
            self.fem_renderer = FEMRGBRenderer(
                fem_file,
                object_file=object_file,
                visible=False,
                indenter_face=self.fem_indenter_face,
                indenter_geom=self.fem_indenter_geom,
            )
            self.mpm_renderer = MPMHeightFieldRenderer(visible=False)    

        if HAS_TAICHI:
            self.mpm_sim = MPMSimulationAdapter()
        self.fem_show_marker = True
        self.mpm_marker_mode = "static"
        self.mpm_depth_tint = True
        self.mpm_show_indenter = False
        self.mpm_debug_overlay = "off"
        self.indenter_square_size_mm = _infer_square_size_mm_from_stl_path(object_file)

        # Export / audit outputs (only active when --save-dir is set)
        self.export_intermediate = False
        self.export_intermediate_every = 1
        self._exported_image_frames = set()
        self._exported_metrics_frames = set()
        self._exported_intermediate_frames = set()
        self._metrics_rows = []
        self._frame_to_phase: Optional[List[str]] = None

    def set_fem_show_marker(self, show: bool) -> None:
        self.fem_show_marker = bool(show)
        if self.fem_renderer is None:
            return
        sim = getattr(self.fem_renderer, "sensor_sim", None)
        if sim is None:
            return
        try:
            sim.set_show_marker(self.fem_show_marker)
        except Exception:
            pass

    def _write_run_manifest(self, record_interval: int, total_frames: int) -> None:
        if not self.save_dir:
            return

        press_steps = int(SCENE_PARAMS["press_steps"])
        slide_steps = int(SCENE_PARAMS["slide_steps"])
        hold_steps = int(SCENE_PARAMS["hold_steps"])
        total_steps = press_steps + slide_steps + hold_steps

        def _phase_for_step(step: int) -> str:
            if step < press_steps:
                return "press"
            if step < press_steps + slide_steps:
                return "slide"
            return "hold"

        frame_to_step = [int(i * record_interval) for i in range(int(total_frames))]
        frame_to_phase = [_phase_for_step(step) for step in frame_to_step]
        self._frame_to_phase = list(frame_to_phase)
        phase_ranges: Dict[str, Dict[str, int]] = {}
        for i, phase in enumerate(frame_to_phase):
            if phase not in phase_ranges:
                phase_ranges[phase] = {"start_frame": i, "end_frame": i}
            else:
                phase_ranges[phase]["end_frame"] = i

        frame_controls = None
        if self.mpm_sim and self.mpm_sim.frame_controls:
            frame_controls = [
                {
                    "frame": int(i),
                    "press_amount_m": float(press_m),
                    "slide_amount_m": float(slide_m),
                }
                for i, (press_m, slide_m) in enumerate(self.mpm_sim.frame_controls)
            ]

        frame_indenter_centers = None
        if self.mpm_sim and getattr(self.mpm_sim, "frame_indenter_centers_m", None):
            centers = self.mpm_sim.frame_indenter_centers_m
            if centers:
                frame_indenter_centers = [
                    {
                        "frame": int(i),
                        "center_m": [float(cx), float(cy), float(cz)],
                    }
                    for i, (cx, cy, cz) in enumerate(centers)
                ]

        manifest = {
            "created_at": datetime.datetime.now().astimezone().isoformat(),
            "argv": list(sys.argv),
            "run_context": _sanitize_run_context_for_manifest(self.run_context),
            "scene_params": dict(SCENE_PARAMS),
            "deps": {
                "has_taichi": bool(HAS_TAICHI),
                "has_ezgl": bool(HAS_EZGL),
                "has_cv2": bool(HAS_CV2),
            },
            "trajectory": {
                "press_steps": press_steps,
                "slide_steps": slide_steps,
                "hold_steps": hold_steps,
                "total_steps": total_steps,
                "record_interval": int(record_interval),
                "total_frames": int(total_frames),
                "phase_ranges_frames": phase_ranges,
                "frame_to_step": frame_to_step,
                "frame_to_phase": frame_to_phase,
                "frame_controls": frame_controls,
                "frame_indenter_centers_m": frame_indenter_centers,
            },
            "outputs": {
                "frames_glob": {
                    "fem": "fem_*.png",
                    "mpm": "mpm_*.png",
                },
                "run_manifest": "run_manifest.json",
                "metrics": {
                    "csv": "metrics.csv",
                    "json": "metrics.json",
                },
                "intermediate": {
                    "dir": "intermediate",
                    "frames_glob": "intermediate/frame_*.npz",
                },
            },
        }

        manifest_path = self.save_dir / "run_manifest.json"
        try:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Warning: failed to write run manifest: {e}")
            return

        try:
            _write_tuning_notes(
                self.save_dir,
                record_interval=int(record_interval),
                total_frames=int(total_frames),
                run_context=manifest.get("run_context") if isinstance(manifest, dict) else {},
                overwrite=False,
                reason="runtime",
            )
        except Exception as e:
            print(f"Warning: failed to write tuning notes (runtime): {e}")

    def _write_metrics_files(self) -> None:
        if not self.save_dir or not self._metrics_rows:
            return

        metrics_csv_path = self.save_dir / "metrics.csv"
        metrics_json_path = self.save_dir / "metrics.json"
        fieldnames = [
            "frame",
            "phase",
            "mode",
            "mae",
            "mae_r",
            "mae_g",
            "mae_b",
            "max_abs",
            "p50",
            "p90",
            "p99",
        ]

        try:
            with metrics_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self._metrics_rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
        except Exception as e:
            print(f"Warning: failed to write metrics.csv: {e}")

        try:
            mae_values = [float(row.get("mae", 0.0)) for row in self._metrics_rows]
            summary = {"frames": int(len(self._metrics_rows))}
            if mae_values:
                mae_arr = np.array(mae_values, dtype=np.float32)
                summary.update(
                    {
                        "mae_mean": float(mae_arr.mean()),
                        "mae_p50": float(np.percentile(mae_arr, 50)),
                        "mae_p90": float(np.percentile(mae_arr, 90)),
                        "mae_max": float(mae_arr.max()),
                    }
                )
            payload = {
                "created_at": datetime.datetime.now().astimezone().isoformat(),
                "rows": list(self._metrics_rows),
                "summary": summary,
            }
            metrics_json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Warning: failed to write metrics.json: {e}")

    def _compute_fem_contact_mask_u8(self) -> Optional[np.ndarray]:
        if not self.fem_renderer:
            return None
        sim = getattr(self.fem_renderer, "sensor_sim", None)
        fem_sim = getattr(sim, "fem_sim", None)
        if fem_sim is None:
            return None
        try:
            contact_idx = fem_sim.contact_state.contact_idx()
            n_top = int(len(fem_sim.top_nodes))
            mask_flat = np.zeros((n_top,), dtype=np.uint8)
            if contact_idx is not None and contact_idx.shape[1] > 0:
                top_idx = contact_idx[0].astype(np.int64, copy=False)
                mask_flat[top_idx] = 1
            return mask_flat.reshape(fem_sim.mesh_shape)
        except Exception:
            return None

    def _export_intermediate_frame(
        self,
        frame: int,
        mpm_height_field_mm: Optional[np.ndarray],
        mpm_uv_disp_mm: Optional[np.ndarray],
        fem_depth_mm: Optional[np.ndarray],
        fem_marker_disp: Optional[np.ndarray],
        fem_contact_mask_u8: Optional[np.ndarray],
    ) -> None:
        if not self.save_dir:
            return
        intermediate_dir = self.save_dir / "intermediate"
        try:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        payload: Dict[str, object] = {"frame": np.array(int(frame), dtype=np.int32)}
        if mpm_height_field_mm is not None:
            height_field_mm = mpm_height_field_mm.astype(np.float32, copy=False)
            payload["height_field_mm"] = height_field_mm
            payload["contact_mask"] = (height_field_mm < -0.01).astype(np.uint8, copy=False)
        if mpm_uv_disp_mm is not None:
            payload["uv_disp_mm"] = mpm_uv_disp_mm.astype(np.float32, copy=False)
        if fem_depth_mm is not None:
            payload["fem_depth_mm"] = fem_depth_mm.astype(np.float32, copy=False)
        if fem_marker_disp is not None:
            payload["fem_marker_disp"] = fem_marker_disp.astype(np.float32, copy=False)
        if fem_contact_mask_u8 is not None:
            payload["fem_contact_mask_u8"] = fem_contact_mask_u8.astype(np.uint8, copy=False)

        out_path = intermediate_dir / f"frame_{int(frame):04d}.npz"
        try:
            np.savez_compressed(out_path, **payload)
        except Exception as e:
            print(f"Warning: failed to export intermediate frame {frame}: {e}")

    def _export_frame_artifacts(
        self,
        frame_id: int,
        fem_rgb: Optional[np.ndarray],
        mpm_rgb: Optional[np.ndarray],
        mpm_height_field: Optional[np.ndarray],
        mpm_uv_disp: Optional[np.ndarray],
    ) -> None:
        if not self.save_dir:
            return

        # Metrics (one-shot per frame)
        if (
            fem_rgb is not None
            and mpm_rgb is not None
            and frame_id not in self._exported_metrics_frames
        ):
            self._exported_metrics_frames.add(int(frame_id))
            try:
                metrics = _compute_rgb_diff_metrics(fem_rgb, mpm_rgb)
                phase = None
                if self._frame_to_phase is not None and frame_id < len(self._frame_to_phase):
                    phase = self._frame_to_phase[frame_id]
                self._metrics_rows.append(
                    {
                        "frame": int(frame_id),
                        "phase": phase,
                        "mode": str(self.mode),
                        **metrics,
                    }
                )
                self._write_metrics_files()
            except Exception as e:
                print(f"Warning: failed to compute metrics for frame {frame_id}: {e}")

        # Intermediate arrays (one-shot per frame, with --export-intermediate-every)
        if self.export_intermediate and frame_id not in self._exported_intermediate_frames:
            self._exported_intermediate_frames.add(int(frame_id))
            every = int(self.export_intermediate_every) if int(self.export_intermediate_every) > 0 else 1
            if frame_id % every == 0:
                fem_depth_mm = None
                fem_marker_disp = None
                fem_contact_mask_u8 = None
                if self.fem_renderer is not None:
                    try:
                        fem_depth_mm = self.fem_renderer.sensor_sim.get_depth()
                    except Exception:
                        pass
                    try:
                        fem_marker_disp = self.fem_renderer.sensor_sim.get_marker_displacement()
                    except Exception:
                        pass
                    fem_contact_mask_u8 = self._compute_fem_contact_mask_u8()
                self._export_intermediate_frame(
                    frame=frame_id,
                    mpm_height_field_mm=mpm_height_field,
                    mpm_uv_disp_mm=mpm_uv_disp,
                    fem_depth_mm=fem_depth_mm,
                    fem_marker_disp=fem_marker_disp,
                    fem_contact_mask_u8=fem_contact_mask_u8,
                )

        # Frame images (avoid overwriting when UI is looping)
        if HAS_CV2 and fem_rgb is not None and frame_id not in self._exported_image_frames:
            self._exported_image_frames.add(int(frame_id))
            try:
                cv2.imwrite(
                    str(self.save_dir / f"fem_{int(frame_id):04d}.png"),
                    cv2.cvtColor(fem_rgb, cv2.COLOR_RGB2BGR),
                )
                if mpm_rgb is not None:
                    cv2.imwrite(
                        str(self.save_dir / f"mpm_{int(frame_id):04d}.png"),
                        cv2.cvtColor(mpm_rgb, cv2.COLOR_RGB2BGR),
                    )
            except Exception as e:
                print(f"Warning: failed to write frame images for {frame_id}: {e}")

    def run_comparison(self, fps: int = 30, record_interval: int = 5, interactive: bool = True):
        """Run side-by-side comparison visualization"""
        if not HAS_EZGL:
            print("ezgl not available, cannot run visualization")
            return

        # Run MPM simulation first to get position history
        if self.mpm_sim:
            positions_history = self.mpm_sim.run_trajectory(record_interval=record_interval)
        else:
            positions_history = []
            print("MPM not available, showing FEM only")

        total_frames = len(positions_history) if positions_history else 100
        self._write_run_manifest(record_interval=record_interval, total_frames=total_frames)

        if interactive:
            # Create UI (loops until closed)
            self._create_ui(positions_history, fps)
        else:
            # Headless batch export (run once and exit)
            self._run_batch(positions_history)

    def _run_batch(self, mpm_positions: List[np.ndarray]) -> None:
        """Run a finite headless loop to export frames, metrics and intermediates."""
        if not self.save_dir:
            print("ERROR: batch mode requires --save-dir")
            return

        total_frames = len(mpm_positions) if mpm_positions else 100
        press_steps = SCENE_PARAMS['press_steps']
        slide_steps = SCENE_PARAMS['slide_steps']
        hold_steps = SCENE_PARAMS['hold_steps']
        total_steps = press_steps + slide_steps + hold_steps
        press_end_ratio = press_steps / total_steps if total_steps > 0 else 0.0
        slide_end_ratio = (press_steps + slide_steps) / total_steps if total_steps > 0 else 0.0

        # Pre-configure MPM renderer mapping if we have initial particle positions
        if (
            self.mpm_renderer
            and self.mpm_sim
            and self.mpm_sim.initial_positions_m is not None
            and mpm_positions
        ):
            self.mpm_renderer.configure_from_initial_positions(
                self.mpm_sim.initial_positions_m, self.mpm_sim.initial_top_z_m
            )
            self.mpm_renderer.scene.set_marker_mode(self.mpm_marker_mode)
            self.mpm_renderer.scene.set_depth_tint_enabled(self.mpm_depth_tint)
            self.mpm_renderer.scene.set_indenter_overlay(
                self.mpm_show_indenter, square_size_mm=self.indenter_square_size_mm
            )
            self.mpm_renderer.scene.set_debug_overlay(self.mpm_debug_overlay)

        for frame_id in range(int(total_frames)):
            # Prefer recorded MPM control signals (strict frame alignment)
            if self.mpm_sim and self.mpm_sim.frame_controls and frame_id < len(self.mpm_sim.frame_controls):
                press_amount_m, slide_amount_m = self.mpm_sim.frame_controls[frame_id]
                press_y_mm = float(press_amount_m) * 1000.0
                slide_x_mm = float(slide_amount_m) * 1000.0
            else:
                # Fallback: Use consistent phase ratios with MPM trajectory
                t = frame_id / max(total_frames - 1, 1)
                if t < press_end_ratio:
                    phase_t = t / press_end_ratio if press_end_ratio > 0 else 0
                    press_y_mm = float(SCENE_PARAMS['press_depth_mm']) * phase_t
                    slide_x_mm = 0.0
                elif t < slide_end_ratio:
                    phase_t = (t - press_end_ratio) / (slide_end_ratio - press_end_ratio) if (slide_end_ratio - press_end_ratio) > 0 else 0
                    press_y_mm = float(SCENE_PARAMS['press_depth_mm'])
                    slide_x_mm = float(SCENE_PARAMS['slide_distance_mm']) * phase_t
                else:
                    press_y_mm = float(SCENE_PARAMS['press_depth_mm'])
                    slide_x_mm = float(SCENE_PARAMS['slide_distance_mm'])

            if frame_id % max(total_frames // 10, 1) == 0:
                print(f"[BATCH] frame={frame_id}/{total_frames-1} press={press_y_mm:.2f}mm slide={slide_x_mm:.2f}mm")

            fem_rgb = None
            if self.fem_renderer:
                y_pos_mm = -press_y_mm
                self.fem_renderer.set_indenter_pose(slide_x_mm, y_pos_mm, 0.0)
                fem_rgb = self.fem_renderer.step()
                if self.mode == "diff":
                    fem_rgb = self.fem_renderer.get_diff_image()

            mpm_rgb = None
            mpm_height_field = None
            mpm_uv_disp = None
            if self.mpm_renderer and self.mpm_sim and mpm_positions and frame_id < len(mpm_positions):
                pos = mpm_positions[frame_id]
                indenter_center_m = None
                if (
                    self.mpm_sim
                    and getattr(self.mpm_sim, "frame_indenter_centers_m", None)
                    and frame_id < len(self.mpm_sim.frame_indenter_centers_m)
                ):
                    indenter_center_m = self.mpm_sim.frame_indenter_centers_m[frame_id]
                mpm_height_field, mpm_uv_disp = self.mpm_renderer.extract_surface_fields(
                    pos,
                    self.mpm_sim.initial_top_z_m,
                    indenter_center_m=indenter_center_m,
                )
                self.mpm_renderer.scene.set_uv_displacement(mpm_uv_disp)
                if self.mode == 'diff':
                    mpm_rgb = self.mpm_renderer.get_diff_image(mpm_height_field)
                else:
                    mpm_rgb = self.mpm_renderer.render(mpm_height_field)

                if self.mpm_show_indenter and self.mpm_sim and self.mpm_sim.frame_controls and frame_id < len(self.mpm_sim.frame_controls):
                    _, slide_amount_m = self.mpm_sim.frame_controls[frame_id]
                    self.mpm_renderer.scene.set_indenter_center(float(slide_amount_m) * 1000.0, SCENE_PARAMS['gel_size_mm'][1] / 2.0)

            self._export_frame_artifacts(
                frame_id=frame_id,
                fem_rgb=fem_rgb,
                mpm_rgb=mpm_rgb,
                mpm_height_field=mpm_height_field,
                mpm_uv_disp=mpm_uv_disp,
            )

        print(f"[BATCH] done: frames={total_frames}, save_dir={self.save_dir}")

    def _create_ui(self, mpm_positions: List[np.ndarray], fps: int):
        """Create side-by-side image display UI"""
        frame_idx = [0]
        total_frames = len(mpm_positions) if mpm_positions else 100

        # Trajectory phase boundaries (matching MPM trajectory phases)
        press_steps = SCENE_PARAMS['press_steps']
        slide_steps = SCENE_PARAMS['slide_steps']
        hold_steps = SCENE_PARAMS['hold_steps']
        total_steps = press_steps + slide_steps + hold_steps

        # Calculate phase boundaries as ratios
        press_end_ratio = press_steps / total_steps
        slide_end_ratio = (press_steps + slide_steps) / total_steps

        # Storage for image view widgets (set in UI building)
        fem_view = [None]
        mpm_view = [None]

        # Pre-configure MPM renderer mapping if we have initial particle positions
        if self.mpm_renderer and self.mpm_sim and self.mpm_sim.initial_positions_m is not None and mpm_positions:
            self.mpm_renderer.configure_from_initial_positions(
                self.mpm_sim.initial_positions_m, self.mpm_sim.initial_top_z_m
            )
            self.mpm_renderer.scene.set_marker_mode(self.mpm_marker_mode)
            self.mpm_renderer.scene.set_depth_tint_enabled(self.mpm_depth_tint)
            self.mpm_renderer.scene.set_indenter_overlay(self.mpm_show_indenter, square_size_mm=self.indenter_square_size_mm)
            self.mpm_renderer.scene.set_debug_overlay(self.mpm_debug_overlay)

        def on_timeout():
            # Get FEM image
            press_depth = SCENE_PARAMS['press_depth_mm']
            slide_dist = SCENE_PARAMS['slide_distance_mm']

            # Prefer recorded MPM control signals (strict frame alignment).
            if self.mpm_sim and self.mpm_sim.frame_controls and frame_idx[0] < len(self.mpm_sim.frame_controls):
                press_amount_m, slide_amount_m = self.mpm_sim.frame_controls[frame_idx[0]]
                press_y_mm = float(press_amount_m) * 1000.0
                slide_x_mm = float(slide_amount_m) * 1000.0
            else:
                # Fallback: Use consistent phase ratios with MPM trajectory
                t = frame_idx[0] / max(total_frames - 1, 1)
                if t < press_end_ratio:
                    phase_t = t / press_end_ratio if press_end_ratio > 0 else 0
                    press_y_mm = press_depth * phase_t
                    slide_x_mm = 0.0
                elif t < slide_end_ratio:
                    phase_t = (t - press_end_ratio) / (slide_end_ratio - press_end_ratio) if (slide_end_ratio - press_end_ratio) > 0 else 0
                    press_y_mm = press_depth
                    slide_x_mm = slide_dist * phase_t
                else:
                    press_y_mm = press_depth
                    slide_x_mm = slide_dist

            # FEM rendering
            # Depth camera: eye=(0,0,0), center=(0,1,0) -> faces +y direction
            # FEM expects depth values close to 0 or negative for contact detection.
            # From debug output: depth_min equals object center y-position (not surface).
            # So we place object center at y = -press to get negative depth when pressing.
            fem_rgb = None
            if self.fem_renderer:
                # Object center at y=0 means just touching surface (depth=0)
                # Object center at y=-press means pressing into surface (depth<0)
                y_pos_mm = -press_y_mm
                print(f"[FEM] Frame {frame_idx[0]}: press={press_y_mm:.2f}mm, slide={slide_x_mm:.2f}mm")
                self.fem_renderer.set_indenter_pose(slide_x_mm, y_pos_mm, 0.0)

                if SCENE_PARAMS.get("debug_verbose", False):
                    mpm_center_mm = None
                    if (
                        self.mpm_sim
                        and getattr(self.mpm_sim, "frame_indenter_centers_m", None)
                        and frame_idx[0] < len(self.mpm_sim.frame_indenter_centers_m)
                    ):
                        cx, cy, cz = self.mpm_sim.frame_indenter_centers_m[frame_idx[0]]
                        mpm_center_mm = [cx * 1000.0, cy * 1000.0, cz * 1000.0]

                    fem_raw_mm = None
                    try:
                        fem_raw_mm = (self.fem_renderer.depth_scene.get_object_pose_raw().xyz * 1000.0).tolist()
                    except Exception:
                        pass

                    if mpm_center_mm is not None or fem_raw_mm is not None:
                        print(f"[POSE] frame={frame_idx[0]} mpm_center_mm={mpm_center_mm} fem_raw_pose_mm={fem_raw_mm}")
                fem_rgb = self.fem_renderer.step()
                if self.mode == 'diff':
                    fem_rgb = self.fem_renderer.get_diff_image()
                if fem_view[0] is not None:
                    fem_view[0].setData(fem_rgb)

            # MPM rendering
            mpm_rgb = None
            mpm_height_field = None
            mpm_uv_disp = None
            if self.mpm_renderer and mpm_positions and frame_idx[0] < len(mpm_positions):
                pos = mpm_positions[frame_idx[0]]
                indenter_center_m = None
                if (
                    self.mpm_sim
                    and getattr(self.mpm_sim, "frame_indenter_centers_m", None)
                    and frame_idx[0] < len(self.mpm_sim.frame_indenter_centers_m)
                ):
                    indenter_center_m = self.mpm_sim.frame_indenter_centers_m[frame_idx[0]]
                mpm_height_field, mpm_uv_disp = self.mpm_renderer.extract_surface_fields(
                    pos,
                    self.mpm_sim.initial_top_z_m,
                    indenter_center_m=indenter_center_m,
                )
                self.mpm_renderer.scene.set_uv_displacement(mpm_uv_disp)
                if self.mode == 'diff':
                    mpm_rgb = self.mpm_renderer.get_diff_image(mpm_height_field)
                else:
                    mpm_rgb = self.mpm_renderer.render(mpm_height_field)
                if mpm_view[0] is not None:
                    mpm_view[0].setData(mpm_rgb)

                if self.mpm_show_indenter and self.mpm_sim and self.mpm_sim.frame_controls and frame_idx[0] < len(self.mpm_sim.frame_controls):
                    _, slide_amount_m = self.mpm_sim.frame_controls[frame_idx[0]]
                    # y 方向用胶体中心做可视化对齐（square 压头在 y 方向不移动）
                    self.mpm_renderer.scene.set_indenter_center(float(slide_amount_m) * 1000.0, SCENE_PARAMS['gel_size_mm'][1] / 2.0)

            if self.save_dir:
                frame_id = int(frame_idx[0])
                self._export_frame_artifacts(
                    frame_id=frame_id,
                    fem_rgb=fem_rgb,
                    mpm_rgb=mpm_rgb,
                    mpm_height_field=mpm_height_field,
                    mpm_uv_disp=mpm_uv_disp,
                )

            frame_idx[0] = (frame_idx[0] + 1) % total_frames

        # Build UI
        with tb.window("MPM vs FEM RGB Compare", None, 10, pos=(100, 100), size=(900, 800)):
            tb.add_text(f"Mode: {self.mode} | FPS: {fps}")
            tb.add_spacer(10)

            with tb.group("images", horizontal=True, show=False):
                # FEM panel
                with tb.group("FEM Sensor", horizontal=False, show=True):
                    fem_view[0] = tb.add_image_view("fem_image", None, img_size=(400, 700), img_format="rgb")

                # MPM panel
                with tb.group("MPM Sensor", horizontal=False, show=True):
                    mpm_view[0] = tb.add_image_view("mpm_image", None, img_size=(400, 700), img_format="rgb")

            tb.add_timer("update_timer", int(1000 / fps), on_timeout)

        tb.exec()


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='MPM vs FEM Sensor RGB Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--fem-file', type=str,
        default=str(PROJ_DIR / "assets/data/fem_data_gel_2035.npz"),
        help='Path to FEM NPZ file'
    )
    parser.add_argument(
        '--object-file', type=str, default=None,
        help='Path to indenter STL file (default: sphere)'
    )
    parser.add_argument(
        '--fem-indenter-face', type=str, choices=['base', 'tip'], default='tip',
        help='FEM indenter STL face selection: base (y_min, may look like square base) or tip (y_max, round tip)'
    )
    parser.add_argument(
        '--fem-indenter-geom', type=str, choices=['auto', 'stl', 'box', 'sphere'], default='auto',
        help=('FEM indenter geometry: auto (stl when --object-file is set or when '
              'MPM indenter-type=cylinder; otherwise match MPM indenter-type), '
              'stl, box, sphere')
    )
    parser.add_argument(
        '--mode', type=str, choices=['raw', 'diff'], default='raw',
        help='Visualization mode: raw (direct RGB) or diff (relative to reference)'
    )
    parser.add_argument(
        '--press-mm', type=float, default=SCENE_PARAMS['press_depth_mm'],
        help='Press depth in mm'
    )
    parser.add_argument(
        '--slide-mm', type=float, default=SCENE_PARAMS['slide_distance_mm'],
        help='Slide distance in mm'
    )
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Total simulation steps (default: press + slide + hold)'    
    )
    parser.add_argument(
        '--record-interval', type=int, default=5,
        help='Record MPM positions every N steps (affects total frames and phase mapping)'
    )
    parser.add_argument(
        '--indenter-type',
        type=str,
        choices=['sphere', 'cylinder', 'box'],
        default=str(SCENE_PARAMS.get('indenter_type', 'cylinder')),
        help=('MPM indenter type: sphere (curved) | cylinder (flat round, matches '
              'circle_r4.STL tip) | box (flat square)')
    )
    parser.add_argument(
        '--fric', type=float, default=None,
        help=('Set friction for both FEM and MPM: FEM fric_coef and MPM mu_s/mu_k '
              '(FEM uses a single coefficient; MPM uses static/kinetic).')
    )
    parser.add_argument(
        '--fem-fric', type=float, default=None,
        help='FEM friction coefficient (overrides fem_fric_coef)'
    )
    parser.add_argument(
        '--mpm-mu-s', type=float, default=None,
        help='MPM static friction coefficient mu_s (overrides --fric for MPM side)'
    )
    parser.add_argument(
        '--mpm-mu-k', type=float, default=None,
        help='MPM kinetic friction coefficient mu_k (overrides --fric for MPM side)'
    )
    parser.add_argument(
        '--mpm-k-normal', type=float, default=None,
        help='MPM contact stiffness (normal) (overrides scene default)'
    )
    parser.add_argument(
        '--mpm-k-tangent', type=float, default=None,
        help='MPM contact stiffness (tangent) (overrides scene default)'
    )
    parser.add_argument(
        '--mpm-dt', type=float, default=None,
        help='MPM time step dt in seconds (overrides scene default)'
    )
    parser.add_argument(
        '--mpm-ogden-mu', type=float, default=None,
        help='MPM Ogden shear modulus mu in Pa (overrides scene default; sets single-term Ogden)'
    )
    parser.add_argument(
        '--mpm-ogden-kappa', type=float, default=None,
        help='MPM Ogden bulk modulus kappa in Pa (overrides scene default)'
    )
    parser.add_argument(
        '--mpm-enable-bulk-viscosity', type=str, choices=['on', 'off'], default=None,
        help='Enable Kelvin-Voigt bulk viscosity for MPM: on|off'
    )
    parser.add_argument(
        '--mpm-bulk-viscosity', type=float, default=None,
        help='MPM bulk viscosity coefficient eta_bulk in Pa*s (enables bulk viscosity if provided)'
    )
    parser.add_argument(
        '--fem-marker', type=str, choices=['on', 'off'], default='on',
        help='FEM marker rendering: on|off (off = white background for shading comparison)'
    )
    parser.add_argument(
        '--mpm-marker', type=str, choices=['off', 'static', 'warp'], default='static',
        help='MPM marker rendering: off|static|warp (warp reflects stretch/shear from tangential displacement)'
    )
    parser.add_argument(
        '--mpm-depth-tint', type=str, choices=['on', 'off'], default='on',
        help='MPM depth tint overlay on marker texture: on|off'
    )
    parser.add_argument(
        '--mpm-height-fill-holes', type=str, choices=['on', 'off'], default='on',
        help='MPM height_field hole filling (diffusion) before rendering: on|off'
    )
    parser.add_argument(
        '--mpm-height-fill-holes-iters', type=int, default=int(SCENE_PARAMS.get("mpm_height_fill_holes_iters", 10)),
        help='Iterations for --mpm-height-fill-holes (default: 10)'
    )
    parser.add_argument(
        '--mpm-height-smooth', type=str, choices=['on', 'off'], default='on',
        help='MPM height_field smoothing before rendering: on|off'
    )
    parser.add_argument(
        '--mpm-height-smooth-iters', type=int, default=int(SCENE_PARAMS.get("mpm_height_smooth_iters", 2)),
        help='Box blur iterations for --mpm-height-smooth (default: 2)'
    )
    parser.add_argument(
        '--mpm-height-reference-edge', type=str, choices=['on', 'off'], default='on',
        help='MPM height_field baseline reference: edge (on) vs none (off)'
    )
    parser.add_argument(
        '--mpm-height-clamp-indenter', type=str, choices=['on', 'off'], default='on',
        help='Clamp MPM height_field not below indenter surface (suppress penetration artifacts): on|off'
    )
    parser.add_argument(
        '--mpm-height-clip-outliers', type=str, choices=['on', 'off'], default='off',
        help='Clip extreme negative MPM height_field outside indenter footprint: on|off'
    )
    parser.add_argument(
        '--mpm-height-clip-outliers-min-mm', type=float, default=float(SCENE_PARAMS.get("mpm_height_clip_outliers_min_mm", 5.0)),
        help='Negative depth threshold in mm for --mpm-height-clip-outliers (default: 5.0)'
    )
    parser.add_argument(
        '--mpm-show-indenter', action='store_true', default=False,
        help='Overlay MPM indenter projection in the RGB view (2D overlay)'
    )
    parser.add_argument(
        '--mpm-debug-overlay', type=str, choices=['off', 'uv', 'warp'], default='off',
        help='MPM debug overlay mode'
    )
    parser.add_argument(
        '--indenter-size-mm', type=float, default=None,
        help='Square indenter side length in mm (only for box mode). If omitted, try infer from STL name like square_d6.STL.'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Display frame rate'
    )
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='Directory to save frame images'
    )
    parser.add_argument(
        '--interactive', action='store_true', default=False,
        help='Run interactive UI loop (default: headless batch when --save-dir is set)'
    )
    parser.add_argument(
        '--export-intermediate', action='store_true', default=False,
        help='Export intermediate arrays (height_field_mm/uv_disp_mm/contact_mask) to --save-dir/intermediate (npz)'
    )
    parser.add_argument(
        '--export-intermediate-every', type=int, default=1,
        help='Export intermediate every N frames (default: 1); effective only with --export-intermediate'
    )
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Enable verbose per-frame debug logging'
    )

    args = parser.parse_args()

    def _validate_nonneg(name: str, value: Optional[float]) -> bool:
        if value is None:
            return True
        if float(value) < 0.0:
            print(f"ERROR: {name} must be >= 0")
            return False
        return True
    def _validate_pos(name: str, value: Optional[float]) -> bool:
        if value is None:
            return True
        if float(value) <= 0.0:
            print(f"ERROR: {name} must be > 0")
            return False
        return True

    if not (
        _validate_nonneg("--fric", args.fric)
        and _validate_nonneg("--fem-fric", args.fem_fric)
        and _validate_nonneg("--mpm-mu-s", args.mpm_mu_s)
        and _validate_nonneg("--mpm-mu-k", args.mpm_mu_k)
        and _validate_nonneg("--mpm-k-normal", args.mpm_k_normal)
        and _validate_nonneg("--mpm-k-tangent", args.mpm_k_tangent)
        and _validate_pos("--mpm-dt", args.mpm_dt)
        and _validate_pos("--mpm-ogden-mu", args.mpm_ogden_mu)
        and _validate_pos("--mpm-ogden-kappa", args.mpm_ogden_kappa)
        and _validate_nonneg("--mpm-bulk-viscosity", args.mpm_bulk_viscosity)
    ):
        return 1

    if int(args.export_intermediate_every) <= 0:
        print("ERROR: --export-intermediate-every must be a positive integer")
        return 1
    if int(args.mpm_height_fill_holes_iters) < 0:
        print("ERROR: --mpm-height-fill-holes-iters must be >= 0")
        return 1
    if int(args.mpm_height_smooth_iters) < 0:
        print("ERROR: --mpm-height-smooth-iters must be >= 0")
        return 1
    if str(args.mpm_height_clip_outliers).lower().strip() == "on" and float(args.mpm_height_clip_outliers_min_mm) <= 0.0:
        print("ERROR: --mpm-height-clip-outliers-min-mm must be > 0 when --mpm-height-clip-outliers on")
        return 1

    # Update scene params from args
    SCENE_PARAMS['press_depth_mm'] = args.press_mm
    SCENE_PARAMS['slide_distance_mm'] = args.slide_mm
    SCENE_PARAMS['indenter_type'] = args.indenter_type
    SCENE_PARAMS['debug_verbose'] = args.debug
    if args.mpm_dt is not None:
        SCENE_PARAMS['mpm_dt'] = float(args.mpm_dt)
    if args.mpm_ogden_mu is not None:
        SCENE_PARAMS['ogden_mu'] = [float(args.mpm_ogden_mu)]
        # 保持与单项 mu 对齐，避免长度不一致导致本构计算混淆
        if isinstance(SCENE_PARAMS.get('ogden_alpha'), list) and len(SCENE_PARAMS['ogden_alpha']) != 1:
            SCENE_PARAMS['ogden_alpha'] = [float(SCENE_PARAMS['ogden_alpha'][0])]
    if args.mpm_ogden_kappa is not None:
        SCENE_PARAMS['ogden_kappa'] = float(args.mpm_ogden_kappa)
    if args.mpm_enable_bulk_viscosity is not None:
        SCENE_PARAMS["mpm_enable_bulk_viscosity"] = (str(args.mpm_enable_bulk_viscosity).lower().strip() == "on")
    if args.mpm_bulk_viscosity is not None:
        SCENE_PARAMS["mpm_bulk_viscosity"] = float(args.mpm_bulk_viscosity)
        # 用户显式提供 eta_bulk 时，默认启用体粘性（除非显式 --mpm-enable-bulk-viscosity off）
        if str(args.mpm_enable_bulk_viscosity).lower().strip() != "off":
            SCENE_PARAMS["mpm_enable_bulk_viscosity"] = True
    SCENE_PARAMS["mpm_height_fill_holes"] = (str(args.mpm_height_fill_holes).lower().strip() == "on")
    SCENE_PARAMS["mpm_height_fill_holes_iters"] = int(args.mpm_height_fill_holes_iters)
    SCENE_PARAMS["mpm_height_smooth"] = (str(args.mpm_height_smooth).lower().strip() != "off")
    SCENE_PARAMS["mpm_height_smooth_iters"] = int(args.mpm_height_smooth_iters)
    SCENE_PARAMS["mpm_height_reference_edge"] = (str(args.mpm_height_reference_edge).lower().strip() != "off")
    SCENE_PARAMS["mpm_height_clamp_indenter"] = (str(args.mpm_height_clamp_indenter).lower().strip() != "off")
    SCENE_PARAMS["mpm_height_clip_outliers"] = (str(args.mpm_height_clip_outliers).lower().strip() == "on")
    SCENE_PARAMS["mpm_height_clip_outliers_min_mm"] = float(args.mpm_height_clip_outliers_min_mm)

    if args.fric is not None:
        fric = float(args.fric)
        SCENE_PARAMS["fem_fric_coef"] = fric
        SCENE_PARAMS["mpm_mu_s"] = fric
        SCENE_PARAMS["mpm_mu_k"] = fric
    if args.fem_fric is not None:
        SCENE_PARAMS["fem_fric_coef"] = float(args.fem_fric)
    if args.mpm_mu_s is not None:
        SCENE_PARAMS["mpm_mu_s"] = float(args.mpm_mu_s)
    if args.mpm_mu_k is not None:
        SCENE_PARAMS["mpm_mu_k"] = float(args.mpm_mu_k)
    if args.mpm_k_normal is not None:
        SCENE_PARAMS["mpm_contact_stiffness_normal"] = float(args.mpm_k_normal)
    if args.mpm_k_tangent is not None:
        SCENE_PARAMS["mpm_contact_stiffness_tangent"] = float(args.mpm_k_tangent)

    if args.indenter_size_mm is not None:
        square_d_mm = float(args.indenter_size_mm)
    else:
        square_d_mm = _infer_square_size_mm_from_stl_path(args.object_file)
    if square_d_mm is not None and args.indenter_type == "box":
        half = square_d_mm / 2.0
        SCENE_PARAMS['indenter_half_extents_mm'] = (half, half, half)

    # Handle --steps: distribute among press/slide/hold phases
    if args.steps is not None:
        # Distribute: 30% press, 55% slide, 15% hold
        SCENE_PARAMS['press_steps'] = int(args.steps * 0.30)
        SCENE_PARAMS['slide_steps'] = int(args.steps * 0.55)
        SCENE_PARAMS['hold_steps'] = args.steps - SCENE_PARAMS['press_steps'] - SCENE_PARAMS['slide_steps']

    if args.record_interval <= 0:
        print("ERROR: --record-interval must be a positive integer")
        return 1

    print("=" * 60)
    print("MPM vs FEM Sensor RGB Comparison")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Press depth: {args.press_mm} mm")
    print(f"Slide distance: {args.slide_mm} mm")
    print(f"MPM indenter type: {args.indenter_type}")
    fem_fric = float(SCENE_PARAMS.get("fem_fric_coef", 0.4))
    mpm_mu_s = float(SCENE_PARAMS.get("mpm_mu_s", 2.0))
    mpm_mu_k = float(SCENE_PARAMS.get("mpm_mu_k", 1.5))
    aligned_fric = (abs(fem_fric - mpm_mu_s) < 1e-9) and (abs(fem_fric - mpm_mu_k) < 1e-9)
    print(f"Friction: FEM fric_coef={fem_fric:.4g}, MPM mu_s={mpm_mu_s:.4g}, mu_k={mpm_mu_k:.4g}, aligned={aligned_fric}")
    k_n = float(SCENE_PARAMS.get("mpm_contact_stiffness_normal", 0.0))
    k_t = float(SCENE_PARAMS.get("mpm_contact_stiffness_tangent", 0.0))
    print(f"MPM contact stiffness: k_n={k_n:.4g}, k_t={k_t:.4g}")
    print(
        f"MPM material: ogden_mu={SCENE_PARAMS.get('ogden_mu')}, "
        f"ogden_alpha={SCENE_PARAMS.get('ogden_alpha')}, "
        f"ogden_kappa={SCENE_PARAMS.get('ogden_kappa')}, "
        f"bulk_viscosity={'on' if bool(SCENE_PARAMS.get('mpm_enable_bulk_viscosity', False)) else 'off'} "
        f"(eta={float(SCENE_PARAMS.get('mpm_bulk_viscosity', 0.0)):.4g})"
    )

    gel_w_mm, gel_h_mm = [float(v) for v in SCENE_PARAMS.get("gel_size_mm", (0.0, 0.0))]
    cam_w_mm = float(SCENE_PARAMS.get("cam_view_width_m", 0.0)) * 1000.0
    cam_h_mm = float(SCENE_PARAMS.get("cam_view_height_m", 0.0)) * 1000.0
    tol_mm = 0.2
    dw_mm = cam_w_mm - gel_w_mm
    dh_mm = cam_h_mm - gel_h_mm
    scale_consistent = (abs(dw_mm) <= tol_mm) and (abs(dh_mm) <= tol_mm)
    print(
        f"Scale: gel_size_mm=({gel_w_mm:.2f}, {gel_h_mm:.2f}), "
        f"cam_view_mm=({cam_w_mm:.2f}, {cam_h_mm:.2f}), "
        f"delta_mm=({dw_mm:+.2f}, {dh_mm:+.2f}), consistent={scale_consistent}"
    )
    if not scale_consistent:
        print("Note: gel_size_mm follows VecTouchSim defaults; cam_view_* follows demo_simple_sensor camera calibration.")

    fem_indenter_geom = args.fem_indenter_geom
    if fem_indenter_geom == "auto":
        if args.object_file:
            fem_indenter_geom = "stl"
        elif str(args.indenter_type).lower().strip() == "cylinder":
            # FEM 侧没有 cylinder primitive，使用 circle_r4.STL (tip) 作为圆柱压头基线。
            fem_indenter_geom = "stl"
        else:
            fem_indenter_geom = args.indenter_type
    print(f"FEM indenter geom: {fem_indenter_geom}")
    if args.object_file and fem_indenter_geom != "stl":
        print(f"Note: --object-file is ignored because --fem-indenter-geom={fem_indenter_geom}")
    if fem_indenter_geom == "stl":
        default_stl = _PROJECT_ROOT / "xengym" / "assets" / "obj" / "circle_r4.STL"
        stl_path_display = args.object_file if args.object_file else str(default_stl)
        print(f"FEM indenter STL: {stl_path_display}")
        print(f"FEM indenter face: {args.fem_indenter_face}")

    # Print effective indenter size (MPM vs FEM) for auditability.
    mpm_indenter_size: Optional[Dict[str, object]] = None
    if args.indenter_type == "box":
        half_extents_mm = SCENE_PARAMS.get("indenter_half_extents_mm", None)
        if half_extents_mm is None:
            r_mm = float(SCENE_PARAMS["indenter_radius_mm"])
            half_extents_mm = (r_mm, r_mm, r_mm)
        hx_mm, hy_mm, hz_mm = [float(v) for v in half_extents_mm]
        mpm_indenter_size = {
            "half_extents_mm": [float(hx_mm), float(hy_mm), float(hz_mm)],
            "full_extents_mm": [float(hx_mm * 2.0), float(hy_mm * 2.0), float(hz_mm * 2.0)],
        }
        print(f"Indenter size (box, mm): half_extents=({hx_mm:.2f}, {hy_mm:.2f}, {hz_mm:.2f}), "
              f"full=({hx_mm*2:.2f}, {hy_mm*2:.2f}, {hz_mm*2:.2f})")
    elif args.indenter_type == "cylinder":
        r_mm = float(SCENE_PARAMS["indenter_radius_mm"])
        half_h_mm = SCENE_PARAMS.get("indenter_cylinder_half_height_mm", None)
        if half_h_mm is None:
            half_h_mm = r_mm
        half_h_mm = float(half_h_mm)
        mpm_indenter_size = {
            "radius_mm": float(r_mm),
            "diameter_mm": float(r_mm * 2.0),
            "half_height_mm": float(half_h_mm),
            "height_mm": float(half_h_mm * 2.0),
        }
        print(
            f"Indenter size (cylinder, mm): radius={r_mm:.2f}, diameter={r_mm*2:.2f}, "
            f"height={half_h_mm*2:.2f}"
        )
    else:
        r_mm = float(SCENE_PARAMS["indenter_radius_mm"])
        mpm_indenter_size = {
            "radius_mm": float(r_mm),
            "diameter_mm": float(r_mm * 2.0),
        }
        print(f"Indenter size (sphere, mm): radius={r_mm:.2f}, diameter={r_mm*2:.2f}")

    stl_stats = None
    fem_contact_face_key = None
    fem_contact_face_size_mm = None
    if fem_indenter_geom == "stl":
        stl_path = Path(args.object_file) if args.object_file else (_PROJECT_ROOT / "xengym" / "assets" / "obj" / "circle_r4.STL")
        stl_stats = _analyze_binary_stl_endfaces_mm(stl_path) if stl_path.exists() else None
        if stl_stats is not None:
            try:
                ymin = stl_stats["endfaces_mm"]["y_min"]
                ymax = stl_stats["endfaces_mm"]["y_max"]
                print(
                    "Indenter STL endfaces (mm): "
                    f"y_min size≈{ymin['size_x_mm']:.1f}x{ymin['size_z_mm']:.1f}, "
                    f"y_max size≈{ymax['size_x_mm']:.1f}x{ymax['size_z_mm']:.1f}, "
                    f"height≈{float(stl_stats['height_mm']):.1f}"
                )
            except Exception:
                pass
            try:
                fem_contact_face_key = "y_max" if str(args.fem_indenter_face).lower().strip() == "tip" else "y_min"
                endfaces = stl_stats.get("endfaces_mm") if isinstance(stl_stats, dict) else None
                endface = endfaces.get(fem_contact_face_key) if isinstance(endfaces, dict) else None
                if isinstance(endface, dict):
                    fem_contact_face_size_mm = {
                        "size_x_mm": float(endface.get("size_x_mm", 0.0)),
                        "size_z_mm": float(endface.get("size_z_mm", 0.0)),
                    }
                    print(
                        f"FEM contact face ({args.fem_indenter_face}/{fem_contact_face_key}) "
                        f"size≈{fem_contact_face_size_mm['size_x_mm']:.1f}x{fem_contact_face_size_mm['size_z_mm']:.1f}mm"
                    )
            except Exception:
                pass
    print(f"MPM marker: {args.mpm_marker}")
    print(f"FEM marker: {args.fem_marker}")
    print(f"MPM depth tint: {args.mpm_depth_tint}")
    print(
        "MPM height_field: "
        f"fill_holes={bool(SCENE_PARAMS.get('mpm_height_fill_holes', False))} "
        f"(iters={int(SCENE_PARAMS.get('mpm_height_fill_holes_iters', 0))}), "
        f"smooth={bool(SCENE_PARAMS.get('mpm_height_smooth', True))} "
        f"(iters={int(SCENE_PARAMS.get('mpm_height_smooth_iters', 0))}), "
        f"ref_edge={bool(SCENE_PARAMS.get('mpm_height_reference_edge', True))}, "
        f"clamp_indenter={bool(SCENE_PARAMS.get('mpm_height_clamp_indenter', True))}, "
        f"clip_outliers={bool(SCENE_PARAMS.get('mpm_height_clip_outliers', False))} "
        f"(min_mm={float(SCENE_PARAMS.get('mpm_height_clip_outliers_min_mm', 0.0)):.3f})"
    )
    if args.mpm_show_indenter:
        print("MPM indenter overlay: enabled")
    if args.steps:
        print(f"Total steps: {args.steps}")
    press_steps = int(SCENE_PARAMS["press_steps"])
    slide_steps = int(SCENE_PARAMS["slide_steps"])
    hold_steps = int(SCENE_PARAMS["hold_steps"])
    total_steps = press_steps + slide_steps + hold_steps
    expected_frames = (total_steps + int(args.record_interval) - 1) // int(args.record_interval)
    print(f"Record interval: {args.record_interval} (expected frames: {expected_frames})")
    print(f"Phase steps: press={press_steps}, slide={slide_steps}, hold={hold_steps} (total={total_steps})")
    print(f"FPS: {args.fps}")
    if args.save_dir:
        print(f"Saving frames to: {args.save_dir}")
        print("Run manifest: run_manifest.json (params + frame→phase mapping)")
        print("Metrics: metrics.csv / metrics.json")
        if args.export_intermediate:
            print(f"Intermediate: enabled (every={int(args.export_intermediate_every)}) -> intermediate/frame_XXXX.npz")
    print()

    run_context = {
        "args": vars(args),
        "resolved": {
            "square_indenter_size_mm": float(square_d_mm) if square_d_mm is not None else None,
            "indenter_stl": stl_stats,
            "fem_indenter_geom": fem_indenter_geom,
            "indenter": {
                "mpm": {
                    "type": str(args.indenter_type),
                    "size_mm": mpm_indenter_size,
                },
                "fem": {
                    "geom": str(fem_indenter_geom),
                    "face": str(args.fem_indenter_face),
                    "contact_face_key": fem_contact_face_key,
                    "contact_face_size_mm": fem_contact_face_size_mm,
                },
            },
            "conventions": {
                "mpm_height_field_flip_x": True,
                "mpm_uv_disp_flip_x": True,
                "mpm_uv_disp_u_negate": False,
                "mpm_warp_flip_x": True,
                "mpm_warp_flip_y": True,
                "mpm_overlay_flip_x_mm": True,
                "mpm_height_fill_holes": bool(SCENE_PARAMS.get("mpm_height_fill_holes", False)),
                "mpm_height_fill_holes_iters": int(SCENE_PARAMS.get("mpm_height_fill_holes_iters", 0)),
                "mpm_height_smooth": bool(SCENE_PARAMS.get("mpm_height_smooth", True)),
                "mpm_height_smooth_iters": int(SCENE_PARAMS.get("mpm_height_smooth_iters", 0)),
                "mpm_height_reference_edge": bool(SCENE_PARAMS.get("mpm_height_reference_edge", True)),
                "mpm_height_clamp_indenter": bool(SCENE_PARAMS.get("mpm_height_clamp_indenter", True)),
                "mpm_height_clip_outliers": bool(SCENE_PARAMS.get("mpm_height_clip_outliers", False)),
                "mpm_height_clip_outliers_min_mm": float(SCENE_PARAMS.get("mpm_height_clip_outliers_min_mm", 0.0)),
            },
            "friction": {
                "fem_fric_coef": float(fem_fric),
                "mpm_mu_s": float(mpm_mu_s),
                "mpm_mu_k": float(mpm_mu_k),
                "aligned": bool(aligned_fric),
            },
            "contact": {
                "mpm_contact_stiffness_normal": float(k_n),
                "mpm_contact_stiffness_tangent": float(k_t),
            },
            "render": {
                "mpm_marker": str(args.mpm_marker),
                "fem_marker": str(args.fem_marker),
                "mpm_depth_tint": bool(str(args.mpm_depth_tint).lower().strip() != "off"),
            },
            "scale": {
                "gel_size_mm": [float(gel_w_mm), float(gel_h_mm)],
                "cam_view_mm": [float(cam_w_mm), float(cam_h_mm)],
                "delta_mm": [float(dw_mm), float(dh_mm)],
                "consistent": bool(scale_consistent),
                "tolerance_mm": float(tol_mm),
            },
            "export": {
                "intermediate": bool(args.export_intermediate),
                "intermediate_every": int(args.export_intermediate_every),
            },
        },
    }

    if args.save_dir:
        preflight_reason = None
        if not HAS_EZGL:
            preflight_reason = "ezgl not available"
        elif not HAS_TAICHI:
            preflight_reason = "taichi not available (mpm disabled)"
        _write_preflight_run_manifest(
            Path(args.save_dir),
            record_interval=int(args.record_interval),
            total_frames=int(expected_frames),
            run_context=run_context,
            reason=preflight_reason,
        )

    # Check dependencies
    if not HAS_EZGL:
        print("ERROR: ezgl not available, cannot run visualization")
        return 1

    if not HAS_TAICHI:
        print("WARNING: Taichi not available, MPM will be disabled")

    interactive = bool(args.interactive) or not bool(args.save_dir)

    # Run comparison
    engine = RGBComparisonEngine(
        fem_file=args.fem_file,
        object_file=args.object_file,
        mode=args.mode,
        visible=True,
        save_dir=args.save_dir,
        fem_indenter_face=args.fem_indenter_face,
        fem_indenter_geom=fem_indenter_geom,
    )
    engine.mpm_marker_mode = args.mpm_marker
    engine.mpm_depth_tint = (str(args.mpm_depth_tint).lower().strip() != "off")
    engine.set_fem_show_marker(str(args.fem_marker).lower().strip() != "off")
    engine.mpm_show_indenter = args.mpm_show_indenter
    engine.mpm_debug_overlay = args.mpm_debug_overlay
    engine.export_intermediate = bool(args.export_intermediate)
    engine.export_intermediate_every = int(args.export_intermediate_every)
    if square_d_mm is not None:
        engine.indenter_square_size_mm = float(square_d_mm)
    engine.run_context = run_context
    engine.run_comparison(
        fps=args.fps,
        record_interval=int(args.record_interval),
        interactive=interactive,
    )

    return 0


if __name__ == '__main__':
    exit(main())
