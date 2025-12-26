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
    python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --save-dir output/rgb_compare/baseline

Output (when --save-dir is set):
    - fem_XXXX.png / mpm_XXXX.png
    - run_manifest.json (resolved params + frame->phase mapping)

Requirements:
    - xengym conda environment with Taichi installed
    - FEM data file (default: assets/data/fem_data_gel_2035.npz)
"""
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import sys
import ast
import re
import datetime

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
        DepthCamera,
        RGBCamera,
        PointLight,
        LineLight,
        GLAxisItem,
        Texture2D,
        Material,
    )
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

    # Trajectory parameters
    'press_depth_mm': 2.0,              # target indentation depth (increased for better friction coupling)
    'slide_distance_mm': 3.0,           # tangential travel (x direction)
    'press_steps': 150,                 # steps to reach press depth (increased for deeper press)
    'slide_steps': 240,                 # steps for sliding phase
    'hold_steps': 40,                   # steps to hold at end

    # MPM simulation parameters
    'mpm_dt': 2e-5,                     # Reduced for stability with higher stiffness
    'mpm_grid_dx_mm': 0.4,              # grid spacing in mm
    'mpm_particles_per_cell': 2,        # particles per cell per dimension

    # Material (soft gel)
    'density': 1000.0,                  # kg/m³
    'ogden_mu': [2500.0],               # Pa
    'ogden_alpha': [2.0],
    'ogden_kappa': 25000.0,             # Pa

    # Indenter (sphere)
    'indenter_radius_mm': 4.0,
    'indenter_start_gap_mm': 0.5,       # initial clearance above gel
    'indenter_type': 'box',             # 'sphere' or 'box' (flat bottom)
    'indenter_half_extents_mm': None,   # Optional (x,y,z) for box mode; overrides indenter_radius_mm

    # Depth camera settings (for FEM path)
    'depth_img_size': (100, 175),       # matches demo_simple_sensor
    'cam_view_width_m': 0.0194,         # 19.4 mm
    'cam_view_height_m': 0.0308,        # 30.8 mm

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


# ==============================================================================
# FEM RGB Renderer (Reuses VecTouchSim)
# ==============================================================================
class FEMRGBRenderer:
    """FEM sensor RGB rendering using existing VecTouchSim pipeline"""

    def __init__(self, fem_file: str, object_file: Optional[str] = None, visible: bool = False):
        if not HAS_EZGL:
            raise RuntimeError("ezgl not available for FEM rendering")

        self.fem_file = Path(fem_file)
        self.object_file = object_file
        self.visible = visible

        # Create depth scene for object rendering
        self.depth_scene = self._create_depth_scene()

        # Create VecTouchSim for FEM sensor rendering
        self.sensor_sim = VecTouchSim(
            depth_size=SCENE_PARAMS['depth_img_size'],
            fem_file=str(fem_file),
            visible=visible,
            title="FEM Sensor"
        )

        # Object pose (indenter position)
        self._object_y = 0.02  # initial y position (m)
        self._object_z = 0.0   # will be set by trajectory

    def _create_depth_scene(self) -> 'DepthRenderScene':
        """Create depth rendering scene for the indenter"""
        return DepthRenderScene(
            object_file=self.object_file,
            visible=self.visible
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

    def __init__(self, object_file: Optional[str] = None, visible: bool = True):
        # Note: visible=True is required for proper OpenGL context initialization
        super().__init__(600, 400, visible=visible, title="Depth Render")
        self.cameraLookAt((0.1, 0.1, 0.1), (0, 0, 0), (0, 1, 0))

        # Camera view parameters
        self.cam_view_width = SCENE_PARAMS['cam_view_width_m']
        self.cam_view_height = SCENE_PARAMS['cam_view_height_m']

        # Create indenter object (sphere by default)
        if object_file and Path(object_file).exists():
            self.object = GLModelItem(
                object_file, glOptions="translucent",
                lights=PointLight()
            )
        else:
            # Use built-in sphere
            self.object = GLModelItem(
                str(ASSET_DIR / "obj/circle_r4.STL"),
                glOptions="translucent",
                lights=PointLight()
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
        self._object_fixed_tf = Matrix4x4()     # local->parent 固定变换（轴对齐/朝向/模型原点偏移等）
        self._object_pose_raw = Matrix4x4()     # 每帧输入 pose（仅平移/旋转控制量）
        self._object_pose = Matrix4x4()         # 最终用于渲染+FEM 的世界变换（raw * fixed）

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
        initial_top_z_m: float
    ) -> np.ndarray:
        """
        Extract top-surface height field from MPM particles

        Args:
            positions_m: Particle positions (N, 3) in meters
            initial_top_z_m: Initial top surface z coordinate in meters

        Returns:
            height_field_mm: (n_row, n_col) array, negative = indentation
        """
        n_row, n_col = self.grid_shape
        gel_w_mm, gel_h_mm = self.gel_size_mm

        if not self._is_configured:
            self.configure_from_initial_positions(positions_m, initial_top_z_m)

        pos_mm = positions_m * 1000.0
        z_top_init_mm = initial_top_z_m * 1000.0

        if self._surface_indices is not None and len(self._surface_indices) > 0:
            pos_mm = pos_mm[self._surface_indices]

        # Map to sensor grid using cached reference frame:
        # x ∈ [-gel_w/2, gel_w/2], y ∈ [0, gel_h]
        pos_sensor = pos_mm.copy()
        pos_sensor[:, 0] -= self._x_center_mm
        pos_sensor[:, 1] -= self._y_min_mm

        # Grid cell dimensions
        cell_w = gel_w_mm / n_col
        cell_h = gel_h_mm / n_row

        # Initialize height field with -inf so we can take max z_disp per cell (top surface)
        height_field = np.full((n_row, n_col), -np.inf, dtype=np.float32)

        for particle_idx in range(len(pos_sensor)):
            x_mm, y_mm, z_mm = pos_sensor[particle_idx]
            col = int((x_mm + gel_w_mm / 2.0) / cell_w)
            row = int(y_mm / cell_h)
            if 0 <= row < n_row and 0 <= col < n_col:
                z_disp = float(z_mm - z_top_init_mm)  # <= 0
                if z_disp > height_field[row, col]:
                    height_field[row, col] = z_disp

        empty_mask = ~np.isfinite(height_field)
        height_field[empty_mask] = 0.0

        # CRITICAL: Use EDGE regions as fixed spatial reference for better contrast
        # This preserves slide motion (unlike global percentile which cancels it)
        edge_margin = max(3, n_row // 20)  # ~5% margin or at least 3 rows/cols
        edge_values = []
        # Top and bottom edges
        edge_values.append(height_field[:edge_margin, :].ravel())
        edge_values.append(height_field[-edge_margin:, :].ravel())
        # Left and right edges (excluding corners already counted)
        edge_values.append(height_field[edge_margin:-edge_margin, :edge_margin].ravel())
        edge_values.append(height_field[edge_margin:-edge_margin, -edge_margin:].ravel())
        edge_values = np.concatenate(edge_values)
        edge_valid = edge_values[np.isfinite(edge_values) & (edge_values > -10)]  # filter outliers
        if len(edge_valid) > 0:
            reference_z = float(np.median(edge_valid))  # Median is robust to outliers
            height_field = height_field - reference_z  # Now center region depression is negative

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
        smooth_uv: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时提取顶面高度场与面内位移场（u,v）。

        Returns:
            height_field_mm: (Ny,Nx), <= 0 表示压入
            uv_disp_mm: (Ny,Nx,2), 单位 mm
        """
        height_field = self.extract_height_field(positions_m, initial_top_z_m)

        if not self._is_configured:
            self.configure_from_initial_positions(positions_m, initial_top_z_m)
        if self._initial_positions_m is None:
            uv = np.zeros((self.grid_shape[0], self.grid_shape[1], 2), dtype=np.float32)
            return height_field, uv

        n_row, n_col = self.grid_shape
        gel_w_mm, gel_h_mm = self.gel_size_mm

        pos_mm_all = positions_m * 1000.0
        init_mm_all = self._initial_positions_m * 1000.0

        if self._surface_indices is not None and len(self._surface_indices) > 0:
            idx = self._surface_indices
            pos_mm_all = pos_mm_all[idx]
            init_mm_all = init_mm_all[idx]

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

        for particle_idx in range(len(pos_sensor)):
            x_mm, y_mm, _ = pos_sensor[particle_idx]
            col = int((x_mm + gel_w_mm / 2.0) / cell_w)
            row = int(y_mm / cell_h)
            if 0 <= row < n_row and 0 <= col < n_col:
                disp_xy = pos_sensor[particle_idx, :2] - init_sensor[particle_idx, :2]
                uv_sum[row, col, 0] += float(disp_xy[0])
                uv_sum[row, col, 1] += float(disp_xy[1])
                uv_cnt[row, col] += 1

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
        self.scene.set_height_field(height_field_mm)
        return self.scene.get_image()

    def get_diff_image(self, height_field_mm: np.ndarray) -> np.ndarray:
        """Get diff image relative to flat reference"""
        self.scene.set_height_field(height_field_mm)
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

    def set_uv_displacement(self, uv_disp_mm: Optional[np.ndarray]) -> None:
        """设置当前帧的面内位移场 (Ny,Nx,2)，单位 mm。"""
        if uv_disp_mm is None:
            self._uv_disp_mm = None
        else:
            # CRITICAL: Apply same horizontal flip as height_field to match mesh x_range convention
            # height_field is flipped with [:, ::-1], so UV must be too
            # Additionally, flip u-component sign since x-direction is reversed
            uv_flipped = uv_disp_mm[:, ::-1].copy()
            uv_flipped[..., 0] = -uv_flipped[..., 0]  # negate u (x displacement)
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
            height_field_mm = self._box_blur_2d(height_field_mm, iterations=2)

        # CRITICAL: Flip horizontally to match mesh x_range convention
        # Height field: col=0 is x=-gel_w/2 (left)
        # Mesh x_range: (gel_w/2, -gel_w/2) means col=0 is x=+gel_w/2 (right)
        height_field_mm = height_field_mm[:, ::-1]

        # Ensure negative values for indentation (SensorScene convention)
        depth = np.minimum(height_field_mm, 0)

        # Debug: verify depth values being sent to mesh
        neg_count = (depth < -0.01).sum()
        if neg_count > 0 and SCENE_PARAMS.get('debug_verbose', False):
            print(f"[MESH UPDATE] depth: min={depth.min():.2f}mm, neg_cells(>0.01mm)={neg_count}")

        self.surf_mesh.setData(depth, smooth)
        self._update_marker_texture()
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
            # CRITICAL: Negate x_mm to match height_field horizontal flip (mesh x_range convention)
            col = int((-x_mm + self.gel_width_mm / 2.0) / cell_w)
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
        grid_extent = [
            int(np.ceil(gel_w_m / dx)) + 8,
            int(np.ceil(gel_h_m / dx)) + 8,
            int(np.ceil(gel_t_m * 2 / dx)) + 8,
        ]

        # Create particles
        n_particles = self._create_particles(gel_w_m, gel_h_m, gel_t_m, dx)

        # Indenter setup
        indenter_r = SCENE_PARAMS['indenter_radius_mm'] * 1e-3
        indenter_gap = SCENE_PARAMS['indenter_start_gap_mm'] * 1e-3
        indenter_type = SCENE_PARAMS.get('indenter_type', 'box')

        # Determine the effective z half-height for indenter placement
        # For sphere: use radius; for box: use half_extents[2]
        if indenter_type == 'sphere':
            indenter_z_half = indenter_r
            half_extents = (indenter_r, 0, 0)  # sphere uses radius in first component
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
        else:
            obstacles.append(SDFConfig(
                sdf_type="box",
                center=indenter_center,
                half_extents=half_extents
            ))

        print(f"[MPM] Indenter: type={indenter_type}, z_half={indenter_z_half*1000:.1f}mm, "
              f"center=({indenter_center[0]*1000:.1f}, {indenter_center[1]*1000:.1f}, {indenter_center[2]*1000:.1f})mm")

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
                enable_bulk_viscosity=False
            ),
            contact=ContactConfig(
                enable_contact=True,
                contact_stiffness_normal=8e2,  # Balanced for stability and response
                contact_stiffness_tangent=4e2,  # Good tangential coupling
                mu_s=2.0,  # High static friction to drag material with indenter
                mu_k=1.5,  # High kinetic friction
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
        # MPM solver has sticky boundary at I[d] < 3, so need padding > 3*dx
        padding = dx * 6
        min_pos = positions.min(axis=0)
        positions += (padding - min_pos)

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
        save_dir: Optional[str] = None
    ):
        self.mode = mode
        self.visible = visible
        self.save_dir = Path(save_dir) if save_dir else None
        self.run_context: Dict[str, object] = {}

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize renderers
        self.fem_renderer = None
        self.mpm_renderer = None
        self.mpm_sim = None

        if HAS_EZGL:
            self.fem_renderer = FEMRGBRenderer(fem_file, object_file=object_file, visible=False)
            self.mpm_renderer = MPMHeightFieldRenderer(visible=False)

        if HAS_TAICHI:
            self.mpm_sim = MPMSimulationAdapter()
        self.mpm_marker_mode = "static"
        self.mpm_show_indenter = False
        self.mpm_debug_overlay = "off"
        self.indenter_square_size_mm = _infer_square_size_mm_from_stl_path(object_file)

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

        manifest = {
            "created_at": datetime.datetime.now().astimezone().isoformat(),
            "argv": list(sys.argv),
            "run_context": dict(self.run_context),
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
            },
            "outputs": {
                "frames_glob": {
                    "fem": "fem_*.png",
                    "mpm": "mpm_*.png",
                },
                "run_manifest": "run_manifest.json",
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

    def run_comparison(self, fps: int = 30, record_interval: int = 5):
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

        # Create UI
        self._create_ui(positions_history, fps)

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
                fem_rgb = self.fem_renderer.step()
                if self.mode == 'diff':
                    fem_rgb = self.fem_renderer.get_diff_image()
                if fem_view[0] is not None:
                    fem_view[0].setData(fem_rgb)

            # MPM rendering
            mpm_rgb = None
            if self.mpm_renderer and mpm_positions and frame_idx[0] < len(mpm_positions):
                pos = mpm_positions[frame_idx[0]]
                height_field, uv_disp = self.mpm_renderer.extract_surface_fields(pos, self.mpm_sim.initial_top_z_m)
                self.mpm_renderer.scene.set_uv_displacement(uv_disp)
                if self.mode == 'diff':
                    mpm_rgb = self.mpm_renderer.get_diff_image(height_field)
                else:
                    mpm_rgb = self.mpm_renderer.render(height_field)
                if mpm_view[0] is not None:
                    mpm_view[0].setData(mpm_rgb)

                if self.mpm_show_indenter and self.mpm_sim and self.mpm_sim.frame_controls and frame_idx[0] < len(self.mpm_sim.frame_controls):
                    _, slide_amount_m = self.mpm_sim.frame_controls[frame_idx[0]]
                    # y 方向用胶体中心做可视化对齐（square 压头在 y 方向不移动）
                    self.mpm_renderer.scene.set_indenter_center(float(slide_amount_m) * 1000.0, SCENE_PARAMS['gel_size_mm'][1] / 2.0)

            # Save frame if requested
            if self.save_dir and HAS_CV2 and fem_rgb is not None:
                cv2.imwrite(
                    str(self.save_dir / f"fem_{frame_idx[0]:04d}.png"),
                    cv2.cvtColor(fem_rgb, cv2.COLOR_RGB2BGR)
                )
                if mpm_rgb is not None:
                    cv2.imwrite(
                        str(self.save_dir / f"mpm_{frame_idx[0]:04d}.png"),
                        cv2.cvtColor(mpm_rgb, cv2.COLOR_RGB2BGR)
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
        '--indenter-type', type=str, choices=['sphere', 'box'], default='box',
        help='MPM indenter type: sphere (curved) or box (flat bottom, matches cylinder STL)'
    )
    parser.add_argument(
        '--mpm-marker', type=str, choices=['off', 'static', 'warp'], default='static',
        help='MPM marker rendering: off|static|warp (warp reflects stretch/shear from tangential displacement)'
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
        '--debug', action='store_true', default=False,
        help='Enable verbose per-frame debug logging'
    )

    args = parser.parse_args()

    # Update scene params from args
    SCENE_PARAMS['press_depth_mm'] = args.press_mm
    SCENE_PARAMS['slide_distance_mm'] = args.slide_mm
    SCENE_PARAMS['indenter_type'] = args.indenter_type
    SCENE_PARAMS['debug_verbose'] = args.debug

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
    print(f"FEM indenter: circle_r4.STL (cylinder, ~4mm radius)")        
    if args.object_file:
        print(f"Object file: {args.object_file}")
    print(f"MPM marker: {args.mpm_marker}")
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
    print()

    # Check dependencies
    if not HAS_EZGL:
        print("ERROR: ezgl not available, cannot run visualization")
        return 1

    if not HAS_TAICHI:
        print("WARNING: Taichi not available, MPM will be disabled")

    # Run comparison
    engine = RGBComparisonEngine(
        fem_file=args.fem_file,
        object_file=args.object_file,
        mode=args.mode,
        visible=True,
        save_dir=args.save_dir
    )
    engine.mpm_marker_mode = args.mpm_marker
    engine.mpm_show_indenter = args.mpm_show_indenter
    engine.mpm_debug_overlay = args.mpm_debug_overlay
    if square_d_mm is not None:
        engine.indenter_square_size_mm = float(square_d_mm)
    engine.run_context = {
        "args": vars(args),
        "resolved": {
            "square_indenter_size_mm": float(square_d_mm) if square_d_mm is not None else None,
        },
    }
    engine.run_comparison(fps=args.fps, record_interval=int(args.record_interval))

    return 0


if __name__ == '__main__':
    exit(main())
