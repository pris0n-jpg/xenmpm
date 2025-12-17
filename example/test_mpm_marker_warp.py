"""
轻量回归：验证 MPM marker warp 能体现非均匀形变（拉伸/压缩/剪切），而不仅是整体平移。

运行：
  python example/test_mpm_marker_warp.py
"""
import numpy as np

import sys
from pathlib import Path

# Allow importing sibling script without packaging `example/` as a module.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mpm_fem_rgb_compare import warp_marker_texture  # noqa: E402


def _make_stripe_texture(h: int, w: int, period: int = 16) -> np.ndarray:
    # 使用确定性噪声纹理，避免周期纹理在平移下“差异不单调”的误判。
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_zero_uv_is_identity():
    base = _make_stripe_texture(128, 128)
    uv = np.zeros((20, 14, 2), dtype=np.float32)
    warped = warp_marker_texture(base, uv, gel_size_mm=(17.3, 29.15), flip_x=False, flip_y=False)
    assert np.array_equal(base, warped), "zero uv should keep texture unchanged"


def test_nonuniform_uv_changes_right_side_more():
    base = _make_stripe_texture(128, 128)

    # Create a non-uniform displacement field: u increases with x (simulates stretch/shear)
    ny, nx = 20, 14
    xv = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    u_mm = (xv[None, :] * 1.2).repeat(ny, axis=0)  # right side moves more (+x), left side moves less
    v_mm = np.zeros((ny, nx), dtype=np.float32)
    uv = np.stack([u_mm, v_mm], axis=-1)

    warped = warp_marker_texture(base, uv, gel_size_mm=(17.3, 29.15), flip_x=False, flip_y=False)

    # Non-uniform warp should not be identical
    assert not np.array_equal(base, warped), "non-uniform uv should change texture"

    # Right side should change more than left side (magnitude monotonic with |u|)
    diff = np.mean(np.abs(warped.astype(np.int16) - base.astype(np.int16)), axis=2)
    left = float(diff[:, :diff.shape[1] // 3].mean())
    right = float(diff[:, -diff.shape[1] // 3:].mean())
    assert right > left * 1.15, f"expected right change > left change (left={left:.3f}, right={right:.3f})"


if __name__ == "__main__":
    test_zero_uv_is_identity()
    test_nonuniform_uv_changes_right_side_more()
    print("OK: marker warp regression checks passed.")
