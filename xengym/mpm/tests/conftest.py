"""
Pytest Configuration and Fixtures for MPM Tests

Provides shared fixtures for Taichi initialization and common test utilities.

Note: Taichi initialization is NOT autouse - tests that need Taichi should
explicitly use the `init_taichi` fixture or use `pytest.importorskip("taichi")`.
"""
import pytest
import numpy as np
import os
import csv
from typing import Tuple, Dict, Optional, Any
from datetime import datetime

# Try to import taichi - tests will be skipped if not available
try:
    import taichi as ti
    _HAS_TAICHI = True
except ImportError:
    _HAS_TAICHI = False
    ti = None


# =============================================================================
# Numerical gradient verification parameters (centralized for maintainability)
# Tier thresholds aligned with OpenSpec refine-mpm-ad-precision verification-suite
# =============================================================================
GRADIENT_EPS_SMALL = 1e-4      # For position/velocity perturbations
GRADIENT_EPS_MEDIUM = 0.1      # For material parameter perturbations
GRADIENT_EPS_LARGE = 1.0       # For stress gradient chain tests

# Gradient accuracy tiers (see verification-suite spec for definitions)
# Tier A (constitutive/stress level): rel_error ≤ 0.01, cosine_sim ≥ 0.99
GRADIENT_RTOL_STRICT = 0.01    # Tier A: 1% relative tolerance

# Tier B (small MPM toy scenes): rel_error ≤ 0.05, cosine_sim ≥ 0.95
GRADIENT_RTOL_NORMAL = 0.05    # Tier B: 5% relative tolerance

# Tier C (high-deformation end-to-end): rel_error ≤ 0.50, cosine_sim ≥ 0.80
GRADIENT_RTOL_LOOSE = 0.50     # Tier C: 50% relative tolerance

# Intermediate tolerance for tests that fall between Tier B and C
GRADIENT_RTOL_RELAXED = 0.20   # 20% relative tolerance (transitional)


@pytest.fixture(scope="session")
def init_taichi():
    """Initialize Taichi once per test session.

    Uses CPU backend with float32 for reproducibility.
    Automatically resets Taichi after all tests complete.

    NOTE: This fixture is NOT autouse. Tests that need Taichi should either:
    1. Use `taichi = pytest.importorskip("taichi")` at module level, OR
    2. Explicitly request this fixture as a parameter

    This allows tests like test_gradient_mode.py to run without Taichi.
    """
    if not _HAS_TAICHI:
        pytest.skip("Taichi not available")
        return

    ti.init(arch=ti.cpu, default_fp=ti.f32, debug=False)
    yield
    ti.reset()


@pytest.fixture
def default_config():
    """Create a default MPM configuration for testing.

    Returns:
        MPMConfig with reasonable defaults for small-scale tests.
    """
    if not _HAS_TAICHI:
        pytest.skip("Taichi not available")

    from xengym.mpm.config import MPMConfig

    return MPMConfig(
        grid=dict(
            grid_size=[32, 32, 32],
            dx=0.01
        ),
        time=dict(
            dt=1e-4,
            num_steps=100
        ),
        material=dict(
            density=1000.0,
            ogden=dict(
                mu=[1000.0],
                alpha=[2.0],
                kappa=10000.0
            ),
            maxwell_branches=[],
            enable_bulk_viscosity=False,
            bulk_viscosity=0.0
        ),
        contact=dict(
            enable_contact=False
        )
    )


@pytest.fixture
def simple_particle_positions() -> np.ndarray:
    """Create simple particle positions for testing.

    Returns:
        (N, 3) array of particle positions in a small cube.
    """
    spacing = 0.02
    positions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                positions.append([
                    0.15 + i * spacing,
                    0.15 + j * spacing,
                    0.15 + k * spacing
                ])
    return np.array(positions, dtype=np.float32)


@pytest.fixture
def random_deformation_gradient() -> np.ndarray:
    """Create a random valid deformation gradient for testing.

    Returns:
        (3, 3) array with det(F) > 0 (physically valid).
    """
    np.random.seed(42)
    F = np.eye(3, dtype=np.float32) + 0.1 * np.random.randn(3, 3).astype(np.float32)
    if np.linalg.det(F) < 0:
        F[:, 0] *= -1
    return F


@pytest.fixture
def identity_deformation_gradient() -> np.ndarray:
    """Create identity deformation gradient (undeformed state)."""
    return np.eye(3, dtype=np.float32)


@pytest.fixture
def config_with_maxwell():
    """Create MPM config with Maxwell viscoelasticity enabled."""
    if not _HAS_TAICHI:
        pytest.skip("Taichi not available")

    from xengym.mpm.config import MPMConfig

    return MPMConfig(
        grid=dict(grid_size=[32, 32, 32], dx=0.01),
        time=dict(dt=1e-4, num_steps=100),
        material=dict(
            density=1000.0,
            ogden=dict(mu=[1000.0], alpha=[2.0], kappa=10000.0),
            maxwell_branches=[dict(G=500.0, tau=0.01)],
            enable_bulk_viscosity=False,
            bulk_viscosity=0.0
        ),
        contact=dict(enable_contact=False)
    )


@pytest.fixture
def config_with_bulk_viscosity():
    """Create MPM config with bulk viscosity enabled."""
    if not _HAS_TAICHI:
        pytest.skip("Taichi not available")

    from xengym.mpm.config import MPMConfig

    return MPMConfig(
        grid=dict(grid_size=[32, 32, 32], dx=0.01),
        time=dict(dt=1e-4, num_steps=100),
        material=dict(
            density=1000.0,
            ogden=dict(mu=[1000.0], alpha=[2.0], kappa=10000.0),
            maxwell_branches=[],
            enable_bulk_viscosity=True,
            bulk_viscosity=100.0
        ),
        contact=dict(enable_contact=False)
    )


def assert_gradient_close(
    analytic: float,
    numerical: float,
    rtol: float = GRADIENT_RTOL_NORMAL,
    atol: float = 1e-6,
    name: str = "gradient"
) -> None:
    """Assert that analytic and numerical gradients are close.

    Args:
        analytic: Analytically computed gradient
        numerical: Numerically computed gradient (finite difference)
        rtol: Relative tolerance (default 10%)
        atol: Absolute tolerance
        name: Name for error message

    Raises:
        AssertionError: If gradients differ by more than tolerance
    """
    abs_diff = abs(analytic - numerical)
    rel_diff = abs_diff / max(abs(numerical), atol)

    if rel_diff > rtol and abs_diff > atol:
        raise AssertionError(
            f"{name} mismatch: analytic={analytic:.6e}, numerical={numerical:.6e}, "
            f"rel_diff={rel_diff:.2%}, abs_diff={abs_diff:.6e}"
        )


def compute_numerical_gradient(
    loss_fn,
    param_getter,
    param_setter,
    eps: float = GRADIENT_EPS_SMALL
) -> float:
    """Compute numerical gradient using central difference.

    Args:
        loss_fn: Function that returns loss value
        param_getter: Function that returns current parameter value
        param_setter: Function that sets parameter value
        eps: Finite difference step size

    Returns:
        Numerical gradient estimate
    """
    original = param_getter()

    param_setter(original + eps)
    loss_plus = loss_fn()

    param_setter(original - eps)
    loss_minus = loss_fn()

    param_setter(original)

    return (loss_plus - loss_minus) / (2 * eps)


def create_target_positions(positions: np.ndarray, z_offset: float = -0.01,
                           x_weighted: bool = False, weight: float = 0.1) -> np.ndarray:
    """Create target positions for gradient tests (reduces test code duplication).

    Args:
        positions: Original particle positions (N, 3)
        z_offset: Base z-direction offset
        x_weighted: If True, add x-position weighted offset
        weight: Weight for x-position offset

    Returns:
        Target positions array (N, 3)
    """
    target = positions.copy()
    target[:, 2] += z_offset

    if x_weighted:
        center_x = np.mean(positions[:, 0])
        for i in range(len(positions)):
            x_offset = positions[i, 0] - center_x
            target[i, 2] += weight * x_offset

    return target


# =============================================================================
# Gradient Quality Metrics Reporting
# =============================================================================
class GradientMetricsReporter:
    """Utility for recording and exporting gradient quality metrics.

    Provides structured output of gradient verification results for offline
    analysis, supporting the Design Decision 4 in refine-mpm-ad-precision spec.

    Usage:
        reporter = GradientMetricsReporter()
        reporter.record("ogden_mu", analytic=0.1, numerical=0.11, tier="A")
        reporter.save_csv("gradient_metrics.csv")
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the reporter.

        Args:
            output_dir: Directory for output files. Defaults to 'output/gradient_metrics'.
        """
        self.records: list[Dict[str, Any]] = []
        self.output_dir = output_dir or os.path.join("output", "gradient_metrics")

    def record(
        self,
        param_name: str,
        analytic: float,
        numerical: float,
        tier: str = "B",
        test_name: str = "",
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Record a gradient comparison result.

        Args:
            param_name: Name of the parameter being tested
            analytic: Analytically computed gradient
            numerical: Numerically computed gradient
            tier: Expected accuracy tier (A, B, or C)
            test_name: Name of the test
            extra: Additional metadata

        Returns:
            Dict with computed metrics
        """
        abs_error = abs(analytic - numerical)
        rel_error = abs_error / max(abs(numerical), 1e-15)
        cos_sim = 1.0 if (analytic * numerical > 0) else -1.0

        # Determine tier thresholds
        tier_thresholds = {
            "A": (GRADIENT_RTOL_STRICT, 0.99),
            "B": (GRADIENT_RTOL_NORMAL, 0.95),
            "C": (GRADIENT_RTOL_LOOSE, 0.80),
        }
        rtol_limit, cos_limit = tier_thresholds.get(tier, (0.1, 0.9))
        passed = rel_error <= rtol_limit and cos_sim >= cos_limit

        record = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "param_name": param_name,
            "tier": tier,
            "analytic": analytic,
            "numerical": numerical,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "cos_sim": cos_sim,
            "rtol_limit": rtol_limit,
            "passed": passed,
        }
        if extra:
            record.update(extra)

        self.records.append(record)
        return record

    def save_csv(self, filename: Optional[str] = None) -> str:
        """Save all records to a CSV file.

        Args:
            filename: Output filename. Defaults to timestamped name.

        Returns:
            Full path to the saved file.
        """
        if not self.records:
            return ""

        os.makedirs(self.output_dir, exist_ok=True)
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gradient_metrics_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        fieldnames = list(self.records[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        return filepath

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of all recorded metrics.

        Returns:
            Dict with pass/fail counts per tier and overall statistics.
        """
        if not self.records:
            return {"total": 0}

        by_tier = {"A": [], "B": [], "C": []}
        for r in self.records:
            tier = r.get("tier", "B")
            if tier in by_tier:
                by_tier[tier].append(r)

        summary = {
            "total": len(self.records),
            "passed": sum(1 for r in self.records if r["passed"]),
            "failed": sum(1 for r in self.records if not r["passed"]),
        }

        for tier, records in by_tier.items():
            if records:
                summary[f"tier_{tier}_total"] = len(records)
                summary[f"tier_{tier}_passed"] = sum(1 for r in records if r["passed"])
                avg_rel = np.mean([r["rel_error"] for r in records])
                summary[f"tier_{tier}_avg_rel_error"] = avg_rel

        return summary

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        s = self.summary()
        if s["total"] == 0:
            print("No gradient metrics recorded.")
            return

        print("\n" + "=" * 60)
        print("GRADIENT QUALITY METRICS SUMMARY")
        print("=" * 60)
        print(f"Total tests: {s['total']}")
        print(f"Passed: {s['passed']}, Failed: {s['failed']}")

        for tier in ["A", "B", "C"]:
            key = f"tier_{tier}_total"
            if key in s:
                passed = s.get(f"tier_{tier}_passed", 0)
                avg = s.get(f"tier_{tier}_avg_rel_error", 0.0)
                print(f"\nTier {tier}: {passed}/{s[key]} passed, avg rel_error={avg:.4f}")

        print("=" * 60)


@pytest.fixture
def gradient_reporter():
    """Fixture providing a fresh GradientMetricsReporter instance."""
    return GradientMetricsReporter()
