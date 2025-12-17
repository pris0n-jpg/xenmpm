"""
Unit tests for gradient mode configuration and validation.

Tests:
1. validate_gradient_mode() strict=True behavior (raises ValueError)
2. validate_gradient_mode() strict=False behavior (returns False with warning)
3. validate_gradient_mode() with valid config (returns True)
4. configure_gradient_mode() analytical mode warning
5. Experimental P_total mode configuration and scale guards (FR-4)
6. Mode switching and state management

Run with:
    python xengym/mpm/gradient_mode_tests.py

Note: This test uses mock implementations to avoid Taichi dependency.
      File renamed from test_gradient_mode.py to avoid pytest auto-collection
      (pytest would try to import xengym.mpm which requires taichi).
"""

import sys
import os
import warnings
from contextlib import contextmanager

try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    pytest = None
    _HAS_PYTEST = False


# Fallback for pytest.raises when pytest is not available
@contextmanager
def _raises_fallback(exception_type):
    """Simple fallback for pytest.raises when pytest is not installed."""
    class ExcInfo:
        def __init__(self):
            self.value = None

    exc_info = ExcInfo()
    try:
        yield exc_info
        raise AssertionError(f"Expected {exception_type.__name__} was not raised")
    except exception_type as e:
        exc_info.value = e


def raises(exception_type):
    """Use pytest.raises if available, otherwise use fallback."""
    if _HAS_PYTEST:
        return pytest.raises(exception_type)
    else:
        return _raises_fallback(exception_type)

from dataclasses import dataclass, field
from typing import List

# ============================================
# Mock implementations (avoid Taichi dependency)
# ============================================

# Module-level configuration (mirrors constitutive_gradients.py)
_USE_NUMERICAL_G_F = True
_FINITE_DIFF_EPS = 1e-4
_J_CLAMP_MIN = 0.5
_J_CLAMP_MAX = 2.0

# Experimental P_total numerical mode (FR-4)
_EXPERIMENTAL_P_TOTAL_MODE = False
_P_TOTAL_MAX_PARTICLES = 5000
_P_TOTAL_MAX_STEPS = 500


def configure_gradient_mode(use_numerical: bool = True, eps: float = 1e-4,
                            experimental_p_total: bool = False,
                            max_particles: int = 5000, max_steps: int = 500):
    """
    Configure gradient computation mode. Call BEFORE Taichi compilation.
    (Mirror of constitutive_gradients.configure_gradient_mode)
    """
    global _USE_NUMERICAL_G_F, _FINITE_DIFF_EPS
    global _EXPERIMENTAL_P_TOTAL_MODE, _P_TOTAL_MAX_PARTICLES, _P_TOTAL_MAX_STEPS

    _USE_NUMERICAL_G_F = use_numerical
    _FINITE_DIFF_EPS = eps
    _EXPERIMENTAL_P_TOTAL_MODE = experimental_p_total
    _P_TOTAL_MAX_PARTICLES = max_particles
    _P_TOTAL_MAX_STEPS = max_steps

    if not use_numerical:
        warnings.warn(
            "Analytical g_F mode enabled. This is INCOMPLETE (missing F @ ∂S/∂F term) "
            "and should only be used for debugging/performance comparison. "
            "For production calibration, use numerical mode (use_numerical=True).",
            UserWarning
        )

    if experimental_p_total:
        warnings.warn(
            f"EXPERIMENTAL: P_total numerical gradient mode enabled. "
            f"Scale guards: max_particles={max_particles}, max_steps={max_steps}.",
            UserWarning
        )


def validate_gradient_mode(config, strict: bool = True,
                           n_particles: int = None, n_steps: int = None) -> bool:
    """
    Validate gradient configuration against MPM config.
    (Mirror of constitutive_gradients.validate_gradient_mode)
    """
    has_maxwell = len(config.material.maxwell_branches) > 0
    has_bulk_visc = config.material.enable_bulk_viscosity
    needs_p_total = has_maxwell or has_bulk_visc

    if needs_p_total:
        if _EXPERIMENTAL_P_TOTAL_MODE:
            # Check scale guards
            if n_particles is not None and n_particles > _P_TOTAL_MAX_PARTICLES:
                msg = (
                    f"EXPERIMENTAL P_total mode: particle count ({n_particles}) exceeds "
                    f"scale guard ({_P_TOTAL_MAX_PARTICLES})."
                )
                if strict:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg, RuntimeWarning)
                    return False

            if n_steps is not None and n_steps > _P_TOTAL_MAX_STEPS:
                msg = (
                    f"EXPERIMENTAL P_total mode: step count ({n_steps}) exceeds "
                    f"scale guard ({_P_TOTAL_MAX_STEPS})."
                )
                if strict:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg, RuntimeWarning)
                    return False

            # Experimental mode is enabled and within limits
            warnings.warn(
                f"Using EXPERIMENTAL P_total numerical mode for Maxwell/bulk viscosity gradients.",
                RuntimeWarning
            )
            return True
        else:
            # Standard mode: block Maxwell/bulk viscosity
            msg = (
                f"Manual adjoint gradient computation is INCOMPATIBLE with current config. "
                f"g_F only differentiates Ogden stress, but P_total includes: "
                f"Maxwell branches={len(config.material.maxwell_branches)}, "
                f"bulk_viscosity={has_bulk_visc}. "
                f"Gradients will be INCORRECT. "
                f"To use Manual Adjoint, disable Maxwell branches and bulk viscosity."
            )
            if strict:
                raise ValueError(msg)
            else:
                warnings.warn(msg, RuntimeWarning)
                return False

    return True


def is_experimental_mode_enabled() -> bool:
    """Check if experimental P_total mode is enabled."""
    return _EXPERIMENTAL_P_TOTAL_MODE


def get_scale_guards() -> tuple:
    """Get current scale guard limits (max_particles, max_steps)."""
    return (_P_TOTAL_MAX_PARTICLES, _P_TOTAL_MAX_STEPS)


# Mock config classes (to avoid Taichi initialization during tests)
@dataclass
class MockMaxwellBranch:
    G: float = 1000.0
    tau: float = 0.1


@dataclass
class MockMaterialConfig:
    maxwell_branches: List[MockMaxwellBranch] = field(default_factory=list)
    enable_bulk_viscosity: bool = False


@dataclass
class MockMPMConfig:
    material: MockMaterialConfig = field(default_factory=MockMaterialConfig)


class TestValidateGradientMode:
    """Tests for validate_gradient_mode function."""

    def test_valid_config_returns_true(self):
        """Valid config (no Maxwell, no bulk viscosity) should return True."""
        config = MockMPMConfig(
            material=MockMaterialConfig(
                maxwell_branches=[],
                enable_bulk_viscosity=False
            )
        )

        result = validate_gradient_mode(config, strict=True)
        assert result is True

    def test_maxwell_strict_raises_valueerror(self):
        """Config with Maxwell branches should raise ValueError in strict mode."""
        config = MockMPMConfig(
            material=MockMaterialConfig(
                maxwell_branches=[MockMaxwellBranch()],
                enable_bulk_viscosity=False
            )
        )

        with raises(ValueError) as excinfo:
            validate_gradient_mode(config, strict=True)

        assert "INCOMPATIBLE" in str(excinfo.value)
        assert "Maxwell branches=1" in str(excinfo.value)

    def test_bulk_viscosity_strict_raises_valueerror(self):
        """Config with bulk viscosity should raise ValueError in strict mode."""
        config = MockMPMConfig(
            material=MockMaterialConfig(
                maxwell_branches=[],
                enable_bulk_viscosity=True
            )
        )

        with raises(ValueError) as excinfo:
            validate_gradient_mode(config, strict=True)

        assert "INCOMPATIBLE" in str(excinfo.value)
        assert "bulk_viscosity=True" in str(excinfo.value)

    def test_maxwell_nonstrict_returns_false_with_warning(self):
        """Config with Maxwell in non-strict mode should return False with warning."""
        config = MockMPMConfig(
            material=MockMaterialConfig(
                maxwell_branches=[MockMaxwellBranch()],
                enable_bulk_viscosity=False
            )
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_gradient_mode(config, strict=False)

            assert result is False
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "INCOMPATIBLE" in str(w[0].message)

    def test_bulk_viscosity_nonstrict_returns_false_with_warning(self):
        """Config with bulk viscosity in non-strict mode should return False with warning."""
        config = MockMPMConfig(
            material=MockMaterialConfig(
                maxwell_branches=[],
                enable_bulk_viscosity=True
            )
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_gradient_mode(config, strict=False)

            assert result is False
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)


class TestConfigureGradientMode:
    """Tests for configure_gradient_mode function."""

    def test_numerical_mode_no_warning(self):
        """Numerical mode (default) should not issue warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            configure_gradient_mode(use_numerical=True, eps=1e-4)

            # Filter out any unrelated warnings
            gradient_warnings = [x for x in w if "analytical" in str(x.message).lower()
                                 or "INCOMPLETE" in str(x.message)]
            assert len(gradient_warnings) == 0

    def test_analytical_mode_issues_warning(self):
        """Analytical mode should issue UserWarning about incomplete gradients."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            configure_gradient_mode(use_numerical=False, eps=1e-4)

            # Find the warning about analytical mode
            analytical_warnings = [x for x in w if "INCOMPLETE" in str(x.message)]
            assert len(analytical_warnings) == 1
            assert issubclass(analytical_warnings[0].category, UserWarning)

    def test_configure_sets_module_globals(self):
        """Configure should set module-level variables."""
        global _USE_NUMERICAL_G_F, _FINITE_DIFF_EPS

        # Set to non-default values
        configure_gradient_mode(use_numerical=True, eps=5e-5)

        assert _USE_NUMERICAL_G_F is True
        assert _FINITE_DIFF_EPS == 5e-5

        # Reset to default
        configure_gradient_mode(use_numerical=True, eps=1e-4)


class TestModuleLevelConstants:
    """Tests for module-level constants."""

    def test_jacobian_clamp_values_are_fixed(self):
        """Jacobian clamp values should be fixed at (0.5, 2.0)."""
        assert _J_CLAMP_MIN == 0.5
        assert _J_CLAMP_MAX == 2.0


class TestExperimentalPTotalMode:
    """Tests for experimental P_total numerical gradient mode (FR-4)."""

    def setup_method(self):
        """Reset to default mode before each test."""
        configure_gradient_mode(
            use_numerical=True,
            eps=1e-4,
            experimental_p_total=False,
            max_particles=5000,
            max_steps=500
        )

    def test_experimental_mode_disabled_by_default(self):
        """Experimental mode should be disabled by default."""
        # Reset first to ensure clean state
        configure_gradient_mode(
            use_numerical=True,
            eps=1e-4,
            experimental_p_total=False,
            max_particles=5000,
            max_steps=500
        )
        assert is_experimental_mode_enabled() is False

    def test_experimental_mode_can_be_enabled(self):
        """Experimental mode can be enabled via configure_gradient_mode."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            configure_gradient_mode(experimental_p_total=True)

            assert is_experimental_mode_enabled() is True
            # Should issue warning
            exp_warnings = [x for x in w if "EXPERIMENTAL" in str(x.message)]
            assert len(exp_warnings) >= 1

    def test_scale_guards_configuration(self):
        """Scale guards should be configurable."""
        configure_gradient_mode(
            experimental_p_total=True,
            max_particles=2000,
            max_steps=200
        )
        max_p, max_s = get_scale_guards()
        assert max_p == 2000
        assert max_s == 200

    def test_experimental_mode_allows_maxwell_within_limits(self):
        """Experimental mode should allow Maxwell when within scale guards."""
        configure_gradient_mode(
            experimental_p_total=True,
            max_particles=5000,
            max_steps=500
        )
        config = MockMPMConfig(
            material=MockMaterialConfig(
                maxwell_branches=[MockMaxwellBranch()],
                enable_bulk_viscosity=False
            )
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_gradient_mode(
                config, strict=True,
                n_particles=1000, n_steps=100
            )
            assert result is True
            # Should issue experimental warning
            exp_warnings = [x for x in w if "EXPERIMENTAL" in str(x.message)]
            assert len(exp_warnings) >= 1

    def test_experimental_mode_particle_limit_strict(self):
        """Exceeding particle limit in strict mode should raise ValueError."""
        configure_gradient_mode(
            experimental_p_total=True,
            max_particles=1000,
            max_steps=500
        )
        config = MockMPMConfig(
            material=MockMaterialConfig(
                maxwell_branches=[MockMaxwellBranch()],
                enable_bulk_viscosity=False
            )
        )
        with raises(ValueError) as excinfo:
            validate_gradient_mode(
                config, strict=True,
                n_particles=2000, n_steps=100
            )
        assert "particle count" in str(excinfo.value)

    def test_experimental_mode_step_limit_strict(self):
        """Exceeding step limit in strict mode should raise ValueError."""
        configure_gradient_mode(
            experimental_p_total=True,
            max_particles=5000,
            max_steps=100
        )
        config = MockMPMConfig(
            material=MockMaterialConfig(
                maxwell_branches=[MockMaxwellBranch()],
                enable_bulk_viscosity=False
            )
        )
        with raises(ValueError) as excinfo:
            validate_gradient_mode(
                config, strict=True,
                n_particles=1000, n_steps=200
            )
        assert "step count" in str(excinfo.value)

    def test_experimental_mode_scale_guard_warn(self):
        """Exceeding scale guards in warn mode should return False."""
        configure_gradient_mode(
            experimental_p_total=True,
            max_particles=1000,
            max_steps=500
        )
        config = MockMPMConfig(
            material=MockMaterialConfig(
                maxwell_branches=[MockMaxwellBranch()],
                enable_bulk_viscosity=False
            )
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_gradient_mode(
                config, strict=False,
                n_particles=2000, n_steps=100
            )
            assert result is False
            assert len(w) >= 1

    def test_mode_switching(self):
        """Mode switching should update state correctly."""
        # Enable experimental mode
        configure_gradient_mode(experimental_p_total=True)
        assert is_experimental_mode_enabled() is True

        # Disable experimental mode
        configure_gradient_mode(experimental_p_total=False)
        assert is_experimental_mode_enabled() is False

        # Enable again with different limits
        configure_gradient_mode(
            experimental_p_total=True,
            max_particles=3000,
            max_steps=300
        )
        assert is_experimental_mode_enabled() is True
        max_p, max_s = get_scale_guards()
        assert max_p == 3000
        assert max_s == 300


def run_tests():
    """Run all tests without pytest."""
    import traceback

    test_classes = [
        TestValidateGradientMode,
        TestConfigureGradientMode,
        TestModuleLevelConstants,
        TestExperimentalPTotalMode,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                try:
                    method()
                    print(f"  [PASS] {test_class.__name__}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  [FAIL] {test_class.__name__}.{method_name}")
                    print(f"         {e}")
                    failed += 1
                except Exception as e:
                    print(f"  [ERROR] {test_class.__name__}.{method_name}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0


if __name__ == '__main__':
    import sys

    # Always use manual runner to avoid xengym package import issues
    print("Running tests with manual runner (avoiding package imports)...\n")
    success = run_tests()
    sys.exit(0 if success else 1)
