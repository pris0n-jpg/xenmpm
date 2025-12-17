"""
Constitutive Model Gradient Verification Tests

Directly tests ogden_mu gradient at the stress level,
bypassing the full MPM simulation chain.

Run with: pytest xengym/mpm/tests/test_constitutive_gradients.py -v
"""
import pytest
import numpy as np
from typing import Tuple

# Import centralized gradient tolerances (Tier A for constitutive tests)
from xengym.mpm.tests.conftest import GRADIENT_RTOL_STRICT

# Skip entire module if Taichi not available
taichi = pytest.importorskip("taichi")


@pytest.fixture(scope="module")
def constitutive_test_class():
    """Create the ConstitutiveGradientTest class for testing"""
    import taichi as ti
    from xengym.mpm.constitutive import compute_ogden_stress_general
    from xengym.mpm.constitutive_gradients import compute_ogden_stress_with_gradients

    @ti.data_oriented
    class ConstitutiveGradientTest:
        """Direct test of constitutive model gradients"""

        def __init__(self):
            self.n_ogden = 1
            self.ogden_mu = ti.field(dtype=ti.f32, shape=4)
            self.ogden_alpha = ti.field(dtype=ti.f32, shape=4)
            self.ogden_kappa = 10000.0

            # Test fields
            self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
            self.P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
            self.g_P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
            self.g_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
            self.g_mu = ti.field(dtype=ti.f32, shape=4)

        @ti.kernel
        def compute_stress(self):
            """Compute P = stress(F, mu)"""
            P_elastic, _ = compute_ogden_stress_general(
                self.F[None], self.ogden_mu, self.ogden_alpha,
                self.n_ogden, self.ogden_kappa
            )
            self.P[None] = P_elastic

        @ti.kernel
        def compute_stress_gradient(self):
            """Compute gradients of stress w.r.t. F and mu"""
            g_F_local, g_mu_local, g_alpha_local, g_kappa_local = compute_ogden_stress_with_gradients(
                self.F[None], self.ogden_mu, self.ogden_alpha,
                self.n_ogden, self.ogden_kappa, self.g_P[None]
            )
            self.g_F[None] = g_F_local
            for k in ti.static(range(4)):
                self.g_mu[k] = g_mu_local[k]

    return ConstitutiveGradientTest


class TestOgdenMuGradientDirect:
    """Test ogden_mu gradient with direct stress computation"""

    @pytest.mark.gradient
    def test_single_deformation_gradient(self, constitutive_test_class):
        """Test gradient for 20% compression in z"""
        test = constitutive_test_class()

        # Set material parameters
        mu_val = 1000.0
        alpha_val = 2.0
        test.ogden_mu[0] = mu_val
        test.ogden_alpha[0] = alpha_val

        # Set deformed F (20% compression in z)
        F_np = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.8]
        ], dtype=np.float32)
        test.F[None] = F_np

        # Compute stress
        test.compute_stress()
        P_np = test.P[None].to_numpy()

        # Define loss = 0.5 * ||P||^2_F (Frobenius norm squared)
        # This gives g_P = P
        test.g_P[None] = P_np

        # Compute gradient
        test.compute_stress_gradient()
        g_mu_analytic = test.g_mu[0]

        # Numerical verification
        eps = 0.1

        # f(mu + eps)
        test.ogden_mu[0] = mu_val + eps
        test.compute_stress()
        P_plus = test.P[None].to_numpy()
        loss_plus = 0.5 * np.sum(P_plus**2)

        # f(mu - eps)
        test.ogden_mu[0] = mu_val - eps
        test.compute_stress()
        P_minus = test.P[None].to_numpy()
        loss_minus = 0.5 * np.sum(P_minus**2)

        # Restore
        test.ogden_mu[0] = mu_val

        grad_mu_numerical = (loss_plus - loss_minus) / (2 * eps)

        # Check gradient (Tier A threshold: 1%)
        if abs(grad_mu_numerical) > 1e-12:
            rel_error = abs(g_mu_analytic - grad_mu_numerical) / abs(grad_mu_numerical)
            same_sign = (g_mu_analytic * grad_mu_numerical > 0)

            assert same_sign, f"Sign mismatch: analytic={g_mu_analytic}, numerical={grad_mu_numerical}"
            assert rel_error < GRADIENT_RTOL_STRICT, f"Relative error {rel_error:.4f} exceeds Tier A ({GRADIENT_RTOL_STRICT*100:.0f}%)"


class TestOgdenMuGradientVariousDeformations:
    """Test gradient across different deformation states"""

    @pytest.mark.gradient
    @pytest.mark.parametrize("name,F_values", [
        ("Identity", [[1,0,0],[0,1,0],[0,0,1]]),
        ("20% compression", [[1,0,0],[0,1,0],[0,0,0.8]]),
        ("20% extension", [[1,0,0],[0,1,0],[0,0,1.2]]),
        ("Simple shear", [[1,0.2,0],[0,1,0],[0,0,1]]),
        ("Volume change", [[0.9,0,0],[0,0.9,0],[0,0,0.9]]),
    ])
    def test_deformation_gradient(self, constitutive_test_class, name, F_values):
        """Test gradient for various deformation states"""
        test = constitutive_test_class()

        # Set material parameters
        mu_val = 1000.0
        alpha_val = 2.0
        test.ogden_mu[0] = mu_val
        test.ogden_alpha[0] = alpha_val

        F_np = np.array(F_values, dtype=np.float32)
        test.F[None] = F_np
        test.compute_stress()
        P_np = test.P[None].to_numpy()

        # g_P = P for loss = 0.5 ||P||^2
        test.g_P[None] = P_np
        test.compute_stress_gradient()
        g_mu_analytic = test.g_mu[0]

        # Numerical
        eps = 0.1
        test.ogden_mu[0] = mu_val + eps
        test.compute_stress()
        loss_plus = 0.5 * np.sum(test.P[None].to_numpy()**2)

        test.ogden_mu[0] = mu_val - eps
        test.compute_stress()
        loss_minus = 0.5 * np.sum(test.P[None].to_numpy()**2)

        test.ogden_mu[0] = mu_val

        grad_mu_numerical = (loss_plus - loss_minus) / (2 * eps)

        # Handle edge case: both near zero (e.g., Identity deformation)
        if abs(grad_mu_numerical) < 1e-6 and abs(g_mu_analytic) < 1e-6:
            rel_error = 0.0
            same_sign = True
        elif abs(grad_mu_numerical) > 1e-15:
            rel_error = abs(g_mu_analytic - grad_mu_numerical) / abs(grad_mu_numerical)
            same_sign = (g_mu_analytic * grad_mu_numerical > 0)
        else:
            rel_error = 0 if abs(g_mu_analytic) < 1e-15 else float('inf')
            same_sign = True

        # Tier A threshold for constitutive-level tests
        assert same_sign and rel_error < GRADIENT_RTOL_STRICT, \
            f"{name}: rel_error={rel_error:.4f}, Tier A limit={GRADIENT_RTOL_STRICT}, sign_match={same_sign}"
