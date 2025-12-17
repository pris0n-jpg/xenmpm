"""
Stress Gradient Chain Tests

Tests the constitutive stress gradient (dP/dmu) in isolation
and verifies the full gradient chain: g_affine -> g_P -> g_mu

Run with: pytest xengym/mpm/tests/test_stress_gradients.py -v
"""
import pytest
import numpy as np

# Import centralized gradient tolerances
from xengym.mpm.tests.conftest import GRADIENT_RTOL_STRICT, GRADIENT_RTOL_NORMAL, GradientMetricsReporter

# Skip entire module if Taichi not available
taichi = pytest.importorskip("taichi")


class TestStressGradient:
    """Test dP/dmu using finite difference"""

    @pytest.mark.gradient
    def test_stress_gradient_numerical(self):
        """Verify dP/dmu matches finite difference"""
        import taichi as ti
        from xengym.mpm.constitutive import compute_ogden_stress_general
        from xengym.mpm.constitutive_gradients import compute_ogden_stress_with_gradients

        # Create Ogden parameter fields
        ogden_mu = ti.field(dtype=ti.f32, shape=4)
        ogden_alpha = ti.field(dtype=ti.f32, shape=4)

        # Test parameters
        mu_val = 500.0
        alpha_val = 2.0
        kappa = 2500.0
        n_ogden = 1

        ogden_mu[0] = mu_val
        ogden_alpha[0] = alpha_val
        for i in range(1, 4):
            ogden_mu[i] = 0.0
            ogden_alpha[i] = 1.0

        # Test with compressed F
        F_test = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.7]  # 30% compression
        ], dtype=np.float32)

        # Test g_P (random direction for testing)
        g_P_test = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, -0.2]
        ], dtype=np.float32)

        # Results storage
        result_P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        result_g_mu = ti.Vector.field(4, dtype=ti.f32, shape=())

        @ti.kernel
        def compute_stress_and_gradient(F: ti.types.ndarray(), g_P: ti.types.ndarray()):
            F_mat = ti.Matrix([
                [F[0, 0], F[0, 1], F[0, 2]],
                [F[1, 0], F[1, 1], F[1, 2]],
                [F[2, 0], F[2, 1], F[2, 2]]
            ])
            g_P_mat = ti.Matrix([
                [g_P[0, 0], g_P[0, 1], g_P[0, 2]],
                [g_P[1, 0], g_P[1, 1], g_P[1, 2]],
                [g_P[2, 0], g_P[2, 1], g_P[2, 2]]
            ])

            P, psi = compute_ogden_stress_general(
                F_mat, ogden_mu, ogden_alpha, n_ogden, kappa
            )
            result_P[None] = P

            g_F, g_mu, g_alpha, g_kappa = compute_ogden_stress_with_gradients(
                F_mat, ogden_mu, ogden_alpha, n_ogden, kappa, g_P_mat
            )
            result_g_mu[None] = g_mu

        @ti.kernel
        def compute_stress_only(F: ti.types.ndarray()):
            F_mat = ti.Matrix([
                [F[0, 0], F[0, 1], F[0, 2]],
                [F[1, 0], F[1, 1], F[1, 2]],
                [F[2, 0], F[2, 1], F[2, 2]]
            ])
            P, psi = compute_ogden_stress_general(
                F_mat, ogden_mu, ogden_alpha, n_ogden, kappa
            )
            result_P[None] = P

        # Compute analytic gradient
        compute_stress_and_gradient(F_test, g_P_test)
        g_mu_analytic = result_g_mu[None].to_numpy()[0]

        # Numerical gradient via finite difference
        eps = 1.0

        # mu + eps
        ogden_mu[0] = mu_val + eps
        compute_stress_only(F_test)
        P_plus = result_P[None].to_numpy()

        # mu - eps
        ogden_mu[0] = mu_val - eps
        compute_stress_only(F_test)
        P_minus = result_P[None].to_numpy()

        # Restore mu
        ogden_mu[0] = mu_val

        # Finite difference dP/dmu
        dP_dmu_numerical = (P_plus - P_minus) / (2 * eps)

        # Numerical gradient: <g_P, dP/dmu>
        g_mu_numerical = np.sum(g_P_test * dP_dmu_numerical)

        # Compare
        if abs(g_mu_numerical) > 1e-15:
            rel_error = abs(g_mu_analytic - g_mu_numerical) / abs(g_mu_numerical)
            sign_match = (g_mu_analytic * g_mu_numerical) > 0
            assert sign_match, f"Sign mismatch: analytic={g_mu_analytic}, numerical={g_mu_numerical}"
            assert rel_error < GRADIENT_RTOL_STRICT, f"Relative error {rel_error:.4f} exceeds Tier A ({GRADIENT_RTOL_STRICT*100:.0f}%)"
        else:
            assert abs(g_mu_analytic) < 1e-10, "Expected zero gradient"


class TestStressGradientChain:
    """Test the full gradient chain: g_affine -> g_P -> g_mu"""

    @pytest.mark.gradient
    def test_gradient_chain_numerical(self):
        """Verify full gradient chain matches finite difference"""
        import taichi as ti
        from xengym.mpm.constitutive import compute_ogden_stress_general
        from xengym.mpm.constitutive_gradients import compute_ogden_stress_with_gradients

        # Create Ogden parameter fields
        ogden_mu = ti.field(dtype=ti.f32, shape=4)
        ogden_alpha = ti.field(dtype=ti.f32, shape=4)

        # Test parameters
        mu_val = 500.0
        alpha_val = 2.0
        kappa = 2500.0
        n_ogden = 1
        V_p = 8e-6  # Small volume (dx^3 with dx=0.02)

        ogden_mu[0] = mu_val
        ogden_alpha[0] = alpha_val
        for i in range(1, 4):
            ogden_mu[i] = 0.0
            ogden_alpha[i] = 1.0

        # Test with compressed F
        F_test = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.7]
        ], dtype=np.float32)

        # Test g_affine (gradient w.r.t. affine = V_p * P @ F^T)
        g_affine_test = np.array([
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, -0.002]
        ], dtype=np.float32)

        # Compute g_P = V_p * g_affine @ F
        g_P_computed = V_p * (g_affine_test @ F_test)

        # Results storage
        result_g_mu = ti.Vector.field(4, dtype=ti.f32, shape=())
        result_P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

        @ti.kernel
        def compute_gradient_with_g_P(F: ti.types.ndarray(), g_P: ti.types.ndarray()):
            F_mat = ti.Matrix([
                [F[0, 0], F[0, 1], F[0, 2]],
                [F[1, 0], F[1, 1], F[1, 2]],
                [F[2, 0], F[2, 1], F[2, 2]]
            ])
            g_P_mat = ti.Matrix([
                [g_P[0, 0], g_P[0, 1], g_P[0, 2]],
                [g_P[1, 0], g_P[1, 1], g_P[1, 2]],
                [g_P[2, 0], g_P[2, 1], g_P[2, 2]]
            ])
            g_F, g_mu, g_alpha, g_kappa = compute_ogden_stress_with_gradients(
                F_mat, ogden_mu, ogden_alpha, n_ogden, kappa, g_P_mat
            )
            result_g_mu[None] = g_mu

        @ti.kernel
        def compute_stress_only(F: ti.types.ndarray()):
            F_mat = ti.Matrix([
                [F[0, 0], F[0, 1], F[0, 2]],
                [F[1, 0], F[1, 1], F[1, 2]],
                [F[2, 0], F[2, 1], F[2, 2]]
            ])
            P, psi = compute_ogden_stress_general(
                F_mat, ogden_mu, ogden_alpha, n_ogden, kappa
            )
            result_P[None] = P

        # Analytic gradient
        compute_gradient_with_g_P(F_test, g_P_computed)
        g_mu_analytic = result_g_mu[None].to_numpy()[0]

        # Numerical verification of full chain
        eps = 1.0

        # Compute affine at mu+eps and mu-eps
        ogden_mu[0] = mu_val + eps
        compute_stress_only(F_test)
        P_plus = result_P[None].to_numpy()
        affine_plus = V_p * P_plus @ F_test.T

        ogden_mu[0] = mu_val - eps
        compute_stress_only(F_test)
        P_minus = result_P[None].to_numpy()
        affine_minus = V_p * P_minus @ F_test.T

        ogden_mu[0] = mu_val

        # d(affine)/d(mu)
        d_affine_dmu = (affine_plus - affine_minus) / (2 * eps)

        # dL/d(mu) = <g_affine, d(affine)/d(mu)>
        g_mu_numerical = np.sum(g_affine_test * d_affine_dmu)

        # Compare
        if abs(g_mu_numerical) > 1e-15:
            rel_error = abs(g_mu_analytic - g_mu_numerical) / abs(g_mu_numerical)
            sign_match = (g_mu_analytic * g_mu_numerical) > 0
            assert sign_match, f"Sign mismatch: analytic={g_mu_analytic}, numerical={g_mu_numerical}"
            assert rel_error < GRADIENT_RTOL_STRICT, f"Relative error {rel_error:.6f} exceeds Tier A ({GRADIENT_RTOL_STRICT*100:.0f}%)"
        else:
            assert abs(g_mu_analytic) < 1e-15, "Expected zero gradient"


class TestStressGradientWithReporter:
    """Demonstrates GradientMetricsReporter usage for gradient verification."""

    @pytest.mark.gradient
    def test_gradient_with_reporter(self, gradient_reporter):
        """Verify gradient and record metrics using GradientMetricsReporter.

        This test demonstrates how to use the reporter for structured
        gradient quality metrics collection.
        """
        import taichi as ti
        from xengym.mpm.constitutive import compute_ogden_stress_general
        from xengym.mpm.constitutive_gradients import compute_ogden_stress_with_gradients

        # Create Ogden parameter fields
        ogden_mu = ti.field(dtype=ti.f32, shape=4)
        ogden_alpha = ti.field(dtype=ti.f32, shape=4)

        # Test parameters
        mu_val = 500.0
        alpha_val = 2.0
        kappa = 2500.0
        n_ogden = 1

        ogden_mu[0] = mu_val
        ogden_alpha[0] = alpha_val
        for i in range(1, 4):
            ogden_mu[i] = 0.0
            ogden_alpha[i] = 1.0

        # Test with 30% compression
        F_test = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.7]
        ], dtype=np.float32)

        g_P_test = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, -0.2]
        ], dtype=np.float32)

        # Results storage
        result_P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        result_g_mu = ti.Vector.field(4, dtype=ti.f32, shape=())

        @ti.kernel
        def compute_stress_and_gradient(F: ti.types.ndarray(), g_P: ti.types.ndarray()):
            F_mat = ti.Matrix([
                [F[0, 0], F[0, 1], F[0, 2]],
                [F[1, 0], F[1, 1], F[1, 2]],
                [F[2, 0], F[2, 1], F[2, 2]]
            ])
            g_P_mat = ti.Matrix([
                [g_P[0, 0], g_P[0, 1], g_P[0, 2]],
                [g_P[1, 0], g_P[1, 1], g_P[1, 2]],
                [g_P[2, 0], g_P[2, 1], g_P[2, 2]]
            ])

            P, psi = compute_ogden_stress_general(
                F_mat, ogden_mu, ogden_alpha, n_ogden, kappa
            )
            result_P[None] = P

            g_F, g_mu, g_alpha, g_kappa = compute_ogden_stress_with_gradients(
                F_mat, ogden_mu, ogden_alpha, n_ogden, kappa, g_P_mat
            )
            result_g_mu[None] = g_mu

        @ti.kernel
        def compute_stress_only(F: ti.types.ndarray()):
            F_mat = ti.Matrix([
                [F[0, 0], F[0, 1], F[0, 2]],
                [F[1, 0], F[1, 1], F[1, 2]],
                [F[2, 0], F[2, 1], F[2, 2]]
            ])
            P, psi = compute_ogden_stress_general(
                F_mat, ogden_mu, ogden_alpha, n_ogden, kappa
            )
            result_P[None] = P

        # Compute analytic gradient
        compute_stress_and_gradient(F_test, g_P_test)
        g_mu_analytic = result_g_mu[None].to_numpy()[0]

        # Numerical gradient via finite difference
        eps = 1.0
        ogden_mu[0] = mu_val + eps
        compute_stress_only(F_test)
        P_plus = result_P[None].to_numpy()

        ogden_mu[0] = mu_val - eps
        compute_stress_only(F_test)
        P_minus = result_P[None].to_numpy()

        ogden_mu[0] = mu_val  # Restore

        dP_dmu_numerical = (P_plus - P_minus) / (2 * eps)
        g_mu_numerical = np.sum(g_P_test * dP_dmu_numerical)

        # Record metrics using reporter
        record = gradient_reporter.record(
            param_name="ogden_mu",
            analytic=float(g_mu_analytic),
            numerical=float(g_mu_numerical),
            tier="A",
            test_name="test_gradient_with_reporter",
            extra={"F_compression": "30%", "eps": eps}
        )

        # Verify using recorded result
        assert record["passed"], f"Tier A check failed: rel_error={record['rel_error']:.4f}"

        # Print summary for visibility
        gradient_reporter.print_summary()
