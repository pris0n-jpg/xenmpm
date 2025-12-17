"""
Test to verify the constitutive stress gradient (∂P/∂μ) in isolation

This test isolates the stress gradient computation from the full MPM solver
to determine if the issue is in the constitutive gradient or in the chain.
"""
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug=False)

# Import constitutive functions
from xengym.mpm.constitutive import compute_ogden_stress_general
from xengym.mpm.constitutive_gradients import compute_ogden_stress_with_gradients


def test_stress_gradient():
    """Test ∂P/∂μ using finite difference"""
    print("=" * 70)
    print("STRESS GRADIENT VERIFICATION (∂P/∂μ)")
    print("=" * 70)

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
        [0.0, 0.0, -0.2]  # Stronger in z direction
    ], dtype=np.float32)

    # Results storage
    result_P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
    result_g_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
    result_g_mu = ti.Vector.field(4, dtype=ti.f32, shape=())
    result_g_alpha = ti.Vector.field(4, dtype=ti.f32, shape=())

    @ti.kernel
    def compute_stress_and_gradient(
        F: ti.types.ndarray(),
        g_P: ti.types.ndarray()
    ):
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

        # Compute stress
        P, psi = compute_ogden_stress_general(
            F_mat, ogden_mu, ogden_alpha, n_ogden, kappa
        )
        result_P[None] = P

        # Compute stress gradients
        g_F, g_mu, g_alpha, g_kappa = compute_ogden_stress_with_gradients(
            F_mat, ogden_mu, ogden_alpha, n_ogden, kappa, g_P_mat
        )
        result_g_F[None] = g_F
        result_g_mu[None] = g_mu
        result_g_alpha[None] = g_alpha

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
    P_center = result_P[None].to_numpy()
    g_mu_analytic = result_g_mu[None].to_numpy()[0]

    print(f"\nF (compressed):")
    print(F_test)
    print(f"\ng_P (input gradient):")
    print(g_P_test)
    print(f"\nP (stress at μ={mu_val}):")
    print(P_center)
    print(f"\nAnalytic ∂L/∂μ = <g_P, ∂P/∂μ>: {g_mu_analytic:.10e}")

    # Numerical gradient via finite difference
    eps = 1.0

    # μ + eps
    ogden_mu[0] = mu_val + eps
    compute_stress_only(F_test)
    P_plus = result_P[None].to_numpy()

    # μ - eps
    ogden_mu[0] = mu_val - eps
    compute_stress_only(F_test)
    P_minus = result_P[None].to_numpy()

    # Restore μ
    ogden_mu[0] = mu_val

    # Finite difference ∂P/∂μ
    dP_dmu_numerical = (P_plus - P_minus) / (2 * eps)

    # Numerical gradient: <g_P, ∂P/∂μ>
    g_mu_numerical = np.sum(g_P_test * dP_dmu_numerical)

    print(f"\n--- Finite Difference Verification ---")
    print(f"ε = {eps}")
    print(f"P(μ+ε):\n{P_plus}")
    print(f"P(μ-ε):\n{P_minus}")
    print(f"\n∂P/∂μ (numerical):\n{dP_dmu_numerical}")
    print(f"\nNumerical ∂L/∂μ = <g_P, ∂P/∂μ>: {g_mu_numerical:.10e}")

    # Compare
    print(f"\n--- Comparison ---")
    print(f"Analytic:  {g_mu_analytic:.10e}")
    print(f"Numerical: {g_mu_numerical:.10e}")

    if abs(g_mu_numerical) > 1e-15:
        rel_error = abs(g_mu_analytic - g_mu_numerical) / abs(g_mu_numerical)
        sign_match = (g_mu_analytic * g_mu_numerical) > 0
        print(f"Relative error: {rel_error:.4f}")
        print(f"Sign match: {sign_match}")

        passed = rel_error < 0.1 and sign_match
    else:
        print(f"Numerical gradient too small")
        passed = abs(g_mu_analytic) < 1e-10

    print(f"\nRESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_stress_gradient_chain():
    """Test the full gradient chain: g_affine -> g_P -> g_mu"""
    print("\n" + "=" * 70)
    print("STRESS GRADIENT CHAIN TEST")
    print("=" * 70)

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
    # Simulate what comes from grid gradient accumulation
    g_affine_test = np.array([
        [0.001, 0.0, 0.0],
        [0.0, 0.001, 0.0],
        [0.0, 0.0, -0.002]  # Gradient in z direction
    ], dtype=np.float32)

    # Compute g_P = V_p * g_affine @ F
    # Actually: affine = V_p * P @ F^T
    # So: g_(V_p * P @ F^T) = g_affine
    # And: g_P = V_p * g_affine @ (F^T)^T = V_p * g_affine @ F
    g_P_computed = V_p * (g_affine_test @ F_test)

    print(f"\nV_p (volume): {V_p:.2e}")
    print(f"g_affine (from grid):\n{g_affine_test}")
    print(f"\ng_P = V_p * g_affine @ F:\n{g_P_computed}")
    print(f"g_P magnitude: {np.linalg.norm(g_P_computed):.10e}")

    # Results storage
    result_g_mu = ti.Vector.field(4, dtype=ti.f32, shape=())
    result_P = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

    @ti.kernel
    def compute_gradient_with_g_P(
        F: ti.types.ndarray(),
        g_P: ti.types.ndarray()
    ):
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

    print(f"\nAnalytic g_mu: {g_mu_analytic:.10e}")

    # Numerical verification of full chain:
    # affine = V_p * P(μ) @ F^T
    # L = <g_affine, affine> = V_p * <g_affine, P @ F^T>
    # ∂L/∂μ = V_p * <g_affine, ∂P/∂μ @ F^T>

    eps = 1.0

    # Compute affine at μ+eps and μ-eps
    ogden_mu[0] = mu_val + eps
    compute_stress_only(F_test)
    P_plus = result_P[None].to_numpy()
    affine_plus = V_p * P_plus @ F_test.T

    ogden_mu[0] = mu_val - eps
    compute_stress_only(F_test)
    P_minus = result_P[None].to_numpy()
    affine_minus = V_p * P_minus @ F_test.T

    ogden_mu[0] = mu_val

    # ∂affine/∂μ
    d_affine_dmu = (affine_plus - affine_minus) / (2 * eps)

    # ∂L/∂μ = <g_affine, ∂affine/∂μ>
    g_mu_numerical = np.sum(g_affine_test * d_affine_dmu)

    print(f"\n--- Full Chain Numerical Verification ---")
    print(f"d(affine)/dμ:\n{d_affine_dmu}")
    print(f"Numerical g_mu: {g_mu_numerical:.10e}")

    # Compare
    print(f"\n--- Comparison ---")
    print(f"Analytic:  {g_mu_analytic:.10e}")
    print(f"Numerical: {g_mu_numerical:.10e}")

    if abs(g_mu_numerical) > 1e-15:
        rel_error = abs(g_mu_analytic - g_mu_numerical) / abs(g_mu_numerical)
        sign_match = (g_mu_analytic * g_mu_numerical) > 0
        print(f"Relative error: {rel_error:.6f}")
        print(f"Sign match: {sign_match}")

        passed = rel_error < 0.1 and sign_match
    else:
        print(f"Numerical gradient too small")
        passed = abs(g_mu_analytic) < 1e-15

    print(f"\nRESULT: {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    import sys

    test1_passed = test_stress_gradient()
    test2_passed = test_stress_gradient_chain()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Stress gradient test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Chain test: {'PASSED' if test2_passed else 'FAILED'}")

    sys.exit(0 if (test1_passed and test2_passed) else 1)
