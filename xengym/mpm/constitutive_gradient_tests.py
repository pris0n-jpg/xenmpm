"""
Constitutive Model Gradient Verification

Directly tests ogden_mu gradient at the stress level,
bypassing the full MPM simulation chain.

This isolates the constitutive gradient computation from
the P2G/GridOps/G2P chain to verify correctness.
"""
import taichi as ti
import numpy as np
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


def test_ogden_mu_gradient_direct():
    """Test ogden_mu gradient with direct stress computation"""
    ti.init(arch=ti.cpu, debug=True)

    test = ConstitutiveGradientTest()

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

    print("=" * 60)
    print("Constitutive Gradient Verification (Direct)")
    print("=" * 60)
    print(f"F (deformation gradient):")
    print(F_np)
    print(f"\nP (stress):")
    print(P_np)
    print(f"\n|P| = {np.linalg.norm(P_np):.6f}")

    # Define loss = 0.5 * ||P||^2_F (Frobenius norm squared)
    # This gives g_P = P
    test.g_P[None] = P_np

    # Compute gradient
    test.compute_stress_gradient()
    g_mu_analytic = test.g_mu[0]

    print(f"\nLoss = 0.5 * ||P||^2 = {0.5 * np.sum(P_np**2):.6f}")
    print(f"Analytic grad_mu: {g_mu_analytic:.10e}")

    # Numerical verification
    eps = 0.1  # Small perturbation for mu

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

    print(f"\n--- Numerical Gradient Verification ---")
    print(f"Analytic gradient:  {g_mu_analytic:.10e}")
    print(f"Numerical gradient: {grad_mu_numerical:.10e}")
    print(f"loss_plus:  {loss_plus:.8f}")
    print(f"loss_minus: {loss_minus:.8f}")
    print(f"loss_diff:  {loss_plus - loss_minus:.10e}")

    # Check gradient
    if abs(grad_mu_numerical) > 1e-12:
        rel_error = abs(g_mu_analytic - grad_mu_numerical) / abs(grad_mu_numerical)
        same_sign = (g_mu_analytic * grad_mu_numerical > 0)
        print(f"\nRelative error: {rel_error:.6f}")
        print(f"Same sign: {same_sign}")

        passed = same_sign and rel_error < 0.1
        print(f"PASSED: {passed}")
        return passed, rel_error
    else:
        print("\nNumerical gradient too small")
        return False, float('inf')


def test_ogden_mu_gradient_various_deformations():
    """Test gradient across different deformation states"""
    ti.init(arch=ti.cpu, debug=True)

    test = ConstitutiveGradientTest()

    # Set material parameters
    mu_val = 1000.0
    alpha_val = 2.0
    test.ogden_mu[0] = mu_val
    test.ogden_alpha[0] = alpha_val

    print("=" * 60)
    print("Gradient Test Across Various Deformations")
    print("=" * 60)

    deformations = [
        ("Identity", np.eye(3, dtype=np.float32)),
        ("20% compression", np.array([[1,0,0],[0,1,0],[0,0,0.8]], dtype=np.float32)),
        ("20% extension", np.array([[1,0,0],[0,1,0],[0,0,1.2]], dtype=np.float32)),
        ("Simple shear", np.array([[1,0.2,0],[0,1,0],[0,0,1]], dtype=np.float32)),
        ("Volume change", np.array([[0.9,0,0],[0,0.9,0],[0,0,0.9]], dtype=np.float32)),
    ]

    results = []
    for name, F_np in deformations:
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

        status = "PASS" if (same_sign and rel_error < 0.1) else "FAIL"
        results.append((name, g_mu_analytic, grad_mu_numerical, rel_error, status))

        print(f"\n{name}:")
        print(f"  det(F) = {np.linalg.det(F_np):.4f}")
        print(f"  |P| = {np.linalg.norm(P_np):.4f}")
        print(f"  Analytic: {g_mu_analytic:.6e}, Numerical: {grad_mu_numerical:.6e}")
        print(f"  Rel error: {rel_error:.6f}, Status: {status}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r[4] == "PASS")
    print(f"Passed: {passed}/{len(results)}")

    return all(r[4] == "PASS" for r in results)


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("TEST 1: Single Deformation Direct Gradient")
    print("#" * 70)
    passed1, _ = test_ogden_mu_gradient_direct()

    print("\n" + "#" * 70)
    print("TEST 2: Various Deformations")
    print("#" * 70)
    passed2 = test_ogden_mu_gradient_various_deformations()

    print("\n" + "=" * 70)
    print(f"FINAL: Test1={'PASS' if passed1 else 'FAIL'}, Test2={'PASS' if passed2 else 'FAIL'}")
    print("=" * 70)
