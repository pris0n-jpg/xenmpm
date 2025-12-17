"""
Linear Algebra Decomposition Utilities
Provides wrappers for polar decomposition and symmetric eigenvalue decomposition.
Uses Taichi built-in functions by default, with slots for custom SafeSVD if needed.
Includes SPD projection with Straight-Through Estimator (STE) for autodiff support.
"""
import taichi as ti


@ti.func
def polar_decompose(F: ti.template()) -> ti.template():
    """
    Polar decomposition: F = R @ S
    Returns rotation matrix R and symmetric stretch matrix S

    Args:
        F: 3x3 deformation gradient matrix

    Returns:
        R: 3x3 rotation matrix
        S: 3x3 symmetric stretch matrix
    """
    U, sig, V = ti.svd(F)
    R = U @ V.transpose()
    S = V @ ti.Matrix([[sig[0, 0], 0, 0],
                       [0, sig[1, 1], 0],
                       [0, 0, sig[2, 2]]]) @ V.transpose()
    return R, S


@ti.func
def safe_svd(F: ti.template(), eps: ti.f32 = 1e-8) -> ti.template():
    """
    Safe SVD with gradient handling for near-zero singular values

    Args:
        F: 3x3 matrix
        eps: Small epsilon for numerical stability

    Returns:
        U: 3x3 left singular vectors
        sig: 3x1 singular values
        V: 3x3 right singular vectors
    """
    # For now, use Taichi's built-in SVD
    # Can be replaced with custom implementation if needed for better autodiff stability
    U, sig, V = ti.svd(F)

    # Clamp singular values to avoid numerical issues
    for i in ti.static(range(3)):
        sig[i, i] = ti.max(sig[i, i], eps)

    return U, sig, V


@ti.func
def eig_sym_3x3(A: ti.template()) -> ti.template():
    """
    Symmetric eigenvalue decomposition for 3x3 matrix: A = Q @ Lambda @ Q^T

    Uses SVD as a more robust alternative to sym_eig for numerical stability.
    For symmetric positive semi-definite matrices: A = U Σ V^T where U = V.

    Args:
        A: 3x3 symmetric matrix (will be symmetrized if not perfectly symmetric)

    Returns:
        eigenvalues: 3x1 vector of eigenvalues
        eigenvectors: 3x3 matrix of eigenvectors (columns)
    """
    # Ensure matrix is symmetric
    A_sym = 0.5 * (A + A.transpose())

    # Use SVD for robust eigenvalue decomposition of symmetric matrix
    # For symmetric A: A = U Σ V^T, and U = V (up to sign)
    U, sig, V = ti.svd(A_sym)

    # Extract eigenvalues from singular values (diagonal of sig)
    eigenvalues = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]])

    # For symmetric matrices, eigenvectors are the columns of U (or V)
    eigenvectors = U

    return eigenvalues, eigenvectors


@ti.func
def make_spd(A: ti.template(), eps: ti.f32 = 1e-8) -> ti.template():
    """
    Project matrix to symmetric positive definite (SPD) space

    Args:
        A: 3x3 matrix
        eps: Minimum eigenvalue threshold

    Returns:
        A_spd: 3x3 SPD matrix
    """
    # Symmetrize
    A_sym = 0.5 * (A + A.transpose())

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eig_sym_3x3(A_sym)

    # Clamp eigenvalues to be positive
    for i in ti.static(range(3)):
        eigenvalues[i] = ti.max(eigenvalues[i], eps)

    # Reconstruct SPD matrix
    Lambda = ti.Matrix([[eigenvalues[0], 0, 0],
                        [0, eigenvalues[1], 0],
                        [0, 0, eigenvalues[2]]])
    A_spd = eigenvectors @ Lambda @ eigenvectors.transpose()

    return A_spd


@ti.func
def make_spd_ste(A: ti.template(), eps: ti.f32 = 1e-8) -> ti.template():
    """
    Project matrix to SPD space with Straight-Through Estimator (STE) for autodiff.

    The STE allows gradients to flow through the non-differentiable SPD projection
    by treating the projection as identity in the backward pass.

    Forward: A_spd = make_spd(A)
    Backward: grad_A = grad_A_spd (straight-through)

    Implementation: A_spd_ste = stop_grad(A_spd) + A - stop_grad(A)

    Args:
        A: 3x3 matrix (the relaxed internal variable b_bar_e_relaxed)
        eps: Minimum eigenvalue threshold

    Returns:
        A_spd: 3x3 SPD matrix with STE gradient
    """
    # Compute actual SPD projection
    A_spd = make_spd(A, eps)

    # STE: gradient flows through A as if projection were identity
    # Forward: returns A_spd
    # Backward: grad flows to A directly
    A_spd_ste = ti.stop_grad(A_spd) + A - ti.stop_grad(A)

    return A_spd_ste


@ti.func
def clamp_J(F: ti.template(), J_min: ti.f32 = 0.5, J_max: ti.f32 = 2.0) -> ti.template():
    """
    Clamp Jacobian (determinant) of deformation gradient to avoid extreme compression/expansion

    Args:
        F: 3x3 deformation gradient
        J_min: Minimum allowed Jacobian
        J_max: Maximum allowed Jacobian

    Returns:
        F_clamped: 3x3 deformation gradient with clamped Jacobian
    """
    J = F.determinant()
    J_clamped = ti.max(ti.min(J, J_max), J_min)

    F_clamped = F
    if ti.abs(J) > 1e-10:
        scale = ti.pow(J_clamped / J, 1.0 / 3.0)
        F_clamped = scale * F

    return F_clamped
