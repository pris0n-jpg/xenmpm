"""
Investigation Summary: ogden_mu Gradient Sign Error
Date: 2025-12-06

## Problem Description (Confirmed)
- Single particle tests: PASS (analytical matches numerical)
- 2 particle tests: PASS (both close and far configurations)
- 8 particle tests: FAIL (analytical has OPPOSITE sign to numerical)
  - Analytical: -3.83e-12
  - Numerical:  +2.76e-11 (reproduced)
  - Magnitude difference: ~7x

## Verified Correct
1. ∂P/∂μ formula (tested with pure NumPy):
   - For F_zz = 0.7 (compression), ∂P[2,2]/∂μ = -0.178 (NEGATIVE!)
   - This is correct: deviatoric projection causes z-stress to DECREASE with μ
   - Analytical and numerical ∂P/∂μ match within 1e-9

2. Code structure:
   - Forward P2G: grid_P += w * (m * v_apic + V_p * P @ F^T @ dpos)
   - Backward P2G: g_affine = Σ w * g_grid_P ⊗ dpos^T, then g_P = V_p * g_affine @ F
   - Gradient accumulation: g_ogden_mu[k] += g_mu_local[k]

## Potential Issues to Investigate

### 1. Multi-particle grid interaction
With 8 particles in 2x2x2 configuration:
- Multiple particles share grid nodes
- Grid velocity = (Σ particle contributions) / (Σ masses)
- Backward: each particle reads FULL g_grid_P (not partitioned)
This is mathematically correct, but could cause subtle numerical issues

### 2. Weighted dpos sum
g_affine = Σ_I w_I * g_grid_P_I ⊗ dpos_I^T
For different particle positions, the weighted sum of dpos differs
Some particles may contribute positive g_mu, others negative
Net effect depends on exact positions and overlap patterns

### 3. Sign of g_P propagation
Chain: loss → g_x → g_grid_v → g_grid_P → g_affine → g_P → g_mu
Each link preserves or flips sign - need to verify each step

## Recommended Debug Steps
1. Add logging to backward kernels to print intermediate values
2. Test with 3,4,5,6,7 particles to find exact threshold
3. Compare per-particle g_mu_local contributions
4. Verify g_grid_P values are consistent between forward/backward

## Physical Insight
For compression (F_zz = 0.7), the material stress pushes outward (+z).
If target is +z and material is undershooting:
  - g_x < 0 (want more +z movement)
  - Higher μ should help (more restoring force)
  - But ∂P/∂μ < 0 in z due to deviatoric projection!
  - So higher μ actually DECREASES stress, making it WORSE
  - Therefore g_mu should be POSITIVE (increasing μ increases loss)

But this contradicts the analytical result of NEGATIVE g_mu!
The issue might be in how g_P is computed from g_affine.
"""
