## Context
- Taichi 1.7.4 的 AD 对含原子 scatter/gather 的 P2G/G2P 不支持反向，Tape 包裹 `solver.step()` 会报 “[auto_diff] Not supported”；现有接口为避免崩溃已阻断。
- 目标：通过手工伴随（Manual Adjoint）实现可微 MPM，使材料参数和初始状态可求梯度；明确可微范围（含/不含 Maxwell、SPD 处理、摩擦边界），保持前向性能与稳定防护。

## Goals / Non-Goals
- Goals: 为 P2G/G2P/GridOps/F 更新/Maxwell/能量求和提供前向+伴随；支持材料参数（Ogden/Maxwell）与初始粒子状态梯度；提供位置/速度/总能量可微 loss；提供可微/非可微模式切换与验证。
- Non-Goals: 修改上游 Taichi AD；首版不追求极致性能；首版不对摩擦参数（μ_s/μ_k/K_clear）求导，边界硬约束法向梯度截断；SPD 精确伴随（采用 STE）。

## High-level Design
1) 前向数据流每步：P2G → Grid Ops（v=P/M、外力、BC） → G2P → F 更新 + SPD → Maxwell 更新 → 能量/loss。  
2) 反向：对 T 步做反向时间循环，子 kernel 顺序：loss backward → F 更新/Maxwell backward → G2P backward → Grid Ops backward → P2G backward；材料参数梯度跨步累加。  
3) 显存：粒子存 x,v,F,C（+b_bar_e 可选）；网格尽量重算，必要时存 M_I；Maxwell 可配置 needs_grad（默认关）。

## Core Adjoints (Skeleton)
Let g_q = ∂L/∂q, weight w_ip(x_p), d_ip = x_I - x_p.

### P2G backward
- Inputs: g_P_I, g_M_I (from Grid Ops backward)。  
- v_apic = v_p + C_p d_ip.  
- g_v_p += Σ_I w_ip m_p g_P_I.  
- g_C_p += Σ_I w_ip m_p (g_P_I ⊗ d_ip^T).  
- g_m_p += Σ_I w_ip (g_P_I · v_apic) + Σ_I w_ip g_M_I.  
- g_x_p:  
  - 权重导数: g_x_p^(w) += Σ_I (∂w_ip/∂x_p)(ΔP_ip·g_P_I + m_p g_M_I).  
  - 仿射项: g_x_p^(affine) += Σ_I w_ip m_p (-C_p)^T g_P_I.  
  - 应力项: 由核梯度与 g_P_I 聚合到 g_F_p/材料参数。  
- g_F_p, g_θ: 来自应力项 g_P_I -> g_P_p -> ∂P/∂F, ∂P/∂θ。

### Grid Ops backward (v = P/M, gravity, BC)
- Forward: v_I = P_I/M_I (M_I>ε)，v_I += dt·gravity，v_I = BC(v_I).  
- Backward (M_I>ε):  
  - g_P_I += g_v_I / M_I  
  - g_M_I += -(g_v_I · v_I)/M_I  
- Gravity: g_v_before = g_v_after.  
- BC：法向截断或按 BC 映射（Sticky: 法向梯度置零；Slip: 法向截断切向保留；Bounce: 法向反号），文档中列明策略。

### G2P backward
- Forward:  
  v_p = Σ_I w_ip v_I_after;  
  C_p = κ Σ_I w_ip v_I_after ⊗ d_ip^T;  
  x_p^{n+1} = x_p^n + dt v_p.  
- Backward:  
  g_v_I += Σ_p w_ip g_v_p + κ Σ_p w_ip (g_C_p : d_ip^T);  
  g_x_p^n += g_x_p^{n+1} + Σ_I ∂w_ip/∂x_p (v_I·g_v_p + κ g_C_p : (v_I ⊗ d_ip^T)) - κ Σ_I w_ip (g_C_p^T v_I).

### F 更新 + SPD (STE)
- Forward: F_raw = (I + dt C) F_old; F_new = SPD(F_raw).  
- Backward (STE): g_F_raw += g_F_new;  
  g_C += dt g_F_raw F_old^T;  
  g_F_old += (I + dt C)^T g_F_raw.  
- SPD 触发频率可统计；文档说明梯度偏差风险。

### Maxwell 内变量（可选）
- Forward: a = exp(-dt/τ); b_e^{n+1} = a b_e^n + (1-a) b_e^trial.  
- Backward:  
  g_{b_e^n} += a g_{b_e^{n+1}};  
  g_{b_e^trial} += (1-a) g_{b_e^{n+1}};  
  g_τ += ⟨g_{b_e^{n+1}}, ∂b_e^{n+1}/∂τ⟩, ∂a/∂τ = dt/τ^2 exp(-dt/τ), ∂b_e^{n+1}/∂τ = ∂a/∂τ (b_e^n - b_e^trial).  
- 存储：默认不求导则不存时间序列；开启梯度则存 b_e^n/b_e^trial 或用 checkpoint。

### Loss
- 位置、速度、总能量（K+E_elastic+E_visco+E_proj）；总能量可选，默认声明范围与实现一致。

## Verification Targets
- 数值梯度对比：rel_err < 1e-4, cos_sim > 0.99（小规模场景，对 x0/v0/部分 μ_i/τ）。  
- Grid normalization backward：质量扰动对 loss 有非零梯度（g_M ≠ 0）。  
- APIC 仿射项：去掉 -C_p 项时梯度对比失败，开启后通过。  
- SPD 投影观察：记录触发比例，极端形变下梯度可能偏差需文档说明。  
- Maxwell（若启用）：对 τ 有限差分对齐。  
- 性能/显存：记录前/反向耗时、峰值显存，与估算同量级（误差 <2x）。

## Open Questions
- 反向归约策略：默认块归约（无原子写）；是否需要原子方案作为可选？  
- Maxwell needs_grad 默认关/开？建议默认关，按配置开启。  
- BC/摩擦：法向硬约束梯度截断；摩擦参数首版 no-grad，需在接口/文档标注。
