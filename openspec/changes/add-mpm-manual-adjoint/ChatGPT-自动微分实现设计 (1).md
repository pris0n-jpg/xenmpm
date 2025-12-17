# 自动微分实现设计

**User:** Anonymous (loilp@magician.elementfx.com)  
**Created:** 12/2/2025 13:42:03  
**Updated:** 12/2/2025 14:09:34  
**Exported:** 12/2/2025 14:17:52  
**Link:** [https://chatgpt.com/g/g-p-692e7affac78819196503d13b4003af9-mpm/c/692e7c2a-9c78-8331-96f6-a41a66f737a8](https://chatgpt.com/g/g-p-692e7affac78819196503d13b4003af9-mpm/c/692e7c2a-9c78-8331-96f6-a41a66f737a8)  

## Prompt:
已思考 2m 5s

下面是一套**已经把 Gemini 指出的 1/2/4 点漏洞都补上**、并整体梳理过的完整版三件套：

*   `design.md`（设计）
*   `spec.md`（需求）
*   `tasks.md`（任务拆分）

你可以直接整体替换 repo 里的同名文件，然后按你实际代码的小细节（权重核、Maxwell 具体形式等）再微调。

* * *

🧩 design.md（完整更新版）
-------------------

````markdown
## Context

- 现状：
  - Taichi 1.7.4 的 AD 无法对包含原子 scatter/gather 的 P2G/G2P kernel 做反向，`solver.step()` 被 Tape 包起来时会直接报错：
    - `[auto_diff.cpp…] Not supported`
  - 为避免崩溃，当前代码在入口处抛出 `NotImplementedError`，autodiff 接口对 MPM 核心步骤是阻断状态。
- 目标：
  - 通过 **手工伴随（Manual Adjoint）** 实现可微 MPM：
    - 支持对材料参数（如 Ogden / Maxwell 参数）与初始粒子状态（x, v, F）求梯度；
    - 明确哪些物理分支（SPD 投影、接触/摩擦、边界条件）提供近似梯度、哪些不求导；
    - 保持现有前向性能与数值稳定防护。

---

## Goals / Non-Goals

### Goals

- 为 P2G / G2P / Grid Ops / F 更新 / Maxwell 内变量 / 能量求和 等子模块设计并实现 **前向 + 手工伴随 backward kernel**。
- 在可微模式下，提供以下量的梯度：
  - 粒子初始状态：x₀, v₀, F₀；
  - 材料参数：Ogden / Maxwell 参数（μ_i, α_i, τ 等），可配置；
  - 可选：Maxwell 内变量相关参数。
- 提供一组可微 loss：
  - 位置、速度、总能量（动能 + 弹性 + 粘性 + 投影）；
  - 保证 loss 定义与文档说明一致。
- 在 API / CLI 层面提供「手工伴随可微模式」开关：
  - 可微模式走手工伴随；
  - 非可微模式保持当前阻断行为（防止误用 Taichi Tape 直接包 step）。
- 提供数值验证：
  - 与有限差分梯度在小规模场景中对齐（给定误差阈值）；
  - 梯度方向经简单下降试验验证合理。

### Non-Goals

- 不改变上游 Taichi AD 的实现；不指望 Taichi 在 1.7.4 中突然支持 P2G/G2P 的原子反向。
- 不追求首版的极致性能优化：
  - 首版以「正确性 + 清晰实现」优先；
  - 性能与显存优化留待后续迭代（如 checkpointing / mixed precision）。
- 不对以下量提供梯度（首版）：
  - 接触/摩擦参数：μ_s, μ_k, K_clear；
  - 所有硬约束式边界条件（例如粘着边界的刚性位置/速度限制）；
  - SPD 投影的 **精确** SVD 伴随（采用 STE 近似，有偏但简单）。
- 不尝试在一次迭代中同时优化「物理参数 + 接触几何 + 边界位置」等高敏感组合。

---

## High-level Design

### 1. 数据结构与前向数据流

每个时间步 n：

1. **P2G：粒子 → 网格**
   - 输入：粒子状态 \(x_p^n, v_p^n, F_p^n, C_p^n, m_p, V_p\) 等。
   - 输出：网格质量 \(M_I^n\)、动量 \(P_I^n\) 或等价的 `grid_v[I]`（未归一化时视为动量）。
   - APIC / MLS-MPM 仿射项、应力项、粘弹力等全部在这里累积到网格。

2. **Grid Ops：网格操作**
   - 包括：
     - 动量 → 速度：\(v_I^n = P_I^n / M_I^n\)（带 eps 防护）；
     - 外力（重力、体力）；
     - 网格内部力（如显式积分时的应力发散，具体视实现而定）；
     - 边界条件（Sticky / Slip / Separate / Bounce 等）。
   - 输出：更新后的网格速度 \(v_I^{n,\,\text{after}}\)。

3. **G2P：网格 → 粒子**
   - 利用 \(v_I^{n,\,\text{after}}\) 更新粒子速度、位置和 APIC 仿射：
     - \(v_p^{n+1}, x_p^{n+1}, C_p^{n+1}\)。
   - 使用权重 w_ip(x_p) 进行插值。

4. **F 更新 + SPD 投影 + Maxwell 内变量**
   - 更新形变梯度 \(F_p\)：
     - \(F_{p,\text{raw}}^{n+1} = (I + \Delta t\, C_p^{n+1}) F_p^n\)
     - \(F_{p,\text{spd}}^{n+1} = \text{SPD}(F_{p,\text{raw}}^{n+1})\)（数值稳定）
   - Maxwell 内变量 \( \bar{b}_e \) 更新：
     - \( \bar{b}_e^{n+1} = a\, \bar{b}_e^{n} + (1-a)\, \bar{b}_e^{n,\text{trial}} \)
     - \(a = \exp(-\Delta t / \tau)\)

5. **能量 / Loss 计算**
   - 动能：\(K = \sum_p \frac{1}{2} m_p \|v_p\|^2\)
   - 弹性能：\(E_{\text{elastic}} = \sum_p V_p \Psi(F_p, \theta)\)
   - 粘性/Maxwell 能量：视本构定义。
   - 总能量：\(E_{\text{total}} = K + E_{\text{elastic}} + E_{\text{visco}} + E_{\text{proj}}\)
   - 用户定义的 loss：
     - 位置/速度差；能量误差；时间序列上的组合等。

### 2. 反向总体策略

- 对 T 步仿真进行 **反向时间循环**（BPTT）：
  - 前向：`for n in 0..T-1: step_forward(n)`；同时在需要的位置存中间状态。
  - 反向：`for n in T-1..0: step_backward(n)`；自后向前清算梯度。
- 每个时间步的 backward 由以下子 kernel 组成：
  1. Loss backward → 粒子端梯度（x, v, F, 内变量, 材料参数）。
  2. F 更新 backward（含 SPD 投影 STE）。
  3. Maxwell 内变量 backward（如果开启）。
  4. G2P backward。
  5. Grid Ops backward（含 v = P/M、外力、边界条件）。
  6. P2G backward。
- 对材料参数（μ_i, α_i, τ 等）在整个时间窗口上累加梯度。

- 显存策略（首版）：
  - 粒子端：每步存储 \(x_p^n, v_p^n, F_p^n, C_p^n\)；若 Maxwell 梯度开启，则存 \(\bar{b}_e^n\)。
  - 网格端：尽量不存每步完整网格状态，而是重算 Grid Ops 所需量（或只存必须的，如 M_I^n）。
  - 通过配置开关决定是否存 Maxwell 内变量及 trial 状态。

---

## 核心伴随公式（包含 Grid Normalization / APIC / Maxwell / SPD）

下面给出的是 **骨架级别** 公式，实际实现可按代码结构展开。

> 记：  
> - \(L\)：最终 loss。  
> - 对任意量 q，有梯度 \(g_q = \partial L / \partial q\)。  
> - 粒子索引 p，网格索引 I。

### 1. P2G: Forward & Backward

#### 1.1 Forward（含 APIC 仿射项）

对每个粒子 p，遍历其邻域网格结点 I（3×3 或 4×4）：

- 权重：
  \[
  w_{ip} = w(x_p, x_I)
  \]

- APIC 仿射项：
  \[
  v_{\text{apic},pI} = v_p^n + C_p^n\,(x_I - x_p^n)
  \]

- 动量贡献（忽略应力项之前）：
  \[
  \Delta P_{Ip}^{(v)} = m_p\, v_{\text{apic},pI}
  \]

- 若考虑应力项（基于 Piola 应力 P）：
  \[
  \Delta P_{Ip}^{(\sigma)} = -\Delta t \, V_p\, P(F_p^n, \theta) B_{Ip}
  \]
  其中 \(B_{Ip}\) 由插值核梯度决定。

- 网格累积：
  \[
  P_I^n = \sum_p w_{ip} \left( \Delta P_{Ip}^{(v)} + \Delta P_{Ip}^{(\sigma)} \right)
  \]
  \[
  M_I^n = \sum_p w_{ip} m_p
  \]

这里的 P_I^n 可以实际存在 `grid_v[I]`，后续在 Grid Ops 中再除以 M_I^n 得到速度。

#### 1.2 Backward 的输入

P2G backward 的输入来自两部分：

1. **来自 G2P → Grid Ops backward 的结果**：
   - \(g_{P_I} = \partial L / \partial P_I^n\)
   - \(g_{M_I} = \partial L / \partial M_I^n\)

2. **如果还有其它 Grid 量（比如能量密度）依赖 P_I, M_I**，也要先归并到 g_P_I, g_M_I。

P2G backward 输出粒子端梯度：

- \(g_{v_p^n}, g_{C_p^n}, g_{x_p^n}, g_{F_p^n}, g_{m_p}, g_\theta\)。

#### 1.3 对 v_p, C_p, m_p 的伴随

对动量贡献 \(\Delta P_{Ip}^{(v)} = m_p \left( v_p^n + C_p^n (x_I - x_p^n) \right)\)：

- 对 v_p：
  \[
  \frac{\partial \Delta P_{Ip}^{(v)}}{\partial v_p^n} = m_p I
  \]
  \[
  g_{v_p^n} \mathrel{+}= \sum_I w_{ip}\, m_p\, g_{P_I}
  \]

- 对 C_p（矩阵）：
  \[
  \frac{\partial \Delta P_{Ip}^{(v)}}{\partial C_p^n} = m_p (x_I - x_p^n)
  \]
  更准确是：
  \[
  \Delta P_{Ip}^{(v)} = m_p \left( v_p^n + C_p^n d_{Ip} \right),\quad d_{Ip}=x_I-x_p^n
  \]
  则：
  \[
  g_{C_p^n} \mathrel{+}= \sum_I w_{ip}\, m_p\, g_{P_I} \otimes d_{Ip}^T
  \]

- 对质量 m_p：
  \[
  \frac{\partial \Delta P_{Ip}^{(v)}}{\partial m_p} = v_{\text{apic},pI}
  \]
  \[
  g_{m_p}^{(P)} \mathrel{+}= \sum_I w_{ip} \left(g_{P_I} \cdot v_{\text{apic},pI}\right)
  \]
  同时质量还直接出现在 \(M_I^n\) 中：
  \[
  g_{m_p}^{(M)} \mathrel{+}= \sum_I w_{ip}\, g_{M_I}
  \]
  最终：
  \[
  g_{m_p} = g_{m_p}^{(P)} + g_{m_p}^{(M)}
  \]

#### 1.4 对 x_p 的伴随（权重 + 仿射项 + 应力）

x_p 对 P_I, M_I 的影响来自三部分：

1. 权重 w_ip(x_p)；
2. APIC 仿射项中的 \(x_I - x_p\)（Gemini 提醒的那项）；
3. 若应力项中的 B_{Ip} 也依赖 x_p（通过核的空间导数）。

**(A) 权重项**

\[
P_I^n = \sum_p w_{ip}(x_p) \Delta P_{Ip},\quad
M_I^n = \sum_p w_{ip}(x_p) m_p
\]

对 x_p：

\[
g_{x_p}^{(w)} \mathrel{+}= \sum_I \left(\frac{\partial w_{ip}}{\partial x_p}\right)
   \left( \Delta P_{Ip} \cdot g_{P_I} + m_p g_{M_I} \right)
\]

其中 \(\partial w_{ip} / \partial x_p\) 由核函数梯度给出。

**(B) APIC 仿射项中的 x_p**

\[
v_{\text{apic},pI} = v_p^n + C_p^n (x_I - x_p^n)
\]

\[
\frac{\partial v_{\text{apic},pI}}{\partial x_p^n} = -C_p^n
\]

所以：

\[
\frac{\partial \Delta P_{Ip}^{(v)}}{\partial x_p^n} = m_p (-C_p^n)
\]

反向：

\[
g_{x_p}^{(\text{affine})} \mathrel{+}= \sum_I w_{ip}\, m_p\, (-C_p^n)^T g_{P_I}
\]

这就是之前文档没有写明的那一项。

**(C) 应力项中的 x_p**

若应力项使用 B-spline 核的空间导数，通常 B_{Ip} 依赖 x_p 或 x_I，不再展开细节，这一部分记为：

\[
g_{x_p}^{(\sigma)} \mathrel{+}= \text{(由 } \partial B_{Ip} / \partial x_p, P(F_p^n, \theta), g_{P_I}\text{ 得到)}
\]

最终：

\[
g_{x_p^n} = g_{x_p}^{(w)} + g_{x_p}^{(\text{affine})} + g_{x_p}^{(\sigma)}
\]

#### 1.5 对 F_p / 材料参数的伴随

应力项大致为：

\[
\Delta P_{Ip}^{(\sigma)} = -\Delta t\, V_p\, P(F_p^n,\theta)\, B_{Ip}
\]

对 F_p：

- 由本构：
  \[
  P = \frac{\partial \Psi(F_p^n, \theta)}{\partial F_p^n}
  \]
- 反向从 g_{P_I} 聚合到粒子：
  \[
  g_{P_p} = -\Delta t\, V_p \sum_I w_{ip} B_{Ip}^T g_{P_I}
  \]
- 再由 \(\frac{\partial P}{\partial F}\) 把 g_P_p 转到 g_F_p（可按实现写成代码级链式）。

对材料参数（以 Ogden μ_i 为例）：

- Ogden 能量：
  \[
  \Psi_{\text{Ogden}} = \sum_i \frac{\mu_i}{\alpha_i}
  (\bar{\lambda}_1^{\alpha_i} + \bar{\lambda}_2^{\alpha_i} + \bar{\lambda}_3^{\alpha_i} - 3) + \text{bulk}(J)
  \]
- 则：
  \[
  \frac{\partial \Psi}{\partial \mu_i} =
  \frac{1}{\alpha_i} (\bar{\lambda}_1^{\alpha_i} + \bar{\lambda}_2^{\alpha_i} + \bar{\lambda}_3^{\alpha_i} - 3)
  \]
- 若 loss 中有能量项 \(L_E\)，则：
  \[
  g_{\mu_i} \mathrel{+}= \sum_p V_p
    \frac{\partial \Psi}{\partial \mu_i}
    \cdot \frac{\partial L}{\partial \Psi_p}
  \]

---

### 2. Grid Ops: v = P/M、外力与边界条件

#### 2.1 Forward

最小版本（忽略外力）：

```python
# 前向（简化）
for I in grid:
    if M_I > eps:
        v_I = P_I / M_I
    else:
        v_I = 0
    # 加重力
    v_I += dt * gravity
    # 边界条件
    v_I = apply_boundary(v_I, I)
````

其中 `apply_boundary` 可能执行：

*   Sticky：某些方向速度置零；
*   Slip：法向速度置零，切向保留；
*   Bounce：法向速度取相反数；
*   Separate：将粒子/网格从非法交叠中移除等。

#### 2.2 Backward：核心是 v = P/M 的伴随

设 g\_v\_I 是来自 G2P backward 的梯度（对 **经过 Grid Ops 后** 的 v\_I）。

**(A) 重力**

*   前向：`v_I_after_gravity = v_I + dt * gravity`
*   反向：
    $$
    g_{v_I}^{(\text{before gravity})} = g_{v_I}^{(\text{after gravity})}
    $$

**(B) 边界条件 apply\_boundary**

需要对每种 BC 约定梯度策略：

*   Sticky：
    *   若法向速度被硬置零，例如 `v.y = 0` 且 `v.x,v.z` 保留：
        *   反向：`g_v.y = 0`（梯度截断），`g_v.x, g_v.z` 原样传递。
*   Slip：
    *   法向分量投影为 0，切向保留：
        *   反向：对法向梯度截断，对切向照常传递。
*   Bounce：
    *   若 `v_n = -v_n_orig`：
        *   反向：`g_v_n_orig = -g_v_n`。
*   其它：
    *   可按线性/非线性函数 `v_after = BC(v_before)` 推导 J = ∂BC/∂v，再用 `g_before += J^T g_after`。

首版可以选取比较保守的策略，例如对「刚性墙」的法向梯度全部截断，避免“穿墙”梯度。

**(C) v = P/M 的伴随（致命遗漏部分的修补）**

假设经过 BC 和重力逆推之后，我们拿到：

*    $g_{v_I^{\text{raw}}}$ ：对「归一化后的速度」的梯度。

Forward：

$$
v_I^{\text{raw}} = \begin{cases} P_I^n / M_I^n & M_I^n > \epsilon, \\ 0 & \text{否则} \end{cases}
$$

忽略 eps 分支（或在实现中把 eps 分支梯度截断），在  $M_I^n > \epsilon$  的区域有：

*   对 P\_I：
    $$
    \frac{\partial v_I}{\partial P_I} = \frac{1}{M_I^n} I \Rightarrow g_{P_I} \mathrel{+}= \frac{1}{M_I^n} g_{v_I^{\text{raw}}}
    $$
*   对 M\_I：
    $$
    v_I = \frac{P_I}{M_I} \Rightarrow \frac{\partial v_I}{\partial M_I} = -\frac{P_I}{M_I^2}
    $$
    $$
    g_{M_I} \mathrel{+}= g_{v_I^{\text{raw}}} \cdot \left(-\frac{P_I}{M_I^2}\right) = -\frac{(g_{v_I^{\text{raw}}} \cdot P_I)}{M_I^2} = -\frac{(g_{v_I^{\text{raw}}} \cdot v_I)}{M_I}
    $$

这一步就是 Gemini 指出的「Grid Normalization Backward」，必须显式实现，否则 g\_M\_I = 0，会丢掉 MASS 分布对速度（进而对 loss）的影响。

* * *

### 3\. G2P: Forward & Backward

#### 3.1 Forward

典型 MLS-MPM / APIC：

$$
v_p^{n+1} = \sum_I w_{ip}(x_p^n)\, v_I^{\text{after}}
$$
 
$$
C_p^{n+1} = \frac{4}{\Delta x^2} \sum_I w_{ip}(x_p^n)\, v_I^{\text{after}} \otimes (x_I - x_p^n)^T
$$
 
$$
x_p^{n+1} = x_p^n + \Delta t\, v_p^{n+1}
$$

（系数如 4/Δx² 可按真实实现调整。）

#### 3.2 Backward

已知：

*   g\_{x\_p^{n+1}}, g\_{v\_p^{n+1}}, g\_{C\_p^{n+1}}。

**(A) x\_p 更新**

$$
x_p^{n+1} = x_p^n + \Delta t\, v_p^{n+1} \Rightarrow \begin{cases} g_{v_p^{n+1}} \mathrel{+}= \Delta t\, g_{x_p^{n+1}} \\ g_{x_p^n} \mathrel{+}= g_{x_p^{n+1}} \end{cases}
$$

**(B) v\_p 插值**

$$
v_p^{n+1} = \sum_I w_{ip}(x_p^n) v_I^{\text{after}}
$$
*   对 grid v：
    $$
    g_{v_I^{\text{after}}} \mathrel{+}= \sum_p w_{ip}(x_p^n)\, g_{v_p^{n+1}}
    $$
*   对 x\_p^n（权重导数）：
    $$
    g_{x_p^n}^{(v)} \mathrel{+}= \sum_I \left(\frac{\partial w_{ip}}{\partial x_p^n}\right) \left( v_I^{\text{after}} \cdot g_{v_p^{n+1}} \right)
    $$

**(C) C\_p 插值**

$$
C_p^{n+1} = \kappa \sum_I w_{ip}(x_p^n)\, v_I^{\text{after}} \otimes d_{Ip}^T, \quad d_{Ip} = x_I - x_p^n,\ \kappa = 4/\Delta x^2
$$
*   对 grid v：
    $$
    g_{v_I^{\text{after}}} \mathrel{+}= \kappa \sum_p w_{ip}\, \left( g_{C_p^{n+1}} : d_{Ip}^T \right)
    $$
*   对 x\_p^n（既来自 w\_ip，也来自 d\_{Ip}）：
    *   权重导数项（类似上节）：
        $$
        g_{x_p^n}^{(C,w)} \mathrel{+}= \kappa \sum_I \left(\frac{\partial w_{ip}}{\partial x_p^n}\right) \left( g_{C_p^{n+1}} : (v_I^{\text{after}} \otimes d_{Ip}^T) \right)
        $$
    *   d\_{Ip} = x\_I - x\_p^n 的显式导数：
        $$
        \frac{\partial d_{Ip}}{\partial x_p^n} = -I
        $$
        $$
        g_{x_p^n}^{(C,d)} \mathrel{+}= -\kappa \sum_I w_{ip}\, \left( g_{C_p^{n+1}}^T v_I^{\text{after}} \right)
        $$
    最终：
    $$
    g_{x_p^n} \mathrel{+}= g_{x_p^n}^{(v)} + g_{x_p^n}^{(C,w)} + g_{x_p^n}^{(C,d)}
    $$

* * *

### 4\. F 更新 + SPD 投影（STE）

#### 4.1 Forward

$$
F_{\text{raw},p}^{n+1} = (I + \Delta t\, C_p^{n+1}) F_p^n
$$
 
$$
F_p^{n+1} = \text{SPD}\left(F_{\text{raw},p}^{n+1}\right)
$$
*   SPD 处理可能基于 polar 分解 / SVD，对 F 做裁剪保证对称正定，防止数值爆炸。

#### 4.2 Backward

已知 g\_{F\_p^{n+1}}。

1.  SPD（STE 策略）：
    *   我们采用**直通估计**：
        $$
        g_{F_{\text{raw},p}^{n+1}} \mathrel{+}= g_{F_p^{n+1}}
        $$
    *   即忽略 SPD 对梯度的影响，只在前向做稳定性限制。
2.  F 更新：
    $$
    F_{\text{raw},p}^{n+1} = (I + \Delta t\, C_p^{n+1}) F_p^n
    $$
    *   对 C\_p^{n+1}：
        $$
        g_{C_p^{n+1}} \mathrel{+}= \Delta t\, g_{F_{\text{raw},p}^{n+1}} F_p^{n,T}
        $$
    *   对 F\_p^n：
        $$
        g_{F_p^n} \mathrel{+}= (I + \Delta t\, C_p^{n+1})^T g_{F_{\text{raw},p}^{n+1}}
        $$

#### 4.3 SPD 近似的风险说明

*   当 F 的前向数值经常落在「非法形变」（如 det(F) < 0）区域时：
    *   SPD 会强行把其投到 SPD 锥上；
    *   真实梯度应当强烈惩罚这种形变；
    *   但 STE 会给出 “好像 SPD 没存在过” 的梯度，导致优化器在这种“坏区域”附近收敛困难或震荡。
*   对策：
    *   记录 SPD 投影被触发的次数 / 粒子比例，作为训练稳定性监控指标；
    *   在文档中明确：这是一个**工程近似**，如果 SPD 触发频繁，应从模型/损失设计调整，而不是单纯责怪梯度。

* * *

### 5\. Maxwell 粘弹性内变量伴随

#### 5.1 Forward

设  $\bar{b}_e$  为 Maxwell 内变量（如等效弹性应变测度）：

$$
a = \exp(-\Delta t/\tau)
$$
 
$$
\bar{b}_e^{n+1} = a\,\bar{b}_e^n + (1-a)\,\bar{b}_e^{n,\text{trial}}
$$

其中 trial 值由当前 F 或其它量计算得到。

#### 5.2 Backward

已知 g\_{\\bar{b}\_e^{n+1}}：

*   对  $\bar{b}_e^n$ ：
    $$
    g_{\bar{b}_e^n} \mathrel{+}= a\, g_{\bar{b}_e^{n+1}}
    $$
*   对  $\bar{b}_e^{n,\text{trial}}$ ：
    $$
    g_{\bar{b}_e^{n,\text{trial}}} \mathrel{+}= (1-a)\, g_{\bar{b}_e^{n+1}}
    $$
*   对  $\tau$ （若需要求梯度）：
    $$
    a = e^{-\Delta t / \tau} \Rightarrow \frac{\partial a}{\partial \tau} = \frac{\Delta t}{\tau^2} e^{-\Delta t / \tau}
    $$
    $$
    \frac{\partial \bar{b}_e^{n+1}}{\partial \tau} = \frac{\partial a}{\partial \tau} (\bar{b}_e^n - \bar{b}_e^{n,\text{trial}})
    $$
    $$
    g_\tau \mathrel{+}= \left\langle g_{\bar{b}_e^{n+1}}, \frac{\partial \bar{b}_e^{n+1}}{\partial \tau} \right\rangle_F
    $$
*   存储策略：
    *   默认配置 `maxwell_needs_grad = false`：
        *   不存储  $\bar{b}_e$  时间序列，只计算前向；
        *   反向中对  $\bar{b}_e, \tau$  的梯度全部视为 0。
    *   如开启：
        *   每步存  $\bar{b}_e^n$  和  $\bar{b}_e^{n,\text{trial}}$  或者使用 checkpoint 重算策略。

* * *

### 6\. Loss 与梯度精度验证

#### 6.1 Loss 类型

*   位置 loss：
    $$
    L_x = \frac{1}{2} \|x_{\text{sim}} - x_{\text{target}}\|^2
    $$
*   速度 loss：
    $$
    L_v = \frac{1}{2} \|v_{\text{sim}} - v_{\text{target}}\|^2
    $$
*   总能量 loss：
    $$
    L_E = \frac{1}{2} (E_{\text{total}} - E_{\text{target}})^2
    $$

整体 loss 可为线性组合：

$$
L = \lambda_x L_x + \lambda_v L_v + \lambda_E L_E
$$

#### 6.2 梯度数值验证标准

对某个参数向量  $\theta$ （可以是 x₀, v₀, F₀ 或某个材料参数）：

*   手工伴随得到 g\_auto；
*   有限差分得到 g\_fd。

评估指标：

*   相对误差：
    $$
    \text{rel\_err} = \frac{\|g_{\text{auto}} - g_{\text{fd}}\|_2} {\max(1, \|g_{\text{fd}}\|_2)}
    $$
*   方向余弦相似度：
    $$
    \text{cos\_sim} = \frac{g_{\text{auto}}\cdot g_{\text{fd}}} {\|g_{\text{auto}}\|_2 \|g_{\text{fd}}\|_2}
    $$

要求（默认）：

*   rel\_err < 1e-4
*   cos\_sim > 0.99

* * *

### 7\. 显存占用估算

假设：

*   粒子数：N\_p
*   时间步数：T
*   float32（4 bytes）

每粒子每步存储：

*   x\_p：3
*   v\_p：3
*   F\_p：9
*   C\_p：9
*   Maxwell 开启时： $\bar{b}_e$ ：9

则：

*   无 Maxwell：n\_float = 3 + 3 + 9 + 9 = 24
*   有 Maxwell：n\_float = 33

显存估算：

$$
M_{\text{bytes}} \approx N_p \cdot T \cdot n_{\text{float}} \cdot 4
$$

例子：

*   N\_p = 1e5, T = 100：
    *   无 Maxwell：
        $$
        M \approx 1e5 \cdot 100 \cdot 24 \cdot 4 = 9.6\times 10^8\ \text{bytes} \approx 0.89\ \text{GB}
        $$
    *   有 Maxwell（n\_float = 33）：
        $$
        M \approx 1e5 \cdot 100 \cdot 33 \cdot 4 \approx 1.32\times 10^9\ \text{bytes} \approx 1.23\ \text{GB}
        $$

实际还需加上网格临时 buffer、Taichi 内部结构等。

* * *

实现策略 / 接口集成
-----------

*   提供 `enable_manual_adjoint` 标志：
    *   若 False：保持当前行为，Tape 包裹 step 时抛出 `NotImplementedError`，阻断梯度；
    *   若 True：
        *   调用新的 `mpm_forward_tape` 接口：
            *   内部执行 multi-step 前向并存储必要状态；
            *   提供 `backward(loss)` 方法执行手工伴随。
*   在 Python 封装中把材料参数与初始状态视为「输入变量」，暴露 gradient：
    *   `solver.x0.grad`, `solver.v0.grad`, `solver.F0.grad`, `solver.material.mu.grad` 等；
    *   摩擦参数在文档中标记为 `no_grad`。
*   CLI：
    *   `--enable-manual-adjoint` 打开；
    *   `--maxwell-needs-grad` 控制 Maxwell 内变量梯度；
    *   `--loss-config` 控制 loss 组合。

* * *

Risks / Open Questions
----------------------

*   SPD STE 梯度偏差：
    *   可能导致大形变场景中 loss 震荡；
    *   需要监控 SPD 触发率，并在文档中明确风险。
*   边界条件与摩擦模型的非光滑性：
    *   在切换点附近梯度不稳定属物理模型特性；
    *   可以考虑在更高层通过正则化几何/损失来缓解。
*   显存与性能：
    *   是否需要在首版就引入 checkpointing；
    *   是否需要针对特定场景实现更紧凑的状态存储。
*   AD 使用体验：
    *   如何以最小 API 变动给用户提供自然的 `loss.backward()` 风格接口。

````

---

## 📐 spec.md（完整更新版）

```markdown
# Spec: 手工伴随支持 MPM 可微仿真

本文件定义在现有 MPM 求解器上增加「手工伴随」以支持可微仿真的需求。

---

## 1. 背景与目标

### 1.1 背景

- 当前基于 Taichi 1.7.4 的 MPM 实现：
  - 前向使用包含原子 scatter/gather 的 P2G/G2P kernel；
  - Taichi AD 无法对这些 kernel 执行反向，Tape 包裹 `solver.step()` 时会报错；
  - 为避免崩溃，现有接口在进入可微模式时直接抛出 `NotImplementedError`。

### 1.2 目标

- 通过手工伴随为 MPM 提供稳定的梯度接口，使得：
  - 材料参数（如 Ogden/Maxwell 参数）可被优化；
  - 初始粒子状态（x₀, v₀, F₀）可用作可微控制量；
  - 在小规模验证场景中，梯度与有限差分结果对齐。

---

## 2. 功能性需求（Functional Requirements）

### FR1: 手工伴随支持可微 MPM

系统必须（SHALL）为以下模块提供手工伴随 forward/backward kernel：

- P2G（含 APIC 仿射项和应力项）；
- Grid Ops（包括动量-速度归一化 v = P/M、外力和边界条件）；
- G2P（含 v 与 C 的插值）；
- F 更新（含 SPD 投影，使用 STE）；
- Maxwell 内变量更新（可选，受配置控制）；
- 能量求和（动能、弹性、粘性、投影等）。

#### Scenario: 可微模式下运行

- **WHEN** 用户在「可微模式」运行 MPM 仿真并调用梯度接口；
- **THEN** 系统使用手工伴随路径返回材料参数与初始状态的非零梯度；
- **AND** 不触发 Taichi AD 的 “Not supported” 错误。

---

### FR2: 可微 Loss 集与定义一致性

系统必须（SHALL）提供以下可微 loss 类型：

- 位置 loss；
- 速度 loss；
- 总能量 loss（动能 + 弹性 + 粘性 + 投影）。

并要求：

- loss 的数学定义在文档中明确；
- 实现的计算路径与文档保持一致。

#### Scenario: 总能量 loss

- **WHEN** 用户配置启用总能量 loss；
- **THEN** 系统按 Spec 中给出的能量定义计算总能量，并对相关状态（v, F, 内变量）回传梯度。

---

### FR3: Grid Normalization Backward（v = P/M）

系统必须（SHALL）显式实现网格动量-质量归一化操作的反向过程：

- 给定前向：

  - \( v_I = P_I / M_I \)（在 \(M_I > \epsilon\) 区域）
- 反向必须满足：

  - \( \partial L / \partial P_I = g_{v_I} / M_I \)
  - \( \partial L / \partial M_I = - (g_{v_I} \cdot v_I) / M_I \)

#### Scenario: Grid Ops Backward

- **WHEN** 在反向过程中从 G2P backward 得到 \(g_{v_I}\)；
- **THEN** 系统必须首先通过 Grid Ops backward（包括归一化）得到 \(g_{P_I}\) 与 \(g_{M_I}\)，再交由 P2G backward。

---

### FR4: APIC 仿射项对位置的导数

系统必须（SHALL）在 P2G backward 中包含 APIC 仿射项 \(C_p(x_I - x_p)\) 对粒子位置 x_p 的导数：

- 对于动量项：
  \[
  \Delta P_{Ip}^{(v)} = m_p \left[ v_p + C_p(x_I - x_p) \right]
  \]
- 反向时：
  \[
  \frac{\partial \Delta P_{Ip}^{(v)}}{\partial x_p} = -m_p C_p
  \]

#### Scenario: APIC 梯度计算

- **WHEN** 系统在场景中启用了 APIC / MLS-MPM 仿射动量；
- **THEN** P2G backward 中必须包含上述对 x_p 的显式贡献（除权重导数之外），以避免粒子位置梯度缺失。

---

### FR5: SPD 投影策略（STE）

系统必须（SHALL）在形变梯度更新中对 SPD 投影采用直通估计器（STE）策略：

- Forward：执行 SPD 投影以保证 F 的数值稳定；
- Backward：视 SPD 为恒等映射，即：
  - \(g_{F_{\text{raw}}} \mathrel{+}= g_{F_{\text{spd}}}\)

并在文档中：

- 明确这是一个有偏梯度近似；
- 提示在 SPD 投影触发频繁时可能导致 loss 震荡。

#### Scenario: SPD 投影开启

- **WHEN** 仿真过程中任意粒子触发 SPD 投影；
- **THEN** 对应的反向必须按 STE 策略传递梯度；
- **AND** 系统应支持统计 SPD 投影触发的粒子比例，用于调试与监控。

---

### FR6: Maxwell 内变量伴随（可配置）

系统必须（SHALL）为 Maxwell 粘弹性内变量提供可配置的伴随实现：

- 在配置中提供开关（例如 `maxwell_needs_grad`）；
- 开启时：
  - 存储足够的 \(\bar{b}_e\) 状态（或使用 checkpoint）；
  - 在 backward 中计算 \(\partial L / \partial \bar{b}_e\) 与 \(\partial L / \partial \tau\)。
- 关闭时：
  - 不存储 \(\bar{b}_e\) 时间序列；
  - 对 Maxwell 相关参数的梯度视为 0。

#### Scenario: Maxwell 梯度开启

- **WHEN** 用户在配置中启用 Maxwell 梯度；
- **THEN** 系统必须在 backward 中正确传播 \(\bar{b}_e\) 与 \(\tau\) 的梯度，并通过测试用例验证数值正确性。

---

### FR7: 接触/摩擦与边界条件的可微范围

系统必须（SHALL）在文档与实现中明确接触/摩擦和边界条件的可微范围：

- 对状态变量（x, v, F）：
  - 在当前分支下提供局部可微梯度；
  - 对不可导分支（如硬截断）需给出截断策略（例如梯度置零）。
- 对摩擦参数（μ_s, μ_k, K_clear）：
  - 首版不提供梯度（Non-Goal）；
  - 尝试对这些量求导时，接口必须给出明确警告或错误。

#### Scenario: 穿墙/摩擦参数优化

- **WHEN** 用户试图优化摩擦参数或通过梯度将粒子推过刚性边界；
- **THEN** 系统必须保持行为与文档一致：不对摩擦参数求导；对严格硬约束边界的法向速度梯度进行截断或约定映射。

---

### FR8: 可微接口与模式切换

系统必须（SHALL）在 API/CLI 中区分以下模式：

- 非可微模式：
  - 对 MPM 的梯度请求直接阻断（例如抛出 `NotImplementedError`）；
  - 不启用任何手工伴随状态存储。
- 手工伴随可微模式：
  - 使用手工伴随实现梯度；
  - 不直接调用 Taichi Tape 对 P2G/G2P kernel 做 AD。

#### Scenario: 模式切换

- **WHEN** 用户通过 CLI/配置切换可微模式；
- **THEN** 系统必须清晰反映当前模式，并在不支持场景下给出可预期的错误信息。

---

## 3. 非功能需求（Non-Functional Requirements）

### NFR1: 梯度精度

在小规模验证场景中，手工伴随梯度与有限差分梯度必须（SHALL）满足：

- 相对误差：
  - \(\text{rel\_err} < 1e-4\)
- 梯度方向余弦相似度：
  - \(\text{cos\_sim} > 0.99\)

#### Scenario: 数值梯度对比

- **WHEN** 对任一被标记为可微的参数执行有限差分梯度对比；
- **THEN** 上述两个指标均需满足，否则视为实现缺陷。

---

### NFR2: 显存占用可估算

系统必须（SHALL）在设计文档中提供手工伴随前向状态存储的显存估算公式，并在至少一个典型场景中验证测量值与估算值的同量级性（例如误差在 2x 内）。

---

### NFR3: 性能与鲁棒性

- 反向阶段可能比前向慢，但整体运行必须在「可接受」范围（具体值视项目标准决定）；
- 不允许因为手工伴随引入新的数值发散：
  - 若加入 SPD 投影、边界条件、摩擦后仍导致反向阶段数值爆炸，视为实现问题。

---

## 4. 验证与测试

### TV1: 单步线性场景验证

- 场景：
  - 少量粒子（<100）、均匀网格；
  - 简单线性弹性或无本构，仅位置/速度 loss。
- 检查：
  - 各模块 forward/backward 是否能运行；
  - 梯度非零且方向合理。

### TV2: Grid Normalization Backward 验证

- 场景：
  - 构造只影响单个网格结点的粒子分布；
  - 对初始位置 x₀ 的 loss 梯度通过有限差分与手工伴随对比。
- 期望：
  - 证明质量分布通过 v = P/M 对 loss 的影响在梯度中被捕获（g_M 不为 0）。

### TV3: APIC 仿射项对位置的梯度验证

- 场景：
  - 开启 APIC 仿射项；
  - 设计带明显旋转/剪切的变形。
- 期望：
  - 若去掉 APIC 中 x_p 的那一项导数，梯度对比测试明显失败；
  - 加上后满足 NFR1 中的误差与方向要求。

### TV4: SPD 投影行为观测

- 场景：
  - 设定极端的形变路径，使部分粒子频繁触发 SPD；
- 期望：
  - 梯度可以计算但可能在这些场景中不再符合严格的数值梯度对齐；
  - 系统提供 SPD 触发比例统计；文档解释此时损失震荡的原因。

### TV5: Maxwell 内变量梯度验证（若启用）

- 场景：
  - 启用 Maxwell 本构，设置简单粘弹响应；
  - 对 τ 或某个相关参数做有限差分 vs 手工伴随。
- 期望：
  - 满足 NFR1 的梯度精度要求，或在文档中解释差异来源。

---

## 5. 兼容性与迁移

- 不要求修改现有前向 API；
- 允许在内部添加一个新的「可微求解」入口（如 `solve_with_gradients`）；
- 保持旧的 `run_with_gradients` 等接口的行为一致（例如继续抛出未实现错误，但提示用户使用新的接口）。

````

* * *

✅ tasks.md（完整更新版）
-----------------

```markdown
## 1. 实施

- [ ] 1.1 伴随设计与文档
  - [ ] 绘制完整前向数据流图：P2G → Grid Ops → G2P → F 更新 → Maxwell → Energies。
  - [ ] 在 design.md 中补充/确认：
    - Grid Normalization（v = P/M）的 forward/backward；
    - APIC 仿射项中对 x_p 的导数（-C_p）；
    - SPD 投影的 STE 策略与风险说明；
    - Maxwell 内变量递推与伴随公式；
    - 边界条件与摩擦的可微范围与梯度截断策略；
    - 显存估算与梯度精度标准。
  - [ ] 校对 spec.md 中 FR/NFR 是否与设计一致。

- [ ] 1.2 Grid Ops Backward 实现
  - [ ] 为 v = P/M、外力（如重力）、边界条件实现 backward kernel。
  - [ ] 确保：
    - g_P, g_M 的计算满足 spec 中 FR3 的公式；
    - 各类边界条件的梯度映射（或截断）与 design.md 一致。

- [ ] 1.3 p2g_backward kernel
  - [ ] 实现从 (g_P_I, g_M_I) 回传到粒子端 (g_x_p, g_v_p, g_C_p, g_F_p, g_m_p, g_θ)。
  - [ ] 覆盖：
    - 权重 w_ip(x_p) 的导数；
    - APIC 仿射项中 (x_I - x_p) 对 x_p 的显式导数（-C_p）；
    - 应力项对 F_p 与材料参数的梯度。
  - [ ] 避开原子反向限制：采用块归约或其他无原子写的策略。

- [ ] 1.4 g2p_backward kernel
  - [ ] 实现对 v_p^{n+1}, x_p^{n+1}, C_p^{n+1} 的 backward：
    - 对 grid v 的梯度累积；
    - 对 x_p^n 的梯度（来自权重与 d_{Ip} = x_I - x_p）。
  - [ ] 与 p2g_backward 共享权重导数实现。

- [ ] 1.5 update_F_backward kernel（含 SPD）
  - [ ] 实现 F 更新：
    - F_raw = (I + dt * C) F_old；
    - F_new = SPD(F_raw)；
  - [ ] 使用 STE：
    - backward 中将 g_F_new 直接加到 g_F_raw。
  - [ ] 将 g_F_raw 分解到 g_C 和 g_F_old。

- [ ] 1.6 reduce_energies_backward kernel
  - [ ] 实现动能、弹性能、粘性/Maxwell 能量、投影能量求和的 backward：
    - 从 loss 对总能量的梯度回传到 v_p, F_p, 内变量等。
  - [ ] 保证能量定义与 spec.md 一致。

- [ ] 1.7 Maxwell 内变量伴随（可选）
  - [ ] 在配置打开时：
    - 存储必要的 \(\bar{b}_e^n\) 与 \(\bar{b}_e^{n,\text{trial}}\)；
    - 实现其 backward（包括对 τ 的梯度）。
  - [ ] 在配置关闭时：
    - 不存储 \(\bar{b}_e\) 时间序列；
    - Maxwell 相关梯度视为 0。

- [ ] 1.8 梯度累积与接口集成
  - [ ] 在 Python/Taichi wrapper 中实现 multi-step 前向 + backward：
    - 反向按时间步逆序执行；
    - 在整个时间窗口上累加材料参数与初始状态梯度。
  - [ ] 提供 public API：
    - 例如 `solve_with_manual_adjoint(config, loss_fn)`；
    - 返回 loss 与可查询的梯度字段。
  - [ ] 保持旧的 `run_with_gradients` 接口仍然显式拒绝执行，并提示使用新接口。

- [ ] 1.9 Loss 集成与 CLI
  - [ ] 实现位置/速度/总能量等 loss 的 kernel；
  - [ ] 在 CLI/配置中支持：
    - 选择 loss 组合；
    - 启用/禁用总能量 loss；
    - 打印/记录 loss 分解（K, E_elastic, E_visco, E_proj）。

---

## 2. 验证

- [ ] 2.1 单步场景功能验证
  - [ ] 构造简单场景（少量粒子 + 单步仿真）；
  - [ ] 检查：
    - forward + backward 能顺利运行；
    - 梯度非零且沿负梯度方向做一次参数更新后 loss 明显下降。

- [ ] 2.2 数值梯度对比（核心）
  - [ ] 针对以下量分别做有限差分 vs 手工伴随对比：
    - 粒子初始位置 x₀；
    - 初始速度 v₀；
    - 代表性的 Ogden 参数 μ_i；
    - 若启用 Maxwell 梯度，则包含 τ。
  - [ ] 对每个被测参数，计算：
    - 相对误差 rel_err；
    - 方向余弦 cos_sim。
  - [ ] 验证满足 spec.md 中 NFR1 的阈值。

- [ ] 2.3 Grid Normalization Backward 专项测试
  - [ ] 构造只改变粒子质量分布的扰动（保持 P 不变）；
  - [ ] 检测：扰动应通过 v = P/M 影响 loss，并在手工伴随梯度中体现为非零的 g_M 和 g_x。

- [ ] 2.4 APIC 仿射项梯度测试
  - [ ] 在启用 APIC 的场景下，对 x_p 做有限差分；
  - [ ] 对比：
    - 若关闭 -C_p 项，梯度误差显著；
    - 打开后误差回到正常范围。

- [ ] 2.5 SPD 投影视觉与统计
  - [ ] 记录并输出：
    - 每步 SPD 投影触发的粒子比例；
    - loss 曲线与梯度范数曲线。
  - [ ] 在极端形变场景下人工检查是否出现严重震荡，并在文档中记录观察结果。

- [ ] 2.6 Maxwell 内变量测试（若启用）
  - [ ] 设计简单粘弹响应测试；
  - [ ] 对 τ 做有限差分 vs 手工伴随对比。

- [ ] 2.7 性能与显存记录
  - [ ] 在典型场景（例如 N_p ~ 1e5, T ~ 100）中测量：
    - 前向耗时；
    - 反向耗时；
    - 峰值显存。
  - [ ] 与 design.md 中的显存估算进行对比，记录误差范围。

- [ ] 2.8 CLI / 配置回归
  - [ ] 验证：
    - 非可微模式下，旧接口仍然给出清晰的未实现提示；
    - 可微模式下，新的接口能够运行所有上述测试用例；
    - 配置错误（如对摩擦参数请求梯度）时，给出明确错误/警告。

```

* * *

如果你愿意，下一步我可以**按你现有 Taichi 代码的 API**，帮你直接写一版 `p2g_backward` / `g2p_backward` 的伪代码或接近可用的 Taichi kernel 雏形（包括 Grid Ops Backward 的 v=P/M + BC 部分），你就可以边看文档边对着实现。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)