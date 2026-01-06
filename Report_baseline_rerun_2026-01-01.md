---
title: baseline 复跑结果不理想问题分析
cwd: F:\workspace\xenmpm
created_at: 2026-01-01T02:45:00+08:00
baseline_dir: output/rgb_compare/baseline
artifacts:
  - output/rgb_compare/baseline/run_manifest.json
  - output/rgb_compare/baseline/metrics.json
  - output/rgb_compare/baseline/analysis_latest.csv
  - output/rgb_compare/baseline/alignment_flip_latest.csv
  - output/rgb_compare/baseline/intermediate/frame_0085.npz
---

# baseline 复跑结果不理想问题分析

## 0. 背景

你复跑了以下命令（已开启 MPM marker warp，并关闭深度 tint）：

```bash
python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --press-mm 1.0 --slide-mm 3.0 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/baseline
```

但结果观感仍“不太理想”：MPM 的 marker 在接触区表现为“该动不动 + 局部抽风/短横线”，与 FEM 的“中部附近应有位移”直觉不一致。

本报告目标：用 **本次 baseline 产物** 给出可追溯证据，说明目前还剩哪些问题、它们更可能属于哪一类（物理/高度场/位移场/渲染），以及下一步怎么最快收敛。

---

## 1. 复跑参数确认（避免“以为开了其实没开”）

来自 `output/rgb_compare/baseline/run_manifest.json:1`：

- 本次 baseline 时间：`created_at=2026-01-01T02:30:02+08:00`
- `--mpm-marker warp`：已启用（不是 static/off）
- `--mpm-depth-tint off`：已关闭
- 摩擦：`fem_fric_coef=0.4`，`mpm_mu_s=0.4`，`mpm_mu_k=0.4`（对齐）
- 轨迹：`press=1mm`，`slide=3mm`，`press_steps=150`，`slide_steps=240`，`hold_steps=40`，`record_interval=5`（共 86 帧）
- 翻转约定：`mpm_render_flip_x=false`，`mpm_warp_flip_x=false`，`mpm_warp_flip_y=true`

> 结论：你这次“marker 位移没看到/不连续”不是因为没开 warp，而更可能是 **中间量（height/uv）本身的分布/离群** 或 **MPM 物理（材料/接触）与 FEM 不同**。

---

## 2. 方向/镜像问题基本排除（至少在 0075/0080/0085 上）

对 `75/80/85` 做了 direct vs mirror 判定（见 `output/rgb_compare/baseline/alignment_flip_latest.csv:1`）：

- `best_counts={'noflip': 3}`
- 三帧均判定：`mpm_vs_fem=direct` 且 `uv_best=noflip`

> 结论：这组 baseline 的“左右镜像/方向错”不是主要矛盾；你看到的“怪”更偏向 **位移场稀疏/尖峰** 与 **高度场伪影**。

---

## 3. 仍然存在的核心问题 A：高度场离群值导致 halo 风险（会干扰判断）

对 intermediate 采样分析（见 `output/rgb_compare/baseline/analysis_latest.csv:1`）：

- 后段（接近 `0075/0080/0085`）：
  - `height_min_mm≈-2.97mm`
  - `height_grad_p99≈0.47 mm/cell`
  - `phenomena_tags=halo_risk`

含义：

- 高度场存在“远超预期的负深值/陡坡”，在多色线光源下容易出现 halo/局部变暗；
- 这会让你很难凭肉眼分辨“物理剪切带”与“高度场伪影”。

**快速验证**（不改物理，只做最后一道防线）：开启离群裁剪后看 halo 是否收敛：

```bash
python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --press-mm 1.0 --slide-mm 3.0 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --mpm-height-clip-outliers on --mpm-height-clip-outliers-min-mm 2.0 --save-dir output/rgb_compare/baseline_clip2
```

> 注：这一步主要是“降低伪影干扰”，不保证 marker 位移会立刻变得像 FEM。

---

## 4. 仍然存在的核心问题 B：MPM 的 uv_disp_mm 在接触区“稀疏 + 尖峰” → 直接导致 marker 观感异常

### 4.1 接触区位移统计（frame_0085）

从 `output/rgb_compare/baseline/intermediate/frame_0085.npz:1` 统计：

- 接触区（`contact_mask==1`）位移幅值 `|uv|`：
  - `p50≈0.00625mm`（中位数很小）
  - `p90≈0.298mm`（只有少量区域明显）
  - `max≈1.13mm`（存在尖峰）
- 接触区 `u` 分量（沿 +x）：
  - `u_p50≈-0.00016mm`（几乎为 0）
  - `u_p90≈0.265mm`

解释（对应你肉眼看到的现象）：

- **大多数接触区格点的 u 位移几乎为 0** → marker 看起来“中部不怎么动”；
- 同时存在 **少量尖峰（max≈1.13mm）** → warp 后会把点拉成短横线/局部“抽风”。

### 4.2 光流证据：FEM 接触区 marker 更“连续右移”，MPM 更“弱且不连续”

对 `frame0 -> frame85`，以 diff centroid 附近作为 ROI 做 LK 光流（复现命令见 `Report_marker_baseline.md`）：

- FEM：`dx_p50≈15.21px`
- MPM：`dx_p50≈4.84px` 且 `dx_p10≈0px`

将图像宽度 400px 与 gel 宽 17.3mm 粗略换算（≈23.1 px/mm）：

- FEM `dx_p50≈0.66mm`
- MPM `dx_p50≈0.21mm`

> 结论：MPM 的接触区“确实在动”，但 **量级明显更小且分布更不均匀**（大量点接近不动）。

---

## 5. 为什么“只调摩擦/接触刚度”往往不够（仍可能存在的物理不一致）

你当前 baseline 的 MPM 材料与接触采用：

- 材料：Ogden（`ogden_mu=[2500Pa]`，`ogden_kappa=300000Pa`）
- 接触：penalty + elastoplastic friction（见 `xengym/mpm/contact.py:224` 与 `xengym/mpm/mpm_solver.py:227`）

而 FEM 侧（VecTouchSim FEM）是另一套接触/摩擦求解（见 `xengym/fem/simulation.py:459`）。

因此即使把摩擦系数统一为 `0.4`，以下差异仍会存在并影响“中部 marker 是否明显右移”：

1) **材料剪切刚度不一致**：MPM 的等效“软硬程度”与 FEM 未标定，剪切带厚度/位移量级会偏。
2) **penalty 接触的法向穿透/法向力分布不同**：会改变真实摩擦上限与 stick/slip 区域分布。
3) **顶面提取 + hole fill/smooth** 的副作用：会让 uv_disp_mm 更“稀疏化/尖峰化”，进一步放大 marker 观感差异。

---

## 6. 结论与建议（按优先级，最短路径收敛）

1) **先“去伪影再判断物理”**：用 `--mpm-height-clip-outliers on` 把 halo/dark blob 风险压下去，避免高度场离群值干扰你对 marker 的判断。
2) **把 uv_disp_mm 当成第一性证据**：当前主要矛盾是 “接触区 u_p50≈0 + 少量尖峰”，这比直接盯 PNG 更可靠。建议对比 `frame_0075/0080/0085` 的 `uv_disp_mm`（或启用 `--mpm-debug-overlay uv`）确认“稀疏/尖峰”是否稳定复现。
3) **再做物理标定（材料优先于摩擦）**：如果你的目标是让 MPM 的接触区位移量级更接近 FEM，更可能需要优先 sweep `--mpm-ogden-mu/--mpm-ogden-kappa`（保持 press=1mm 不变），而不是只调 `mu/k_t`。

---

## 7. 参考（本次复盘用到的关键文件）

- 复跑 manifest：`output/rgb_compare/baseline/run_manifest.json:1`
- 镜像/方向判定：`output/rgb_compare/baseline/alignment_flip_latest.csv:1`
- intermediate 采样分析：`output/rgb_compare/baseline/analysis_latest.csv:1`
- 关键帧中间量：`output/rgb_compare/baseline/intermediate/frame_0085.npz:1`
- UV 与高度提取实现：`example/mpm_fem_rgb_compare.py:1122`
- marker warp 映射实现：`example/mpm_fem_rgb_compare.py:474`
- MPM 接触/摩擦实现：`xengym/mpm/contact.py:224`
- MPM grid contact 入口：`xengym/mpm/mpm_solver.py:227`
