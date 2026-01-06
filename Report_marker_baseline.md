---
title: MPM marker 滑移异常复盘（baseline）
cwd: F:\workspace\xenmpm
created_at: 2025-12-31T17:56:50+08:00
updated_at: 2025-12-31T18:25:00+08:00
inputs:
  baseline_dir: output/rgb_compare/baseline
  rerun_dir: output/rgb_compare/baseline_marker_clip2
tuning_dirs:
  - output/rgb_compare/tune_kt800
  - output/rgb_compare/tune_kt2000
  - output/rgb_compare/tune_mu1
  - output/rgb_compare/tune_mu2
  - output/rgb_compare/tune_kn4000_kt2000
  - output/rgb_compare/tune_press2
---

# MPM marker 滑移异常复盘（baseline）

## 0. 背景与目标

用户反馈：在 `output/rgb_compare/baseline` 的结果里，MPM 的 marker 点出现“该滑移的不滑移/不该滑移的诡异滑移”，与 FEM 观感不一致；直觉上“压头从中间按下并向右滑移时，中部附近 marker 应该有位移”。

本报告目标：

1) 用现有产物（PNG + intermediate）对“是否真的不动/乱动”做定量证据；  
2) 区分 **渲染/warp 链路问题** 与 **MPM 物理/接触导致的 slip**；  
3) 给出可复现的诊断命令与下一步建议。

---

## 1. 复现与产物

### 1.1 基线目录

- `output/rgb_compare/baseline/fem_0075.png` / `output/rgb_compare/baseline/fem_0080.png` / `output/rgb_compare/baseline/fem_0085.png`
- `output/rgb_compare/baseline/mpm_0075.png` / `output/rgb_compare/baseline/mpm_0080.png` / `output/rgb_compare/baseline/mpm_0085.png`
- `output/rgb_compare/baseline/run_manifest.json`
- `output/rgb_compare/baseline/intermediate/frame_0085.npz`

`run_manifest.json`（本次分析对应的最近一次 baseline run）显示：

- `argv`：`example/mpm_fem_rgb_compare.py --mode raw --mpm-marker warp --mpm-depth-tint off --fric 0.4 --record-interval 5 --export-intermediate --save-dir output/rgb_compare/baseline`
- `resolved.render.mpm_marker="warp"`（MPM marker 位移已启用）
- `resolved.conventions.mpm_warp_flip_x=false, mpm_warp_flip_y=true`
- `resolved.conventions.mpm_height_clip_outliers=false`（高度场离群裁剪未开启）

### 1.2 额外短跑（用于验证离群裁剪对伪影的影响）

为了解耦 “高度场极端值→暗盘/halo” 与 “marker warp”，额外跑了一次短序列：

- 输出：`output/rgb_compare/baseline_marker_clip2`
- 命令：
  ```bash
  python example/mpm_fem_rgb_compare.py --mode raw --steps 60 --record-interval 5 --press-mm 1.0 --slide-mm 3.0 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --mpm-height-clip-outliers on --save-dir output/rgb_compare/baseline_marker_clip2
  ```

---

## 2. 结论先行（你关注的“中部 marker 应该动”）

结论：你的直觉是对的——**在 FEM 中，接触附近的 marker 具有明显的右向位移；但在 MPM baseline 中，该位移更弱、更不连续，且伴随尖峰导致的局部拉丝**，因此主观观感会变成“该动的不动/局部抽风”。

我们用 **光流（feature tracking）** 对 `frame0 -> frame85` 的 ROI 做了定量：

- FEM（ROI 取 `diff centroid` 附近）：`dx_p50≈15.2px`（整体右移明显）
- MPM（ROI 同样取 `diff centroid` 附近）：`dx_p50≈4.84px`（右移更弱，且分布更不均匀）

这意味着：**MPM 的 marker warp 并不是“完全没生效”，但在接触附近更像“少量点大幅移动 + 大量点接近不动”**，与 FEM 的“较连续剪切带”不一致。

---

## 3. 证据与指标

### 3.1 baseline：高度场离群值仍在（会贡献 halo/暗盘）

对 `output/rgb_compare/baseline` 跑中间量分析（采样每 phase 3 帧）：

```bash
python example/analyze_rgb_compare_intermediate.py --save-dir output/rgb_compare/baseline --sample-per-phase 3 --out output/rgb_compare/baseline/analysis_new.csv
```

在 slide/hold 末段（接近 `0075/0080/0085`）：

- `height_min_mm≈-2.97mm`（极端负值仍存在）
- `uv_disp_p99_mm≈0.94mm`
- `uv_disp_edge_p99_mm≈0.63mm`

这解释了 MPM 图里接触边缘的 halo/颜色不稳定，以及部分 marker “拖影/短横线”的风险。

### 3.2 baseline：marker 位移的“连续性”不足（光流证据）

对 `baseline` 的 frame0→frame85，用 RGB diff 的质心作为 ROI 中心做光流统计（cv2 LK）：

- FEM diff centroid：`(x≈325.7, y≈392.7)`
- MPM diff centroid：`(x≈331.5, y≈357.6)`

光流统计（ROI 半径约 120px）：

- FEM：`dx_p50≈15.21px`、`dx_p90≈21.27px`
- MPM：`dx_p50≈4.84px`、`dx_p90≈19.44px`、`dx_p10≈~0px`

复现命令（会打印 diff centroid 与光流统计；依赖 `opencv-python`）：

```bash
python -c "import cv2, numpy as np; from pathlib import Path
def wdiff(cur, ref, p=99.0):
    diff=np.abs(cur.astype(np.int16)-ref.astype(np.int16)).sum(axis=2).astype(np.float32)
    thr=float(np.percentile(diff,p)); m=diff>=thr
    if not m.any(): return None
    ys,xs=np.nonzero(m); w=diff[m]; s=float(np.sum(w))
    return (float(np.sum(xs*w)/s), float(np.sum(ys*w)/s))
def flow(p0,p1, c, r=120):
    a=cv2.imread(str(p0),0); b=cv2.imread(str(p1),0); h,w=a.shape; cx,cy=c
    x0=max(int(cx-r),0); x1=min(int(cx+r),w-1); y0=max(int(cy-r),0); y1=min(int(cy+r),h-1)
    m=np.zeros_like(a,np.uint8); m[y0:y1+1,x0:x1+1]=255
    pts=cv2.goodFeaturesToTrack(a,mask=m,maxCorners=500,qualityLevel=0.01,minDistance=5,blockSize=7)
    nxt,st,_=cv2.calcOpticalFlowPyrLK(a,b,pts,None,winSize=(21,21),maxLevel=3,criteria=(3,30,0.01))
    ok=(st.reshape(-1)==1); d=(nxt.reshape(-1,2)[ok]-pts.reshape(-1,2)[ok]); dx=d[:,0]; dy=d[:,1]
    pct=lambda x,q: float(np.percentile(x,q))
    return dict(n=int(dx.size),dx_p50=pct(dx,50),dx_p90=pct(dx,90),dx_p10=pct(dx,10),dy_p50=pct(dy,50))
root=Path('output/rgb_compare/baseline'); f=85
fem0=cv2.cvtColor(cv2.imread(str(root/'fem_0000.png')),cv2.COLOR_BGR2RGB); mpm0=cv2.cvtColor(cv2.imread(str(root/'mpm_0000.png')),cv2.COLOR_BGR2RGB)
fem=cv2.cvtColor(cv2.imread(str(root/f'fem_{f:04d}.png')),cv2.COLOR_BGR2RGB); mpm=cv2.cvtColor(cv2.imread(str(root/f'mpm_{f:04d}.png')),cv2.COLOR_BGR2RGB)
fc=wdiff(fem,fem0); mc=wdiff(mpm,mpm0); print('fem_diff_centroid',fc,'mpm_diff_centroid',mc)
print('fem_flow',flow(root/'fem_0000.png',root/f'fem_{f:04d}.png',fc)); print('mpm_flow',flow(root/'mpm_0000.png',root/f'mpm_{f:04d}.png',mc))"
```

解释：

- FEM：大部分 tracked features 都在向右移动（“中部附近 marker 应该动”更符合直觉）。  
- MPM：存在一部分点大幅右移（p90 很高），但大量点接近 0（dx_p10~0，dx_p50 更小）→ 观感上会像“局部抽风/稀疏运动”。

### 3.3 baseline_marker_clip2：离群裁剪抑制了高度极端值，但 uv 尖峰更大

对短跑目录 `output/rgb_compare/baseline_marker_clip2`（挑 9 帧）：

```bash
python example/analyze_rgb_compare_intermediate.py --save-dir output/rgb_compare/baseline_marker_clip2 --frames 3,4,5,6,7,8,9,10,11 --out output/rgb_compare/baseline_marker_clip2/analysis_3_11.csv
```

结果显示：

- `height_min_mm` 被裁剪到约 `-2mm`（符合 `clip_outliers min_mm=2.0` 的预期）
- 但后段 `uv_disp_p99_mm` 升至 `~1.74mm`，且 `uv_disp_edge_p99_mm` 在最后帧达到 `~1.59mm`

这意味着：**“暗盘/halo”会收敛，但 marker 拉丝/局部异常位移不一定会一起收敛**（因为它更直接由 `uv_disp_mm` 的尖峰驱动）。

---

## 4. 可能根因（按优先级）

1) **MPM 的 `uv_disp_mm` 分布更“尖峰化/稀疏化”**：大量区域接近 0，少量区域很大 → warp 后就会出现“多数点不动 + 少数点拖影/短横线”。  
2) **接触/摩擦导致的 slip 与 FEM 不一致**：即使摩擦系数对齐，MPM 的 penalty 接触 + 参数（`k_n/k_t`）可能使“粘滑窗口/剪切带厚度”与 FEM 不同；这属于物理差异而非纯渲染 bug。  
3) **高度场极端值（baseline 未开 clip_outliers）**：会强化 halo/暗盘，间接增加局部光流追踪的不稳定；但它不是 marker “不动”的主因（更像伪影放大器）。

---

## 5. 建议下一步（最短路径）

1) **先把 baseline 的高度离群收敛**（避免 halo 干扰判断）：在完整 baseline 上启用  
   `--mpm-height-clip-outliers on --mpm-height-clip-outliers-min-mm 2.0`，并保留 `--export-intermediate`。  
2) **把 `uv_disp_mm` 的尖峰做工程化收敛**：建议新增开关（不改变默认行为）：
   - `--mpm-uv-clip-outliers on`（对 `|uv|` 做分位数裁剪/上限裁剪）
   - `--mpm-uv-smooth-iters N`（对 uv 场做轻量平滑，降低点状尖峰）
3) **用光流做验收**：以 FEM 为对照，要求 MPM 在接触 ROI 的 `dx_p50` 至少达到一个合理比例（例如 ≥50% FEM），并且 `dx_p10` 不再接近 0（减少“大片不动”）。

---

## 6. B 路线（物理调参）实测：单调调参难以让“中部 marker 连续右移”

用户选择 B（优先从物理侧调参让 stick/slip 更像 FEM）。在不改渲染/warp 链路前提下，
我对 baseline 做了多组 **物理参数**改动并复跑（均为 `record_interval=5`、`press=1mm`、
`slide=3mm`、`mpm_marker=warp`、`mpm_depth_tint=off`、`export_intermediate=on`，除特别说明）。

### 6.1 复跑目录与命令

- baseline（对照组）：`output/rgb_compare/baseline`
  - `python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --press-mm 1.0 --slide-mm 3.0 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/baseline`
- 增大切向接触刚度：
  - `output/rgb_compare/tune_kt800`：`--mpm-k-tangent 800`
  - `output/rgb_compare/tune_kt2000`：`--mpm-k-tangent 2000`
- 增大摩擦系数（仅 MPM 侧，故意与 FEM 不对齐，用于观察 stick/slip 变化）：
  - `output/rgb_compare/tune_mu1`：`--mpm-mu-s 1.0 --mpm-mu-k 1.0`
  - `output/rgb_compare/tune_mu2`：`--mpm-mu-s 2.0 --mpm-mu-k 1.5`
- 同时增大法向/切向接触刚度：
  - `output/rgb_compare/tune_kn4000_kt2000`：`--mpm-k-normal 4000 --mpm-k-tangent 2000`
- 增大压深（非“纯物理调参”，但可作为“增大法向力”的对照）：
  - `output/rgb_compare/tune_press2`：`--press-mm 2.0`

### 6.2 定量结果（frame0 → frame85，接触 ROI 光流 + uv_disp_mm 统计）

关键观察：

1) **baseline 的 MPM “中部不怎么动”有定量证据**：MPM ROI 的 `dx_p50≈4.84px`，而 FEM
   同口径约 `dx_p50≈15.21px`（≈3 倍差异）。
2) **在 press=1mm 的前提下，单独调 `k_t` 或 `mu` 并不能把 `dx_p50` 拉上来**：多数调参
   组的 `dx_p50` 仍在 `~3–5px`，且 `dx_p10≈0` 依旧存在（“大片不动”没有被消除）。
3) **`uv_disp_mm` 的根因信号很一致**：接触区 `u_inside_p50` 长期接近 0mm（甚至略为负），
   只有 `p90` 才到 `0.25–0.31mm`；这会导致 warp 后出现“少量点大幅变形/拖影 + 多数点不动”。
4) **press=2mm 能显著提高 MPM 的位移，但会把尖峰/拉丝放大**：`dx_p50≈10.66px`，
   但 `dx_p90≈68.99px`、`uv_p99_max≈1.79mm`，对应肉眼看到的强烈短横线/扭曲。

汇总表（均取 `frame_0085`，光流基于 diff centroid 的 ROI；`height_min_min` 是 `analysis.csv` 的全帧最小值，仅用于离群风险提示）：

| 输出目录 | press(mm) | mu_s/mu_k | k_n/k_t | FEM dx_p50(px) | MPM dx_p10/p50/p90(px) | MPM u_inside p50/p90(mm) | uv_p99_max/edge_p99_max(mm) | height_min_min(mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `output/rgb_compare/baseline` | 1.0 | 0.4/0.4 | 800/400 | 15.21 | ~0.00/4.84/19.44 | ~0.0000/0.2651 | 0.942/0.632 | -2.970 |
| `output/rgb_compare/tune_kt800` | 1.0 | 0.4/0.4 | 800/800 | 15.21 | 0.01/4.45/18.84 | 0.0001/0.2458 | 0.940/0.682 | -2.970 |
| `output/rgb_compare/tune_kt2000` | 1.0 | 0.4/0.4 | 800/2000 | 15.21 | 0.04/4.34/19.78 | 0.0002/0.2535 | 0.942/0.812 | -2.969 |
| `output/rgb_compare/tune_mu1` | 1.0 | 1.0/1.0 | 800/400 | 15.21 | 0.00/4.93/21.41 | ~0.0000/0.2681 | 0.962/0.748 | -2.967 |
| `output/rgb_compare/tune_mu2` | 1.0 | 2.0/1.5 | 800/400 | 15.21 | -0.03/3.27/23.23 | ~0.0000/0.3143 | 0.986/0.773 | -2.969 |
| `output/rgb_compare/tune_kn4000_kt2000` | 1.0 | 0.4/0.4 | 4000/2000 | 15.21 | -0.05/1.78/11.06 | -0.0008/0.0948 | 0.520/0.470 | -3.818 |
| `output/rgb_compare/tune_press2` | 2.0 | 0.4/0.4 | 800/400 | 6.38 | 0.09/10.66/68.99 | 0.0059/0.8569 | 1.788/1.657 | -3.391 |

### 6.3 结论（对 B 路线的现实判断）

- 如果你的目标是 **press=1mm、slide=3mm** 下，让 MPM 的 marker 观感接近 FEM（接触区
  出现“较连续的右向位移/剪切带”），那么仅靠 `mu/k_n/k_t` 的单调调参在当前实现下
  **很难稳定达成**：`u_inside_p50≈0` 的“稀疏位移场”是关键瓶颈。
- “增大压深”确实能让 marker 更明显地动起来，但会显著放大 uv 尖峰（短横线/拉丝），同时
  也会破坏与当前 baseline 的可比性（相当于改变法向加载）。

因此更建议把下一步重点放在 **A 路线（工程侧收敛 uv/height 的离群与空洞）**：先让
`uv_disp_mm` 在接触区变得“连续、低尖峰”，再回到 B（物理调参）去微调 stick/slip。

---

## 7. 参考与定位入口

- baseline 产物：`output/rgb_compare/baseline/run_manifest.json:1`
- baseline 中间量：`output/rgb_compare/baseline/intermediate/frame_0085.npz:1`  
- 中间量分析脚本：`example/analyze_rgb_compare_intermediate.py:1`
- 对齐/翻转分析脚本：`example/analyze_rgb_compare_flip_alignment.py:1`
- 生成对比脚本：`example/mpm_fem_rgb_compare.py:1`
