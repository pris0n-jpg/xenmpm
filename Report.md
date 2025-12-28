# MPM vs FEM RGB 对比排查报告

> 最近更新：2025-12-29  
> 目标：解释并收敛“MPM 侧非物理渲染伪影”，让 MPM vs FEM 具备可对比性（同帧、同几何、同渲染开关下，MPM 不再出现块状硬边暗影与异常 halo）。

## TL;DR（结论先行）

在 **Windows + PowerShell + conda 环境 `xengym`（Python 3.9.25 / Taichi 1.7.4 / ezgl 可用）** 下已完成真实复测与闭环调试，核心结论：

1. **“整块发黑/暗盘（dark blob）”的主因是 MPM 高度场出现异常深值**  
   - 机制：表面粒子在局部缺失时，某些 grid cell 会把更深层粒子误当成“表面”，导致 `height_field_mm` 出现远超几何允许的负值；该异常会被多色线光源放大成“整块变暗/发脏”。
   - 证据：`output/rgb_compare/issue080-repro-2025-12-29_01-55-01-k800/intermediate/frame_0032.npz`  
     `height_field_mm min ≈ -2.84mm`（明显超过 press=1mm、gap=0.5mm 的几何上限）。

2. **已通过“压头表面 clamp”把接触区深度限制到几何上限，暗盘伪影显著消失**  
   - 修复：在 `example/mpm_fem_rgb_compare.py` 的 `MPMHeightFieldRenderer.extract_height_field()` 中引入 `mpm_height_clamp_indenter`（默认开），将接触 footprint 内的高度场限制为“不低于压头表面位移”。
   - 证据：`output/rgb_compare/issue080-final3-2025-12-29_04-47-14/intermediate/frame_0032.npz`  
     压头 footprint 内 `height_field_mm ≈ -0.50mm`（与几何一致：press=1.0mm、gap=0.5mm → 最大压入=0.5mm）。

3. **“彩虹 halo”主要来自“高度场台阶/陡坡 + 多色灯光”，不是 RGB 叠色 bug**  
   - 在暗盘被抑制后，halo 收敛到主要集中在接触边界；仍可通过 `fill_holes/smooth` 进一步抑制，但不再出现大块硬边矩形暗影。

4. **marker“几乎不变形”已具备正确输入通路（uv_disp → warp），但位移量级偏小属物理/参数问题**  
   - 证据：`output/rgb_compare/issue080-final3-2025-12-29_04-47-14/intermediate/frame_0032.npz`  
     `uv_disp_mm`：`p99≈0.082mm，max≈0.25mm`，可驱动 `--mpm-marker warp`。

5. **FEM 深度幅值语义存在缩放：`depth *= 0.4`**  
   - 这会让 FEM 在 flat indenter 下更像“边界 ring”（梯度主要在边缘），而不是“整块压入”；对比时必须明确该语义，避免误把幅值差异归因成 MPM bug。

---

## 0) 可复现命令与交付物

### 0.1 冒烟测试（必须通过）

```bash
conda run -n xengym python quick_test.py
```

期望输出：`OK: quick_test`

### 0.2 复测基线命令（可审计输出）

推荐（marker warp，关闭 tint，导出中间量）：

```bash
conda run -n xengym python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --fric 0.4 ^
  --mpm-marker warp --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/baseline
```

推荐（白底对照：两边都关 marker）：

```bash
conda run -n xengym python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --fric 0.4 ^
  --fem-marker off --mpm-marker off --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/marker_off
```

### 0.3 输出结构（交付检查）

当 `--save-dir` 设置后，目录必须包含：

- `run_manifest.json`：生效参数 + 轨迹帧映射（frame→phase/step）+ 对比约定（flip/滤波/开关）
- `metrics.csv` / `metrics.json`：逐帧 RGB 差异指标（MAE/p50/p90/p99 等）
- `intermediate/frame_XXXX.npz`（开启 `--export-intermediate` 时）：  
  `height_field_mm`、`uv_disp_mm`、`contact_mask`、`fem_depth_mm`、`fem_marker_disp`、`fem_contact_mask_u8`

---

## 1) 现象逐项归因（对应最初截图的 5 点）

### 1.1 大块“硬边矩形暗影”接触印记

**根因组合：**

- 几何未对齐（MPM box vs FEM STL face 选错 → 方形接触面）会产生“直线硬边”；
- MPM 高度场存在异常深值（更深层粒子被误当成表面），在多色线光源下形成“整块变暗/发脏”。

**本次收敛措施：**

- FEM：默认使用 `circle_r4.STL` 的 `tip(y_max)` 端面（圆形 8mm×8mm）；
- MPM：支持 `cylinder` SDF（flat round pad），默认与 `circle_r4.STL tip` 语义更接近；
- MPM：新增 `mpm_height_clamp_indenter`，把接触 footprint 内深度 clamp 到压头表面（抑制异常深值）。

### 1.2 彩虹/光晕带（halo）

**主要成因：**高度场陡坡/台阶 + 多色线光源导致的彩边分离。  
当暗盘被抑制后，halo 主要集中在接触边界；如果还需进一步收敛，优先调：

- `--mpm-height-fill-holes on/off` + `--mpm-height-fill-holes-iters N`
- `--mpm-height-smooth on/off` + `--mpm-height-smooth-iters N`

### 1.3 整体亮度/色调不一致（MPM 更暗/更脏）

**主要成因：**

- MPM 的 depth tint 属于“非物理叠色”，会系统性压暗 G/B 并增强 R（基线对比应默认关闭）；
- marker 纹理生成/投影语义不同（FEM vs MPM pipeline 不同），导致纯像素差异不一定代表物理差异。

### 1.4 marker 点阵在接触附近“几乎不变形”

**结论：**必须让 marker 参与面内位移的 warp。  
MPM 侧已支持 `--mpm-marker warp` 并把 `uv_disp_mm` 每帧写入渲染；位移幅值是否足够明显，取决于材料/摩擦/轨迹与记录频率。

### 1.5 接触印记边界与背景融合不自然

**根因：**同 1.1/1.2。先消除“几何不对齐 + 异常深值”，再谈滤波/融合；否则会用滤波掩盖物理差异。

---

## 2) 关键证据（推荐对照帧）

### 2.1 旧问题复现（暗盘明显）

- 目录：`output/rgb_compare/issue080-repro-2025-12-29_01-55-01-k800/`
- 关键帧：`mpm_0032.png`
- 中间量：`intermediate/frame_0032.npz`
  - `height_field_mm min ≈ -2.84mm`

### 2.2 修复后（暗盘消失，接触边界收敛）

- 目录：`output/rgb_compare/issue080-final3-2025-12-29_04-47-14/`
- 关键帧：`fem_0032.png`、`mpm_0032.png`
- 中间量：`intermediate/frame_0032.npz`
  - 压头 footprint 内：`height_field_mm ≈ -0.50mm`（与几何上限一致）
  - FEM 深度幅值：`fem_depth_mm_max≈0.40mm`（对齐时需考虑 `depth *= 0.4`）

---

## 3) 已实现改动清单（与对比可信度相关）

- `xengym/mpm/contact.py`：新增 cylinder SDF 与法线计算（Z 轴对齐 capped cylinder）
- `xengym/mpm/mpm_solver.py`：支持 cylinder obstacle；引入 obstacle velocity（用于相对速度摩擦更合理）
- `xengym/mpm/config.py`：SDFConfig 文档补充 cylinder 语义
- `example/mpm_fem_rgb_compare.py`：
  - FEM 默认 `--fem-indenter-face tip`（避免误用 `circle_r4.STL` 的 15mm 方形底座）
  - MPM 默认 indenter 为 `cylinder`
  - 高度场：hole filling + 平滑 +（新增）`mpm_height_clamp_indenter`
  - UV：按“接近当前顶面”的粒子筛选聚合，支持 `--mpm-marker warp`
  - headless batch：`--save-dir` 默认走 batch，稳定落盘 `run_manifest/metrics/intermediate`

---

## 4) 风险与下一步建议

1. **仍可能存在“非 footprint 区域的异常深值”**（由表面采样缺粒子导致）  
   当前已不再主导观感，但若要进一步提升稳健性，建议增加一层“离群点剔除 + inpaint”（基于邻域一致性，而非仅依赖 hole filling）。

2. **FEM 深度缩放语义**  
   `xengym/fem/simulation.py` 中存在 `depth *= 0.4`，会直接影响 MPM vs FEM 的幅值对齐；如需“严格幅值对齐”，建议将该缩放做成可配置项或在 compare 脚本提供可审计的 scale 开关（写入 `run_manifest.json`）。

3. **准静态一致性（物理层面）**  
   `--steps` 与 `--mpm-dt` 的组合会改变压入速度；若要更像 FEM 的准静态，可考虑增加阻尼/Maxwell 分支并固定物理时长（这属于后续物理对齐工作）。

---

## 5) Issue 080 交付复核

Issue：`MPMFEMRGBALIGN-080 xengym 环境复测并收敛剩余深度/伪影`

- 已在 `xengym` conda 环境复测：`quick_test.py` 通过
- `mpm_fem_rgb_compare` 可稳定落盘 `run_manifest.json` + `metrics.*` + `intermediate/*.npz`
- 暗盘伪影已收敛：通过 `mpm_height_clamp_indenter` 把接触区限制到几何上限，并在证据目录中给出关键帧与中间量
