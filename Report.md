# MPM vs FEM RGB 对比排查报告

> 生成时间：2025-12-27  
> 排查范围：`example/mpm_fem_rgb_compare.py`、`xengym/render/sensorScene.py`、`xengym/fem/simulation.py`、`xengym/assets/obj/*.STL`  
> 目标：解释并收敛“MPM 侧非物理渲染伪影”，让 MPM vs FEM 具备可对比性

## TL;DR（结论先行）

你截图里 MPM 侧的主要问题（整块矩形变暗 + 彩虹 halo + 整体偏暗偏脏 + marker 不随形变）**当前都能被“几何/坐标/渲染策略不对齐”解释**，因此现阶段还不能把差异直接归因到 MPM 力学本身。

对比可信度最敏感的污染源（按优先级）：

1. **压头几何未对齐（基线失效）**  
   - MPM 物理压头：默认 SDF `box`（见 `SCENE_PARAMS['indenter_type']`、`MPMSimulationAdapter.setup()`）  
   - FEM 渲染/接触压头：默认用 `circle_r4.STL` 做深度渲染（见 `DepthRenderScene` 默认路径）
2. **`circle_r4.STL` 的“接触面”如果选错，会变成 15mm 方形底座接触**  
   - `circle_r4.STL` 的 `y_min` 端面是 **15mm 方形底座**，`y_max` 端面才是 **直径约 8mm（r≈4mm）圆端面**  
3. **摩擦系数数量级不一致（直接改变剪切带/粘滑表现）**  
   - FEM：`fric_coef` 默认约 `0.4`（`xengym/fem/simulation.py`）  
   - MPM：`mu_s/mu_k` 默认约 `2.0/1.5`（`example/mpm_fem_rgb_compare.py`）
4. **坐标/滑移方向存在“多处补丁式翻转”**（容易出现“强边缘在相反侧”）  
   - MPM 物理轨迹、渲染侧 `height_field`、overlay 的翻转逻辑分散在多处（`_mpm_flip_x_*` 以及 dx_slide 的正负号）
5. **渲染策略差异在放大观感差异**  
   - MPM 侧默认开启 `depth tint`（按压深度叠色，会压暗 G/B 并增强 R）  
   - MPM marker 默认 `static`（不随面内位移变化），而 FEM marker 通过“先在变形网格上渲染纹理再投影”表现为“跟着表面走”

---

## 0) 复现与证据（仓库内现成样例）

> 说明：本工作区当前 Python=3.13，`taichi`/`xensesdk.ezgl` 依赖与本机 wheel/二进制不匹配，无法在此环境直接跑 OpenGL/MPM。下面证据来自静态代码审查 + 仓库已落盘输出。

仓库内已有一组对比输出（可直接目视）：

- FEM：`xengym/output/mpm_fem_warp_test/fem_0015.png`
- MPM：`xengym/output/mpm_fem_warp_test/mpm_0015.png`（可见：大块暗影 + 强烈 halo）

额外定量证据（同一帧 15，中央区域裁剪后统计亮度）：

- FEM crop（中央区域）：`mean_y≈135.8`
- MPM crop（中央区域）：`mean_y≈105.0`（显著更暗）
- 对 MPM crop 做 box blur（半径 12）后，低亮区域（取亮度第 15 分位阈值）形成相对“紧凑的轴对齐包围盒”，符合“整块矩形变暗”的观感；FEM 不呈现同样集中块状结构。

---

## 1) 现象逐项归因（对应你截图里的 5 点）

### 1.1 大块“硬边矩形暗影”接触印记

**高概率成因（可叠加）：**

1. **压头几何/接触面本身就是方形或近方形**  
   - MPM 默认 `box` 压头会产生轴对齐的接触边界；  
   - FEM 若把 `circle_r4.STL` 的 `base(y_min)` 朝向硅胶，则接触轮廓会落在 15mm 方形底座上。
2. **MPM 的 depth tint 属于“非物理叠色”，会把压入区域做成“整块变暗/变脏”**  
   - `MPMSensorScene._update_depth_tint_texture()` 会按 `depth_pos/max(depth_pos)` 归一化，压暗 G/B、增强 R；在网格离散较粗或深度场台阶明显时，很容易形成“硬边块状”观感。

**最小验证：**

- 先把几何对齐：两边都用同一几何（都用 `box` 或都用同一 tip STL），确认“方框轮廓”能否稳定复现/消除。
- 关闭 MPM depth tint：`--mpm-depth-tint off`，看“整块变暗”是否显著收敛。

### 1.2 接触区左侧“彩虹/光晕带”伪影

**更像“法线/曲率异常 + 彩色灯光”导致的 halo，而不是一个单纯的 RGB 叠加 bug：**

- 传感器渲染里使用了多路彩色线光源（`xengym/assets/data/light.txt`）。  
  当表面出现陡峭的高度台阶/尖锐边界时，法线快速变化会把不同颜色光源分离成彩边，看起来像“彩虹 halo”。
- MPM 高度场来自粒子离散到 `(140,80)` 网格（每格取最大值），再做轻量 box blur；相比 FEM（变形网格 + 深度模式纹理投影），更容易出现“台阶/陡坡”，从而放大 halo。

**验证路径：**

- 关闭 tint + 关闭 marker（两边都白底）：若 halo 仍存在，优先检查高度场平滑/插值（而不是继续追“颜色叠加”）。
- 临时提高平滑强度（例如把 `set_height_field()` 的 blur 迭代数加大）观察 halo 是否缩小。

### 1.3 整体亮度/色调与 FEM 不一致（MPM 更暗、更脏）

主要来自 MPM 侧“额外的非物理处理”：

- depth tint 默认开启：会系统性压暗 G/B（看起来更“脏/暗”）；
- marker 纹理生成语义不同：FEM 通过 `MarkerTextureCamera` 从变形网格渲染纹理后再投影到深度网格，MPM 默认静态 marker + tint，整体观感天然不一致。

**建议：**先在对比模式里把 depth tint 默认关掉，把“增强对比的可视化手段”从基线里移除，避免污染 MPM vs FEM 的主对比。

### 1.4 Marker 点阵在接触附近“几乎不变形”

这是当前实现差异导致的“预期现象”，不是渲染 bug：

- FEM：虽然最终显示的是深度网格，但纹理来自“变形 FEM 网格的正交渲染”，因此 marker 会体现面内位移/剪切（看起来像“跟着表面走”）。
- MPM：高度场只更新 z，不含 x/y 面内位移；若 marker 纹理不做 warp，点阵就会像“贴在屏幕上的网格”。

**对应修复方向：**

- 启用并对齐 MPM marker warp（`--mpm-marker warp`），并确保每帧把 `uv_disp_mm` 传入 `MPMSensorScene.set_uv_displacement()`。

### 1.5 接触印记边界与背景融合不自然

根因与 1.1/1.2 同源：

- 几何/接触面不对齐会直接产生“直线/硬边”边界；
- 高度场离散（max-per-cell）+ 平滑不足会产生台阶，导致边界过锐；
- depth tint 会把边界进一步“涂抹成块状色阶”。

**建议顺序：**先对齐几何与坐标（消除非物理硬边来源），再讨论更细的平滑/融合策略；否则会用滤波掩盖真实对比差异。

---

## 2) 仍需重点确认的“非物理差异”（会继续污染对比）

1. **投影尺度可能不一致**  
   - FEM 深度相机视场参数（`cam_view_width_m/cam_view_height_m`）与 `gel_size_mm` 不一致时，会造成 depth→gel 的比例失配，影响接触区域尺度与强度。
2. **overlay/调试模式可能混入对比图**  
   - `--mpm-debug-overlay`、`--mpm-show-indenter` 应确保在“基线对比”默认关闭，否则会把中间量可视化污染到 RGB。

---

## 3) 建议的修复/收敛顺序（按性价比）

1. **先修压头几何与接触面**：确保 FEM 侧接触的是 tip 端面（不要让 15mm 方形底座接触），并让 MPM/FEM 使用同一几何与尺寸。  
2. **统一滑移方向/坐标翻转约定**：把翻转集中到单一坐标变换函数，增加同帧 pose 对照日志/overlay 验证左右一致。  
3. **对齐摩擦系数**：先把数量级拉齐（例如两边先统一到 `0.4`），避免剪切带差异被摩擦主导。  
4. **对齐渲染策略**：基线默认关闭 depth tint；MPM marker 使用 warp（语义更接近 FEM）。  
5. **导出中间产物与差异指标**：`height_field_mm/uv_disp_mm/contact_mask` + RGB diff metrics，用于把“渲染差异”与“物理差异”剥离并可回归。

---

## 4) 推荐基线命令（可审计输出）

当环境具备依赖（Taichi + ezgl）时，建议用下面命令产出可追溯对比输出（剥离 tint 污染，并启用更接近 FEM 的 marker 语义）：

```bash
python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/baseline
```

也建议额外保存一组“白底基线”用于观感归因（两边都关 marker + 关 tint）：

```bash
python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --fric 0.4 --fem-marker off --mpm-marker off --mpm-depth-tint off --save-dir output/rgb_compare/marker_off
```

输出目录（`--save-dir`）建议至少包含：

- `run_manifest.json`：生效参数 + frame→phase 映射（审计入口）
- `metrics.csv`/`metrics.json`：逐帧 RGB 差异指标（MAE/percentiles 等）
- `intermediate/frame_XXXX.npz`：逐帧中间产物（按需启用），关键 keys：`height_field_mm`、`uv_disp_mm`、`contact_mask`

---

## 5) 仍需补充的信息（用于把问题收敛到“可复现的单点 bug”）

建议在一次对比运行里记录（写日志或写入 manifest）：

- 截图对应的 `frame_id` 与当次 CLI 参数
- 当帧的：
  - MPM obstacle center（x,y,z）与 `frame_controls(dz, dx_slide)`
  - FEM object_pose（x,y,z）
- 是否启用了：`--mpm-marker warp`、`--mpm-depth-tint off`、`--fem-indenter-face tip`

---

## 6) 排查 Checklist（建议顺序）

1. **确认几何基线**
   - FEM：`--fem-indenter-geom` / `--fem-indenter-face` 是否正确（tip/base）
   - MPM：`--indenter-type` 与尺寸（box/sphere）是否与 FEM 对齐
2. **确认摩擦对齐**
   - 优先用 `--fric` 统一两侧数值，并在 `run_manifest.json` 里核对 `resolved.friction.aligned=true`
3. **确认坐标与滑移方向**
   - 需要时开启 `--mpm-show-indenter --debug`，用 `[POSE]` 日志与 overlay 核对左右一致
4. **确认渲染开关**
   - 基线对比时建议：`--mpm-depth-tint off`
   - 需要白底对照时：`--fem-marker off --mpm-marker off`
   - 需要对齐 marker 语义时：`--mpm-marker warp`
5. **若仍有 halo，再调 height_field 后处理**
   - 先保持基线：tint=off 且 marker=off
   - 尝试：`--mpm-height-fill-holes on`（必要时调 `--mpm-height-smooth-iters`）并对同一 `frame_id` 做前后对比
6. **用导出产物做可审计归因**
   - 开启：`--export-intermediate`
   - 结合 `metrics.csv/metrics.json` 与 `intermediate/frame_*.npz`，把“渲染差异”和“物理差异”拆开分析
