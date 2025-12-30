# MPM vs FEM 传感器 RGB “看起来不一样”全面调查报告

> 目标：把 **非力学因素**（参数/尺度/坐标/渲染/marker 语义）从 **物理差异** 中剥离出来，让 “不一样” 可被可靠归因；并给出一套可执行的基线与排查 checklist。

## 0. 调查范围与证据来源

本次调查基于：

1) 仓库现有实现的静态审阅（本环境未重新跑 Taichi/ezgl 的 MPM 仿真与 OpenGL 渲染）；  
2) 你已产出的基线结果 `output/rgb_compare/baseline` 的离线复盘（图片/manifest/metrics/intermediate）。

- 对比脚本：`example/mpm_fem_rgb_compare.py`
- FEM 深度/摩擦语义：`xengym/fem/simulation.py`
- VecTouchSim/SensorScene（基线渲染风格与 texcoords 约定）：`xengym/render/sensorScene.py`
- 光源配置（彩虹 halo 的颜色来源）：`xengym/assets/data/light.txt`
- 轻量回归/防回退：`quick_test.py`
- 本次实测证据（baseline 目录）：
  - `output/rgb_compare/baseline/run_manifest.json`
  - `output/rgb_compare/baseline/metrics.json` / `metrics.csv`
  - `output/rgb_compare/baseline/intermediate/frame_*.npz`
  - `output/rgb_compare/baseline/fem_*.png` / `mpm_*.png`

> 运行依赖提示：`example/mpm_fem_rgb_compare.py` 需要 `xensesdk.ezgl`（OpenGL）与 `taichi` 才能跑完整 MPM+渲染；即使缺依赖，脚本也会在 `--save-dir` 下写出 `run_manifest.json` 与 `tuning_notes.md` 便于审计（见 `example/mpm_fem_rgb_compare.py:724` 与 `example/mpm_fem_rgb_compare.py:623`）。离线校验/抽样统计可用：`example/validate_rgb_compare_output.py` 与 `example/analyze_rgb_compare_intermediate.py`。

---

## 1. 总结结论（Executive Summary）

你当前看到的现象可以拆成四类，优先级从高到低：

1. **基线未对齐导致“看起来不一样”**：摩擦系数、深度幅值语义、相机正交视野与 gel 尺度、压头接触面（tip/base）等任何一个不一致，都会直接改变滑移剪切带/粘滑范围与 RGB 阴影结构。
2. **MPM 高度场重建是主要伪影源头**：粒子局部缺失或 max-per-cell 选错“表面”会产生异常深值（远超几何允许），在多色线光源下表现为“暗盘/dark blob”与“彩虹 halo”。
3. **marker 语义不对会让位移“看起来怪”**：MPM 若用 `static`，天然“几乎不动”；即便 `warp`，坐标翻转/符号修正做多次会出现“压头向左滑但 marker 向右/不动/边缘抽风”。
4. **UI 循环导致 FEM 帧号重复不是 bug**：交互 UI 播放会对帧索引取模循环（见 `example/mpm_fem_rgb_compare.py:2626`）。

---

## 2. 推荐“可比性最强”的基线命令

### 2.1 基线（建议先跑这一条）

```bash
python example/mpm_fem_rgb_compare.py --mode raw --record-interval 5 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/baseline
```

解释：

- `--fric 0.4`：把 FEM `fric_coef` 与 MPM `mu_s/mu_k` 一键对齐（见 `example/mpm_fem_rgb_compare.py:2704` 与 `:2890`）。
- `--mpm-marker warp`：让 marker 参与面内位移（warp 通过 `uv_disp_mm` 做纹理变形，见 `example/mpm_fem_rgb_compare.py:460-549` 与 `:1659-1705`）。
- `--mpm-depth-tint off`：先去掉“深度着色”这种非物理叠色，降低伪影干扰（见 `example/mpm_fem_rgb_compare.py:2757`）。
- `--export-intermediate`：导出 `height_field_mm/uv_disp_mm/contact_mask`，便于定位伪影根因（见 `example/mpm_fem_rgb_compare.py:2809`）。

### 2.2 方形压头 + 叠加对齐辅助（建议复现“左右/方向”问题时使用）

```bash
python example/mpm_fem_rgb_compare.py --mode raw --object-file xengym/assets/obj/square_d6.STL --indenter-type box --mpm-marker warp --mpm-show-indenter --mpm-debug-overlay warp --save-dir output/rgb_compare/box_debug
```

---

## 3. A. FEM vs MPM 基线对齐项（否则“看起来不一样”不具备物理指向性）

### A1) 摩擦与接触参数不一致（直接改变滑移阶段）

现状（脚本默认）：

- FEM：`fem_fric_coef = 0.4`（`example/mpm_fem_rgb_compare.py:196`；实际生效写入 `VecTouchSim.fem_sim.fric_coef`，见 `xengym/render/sensorScene.py:243-245`）
- MPM：`mpm_mu_s = 2.0`, `mpm_mu_k = 1.5`（`example/mpm_fem_rgb_compare.py:197-198`）

影响：

- 这会直接改变粘滑窗口、剪切带范围、marker 位移量级，从而产生“看起来不像”的差异，但不代表 MPM 物理一定错。

建议：

- 对比基线务必使用 `--fric <value>` 或显式 `--fem-fric/--mpm-mu-s/--mpm-mu-k`，并以脚本启动日志中的 `aligned=` 为准（见 `example/mpm_fem_rgb_compare.py:2932-2936`）。

### A2) FEM 深度语义自带缩放（depth *= 0.4）

证据：

- FEMSimulator 在 `_step_fric()` 内对输入深度做了幅值缩放：`depth *= 0.4`（`xengym/fem/simulation.py:462-466`），随后 `depth_map = depth * 1000`（mm）。
- 对比脚本在深度链路处也明确记录了这条语义与背景值处理（`example/mpm_fem_rgb_compare.py:914-928`）。

影响：

- 若你把 FEM 的 `depth_mm` 当作“真实深度/压入量”去和 MPM 的 `height_field_mm`（顶面位移）直接比，会把“幅值语义差”误判成 MPM bug。

建议：

- 在看“幅值”前，先统一语义：明确比较的是 **压入位移** 还是 **相机深度**，必要时做同一参考面/同一坐标系的变换后再比较。

### A3) 相机视野 vs gel 尺度必须严格一致（否则产生隐式缩放差）

证据：

- `SCENE_PARAMS['gel_size_mm']=(17.3,29.15)`，并要求 `cam_view_width_m/height_m` 与其一致（见 `example/mpm_fem_rgb_compare.py:138-212`）。
- DepthCamera 使用 `ortho_space=(-w/2,w/2,-h/2,h/2,...)`（见 `example/mpm_fem_rgb_compare.py:831-845`）。
- 脚本启动时会打印尺度一致性（`example/mpm_fem_rgb_compare.py:2948-2961`）。

建议：

- 一旦看到 `consistent=False`，先别看 MPM/FEM 差异：先把相机正交范围与 `gel_size_mm` 对齐再继续。

---

## 4. B. MPM 高度场重建：当前主要“伪影源头”

### B1) “整块发黑/暗盘（dark blob）”的机制

核心机制（与渲染无关，来自高度场异常深值）：

- 局部表面粒子缺失/稀疏时，binning/max-per-cell 可能把更深层粒子当作“表面”，导致 `height_field_mm` 出现远超几何允许的负值；
- 在多色线光源（`xengym/assets/data/light.txt`）下，陡峭台阶/异常法线会把局部区域渲染成“整块发黑/发脏”。

脚本中的直接提示：

- `extract_height_field()` 明确说明“不能只追踪初始顶面索引，否则会高度场过深并出现整块变暗”（`example/mpm_fem_rgb_compare.py:1013-1016`）。

### B2) 关键收敛手段：footprint 内 clamp 到压头表面

当前实现（已存在开关，默认开启）：

- `SCENE_PARAMS['mpm_height_clamp_indenter']=True`（`example/mpm_fem_rgb_compare.py:154`）
- 在 `extract_height_field()` 内，若提供 `indenter_center_m`，会对 footprint 内的高度场做几何上限约束（sphere/cylinder/box 都覆盖，见 `example/mpm_fem_rgb_compare.py:1078-1127`）。

效果预期：

- dark blob 通常会显著收敛（至少不会出现“压头 footprint 内整块过深”的非物理暗盘）。

残余风险：

- footprint 外仍可能存在异常深值（clamp_field 只覆盖 inside 区域）；建议结合离群点剔除/更稳健 inpaint（见下一条）。

### B3) “彩虹 halo”主要来源：高度场台阶/陡坡 + 多色灯光

解释：

- 高度场存在台阶（holes/离群值/过硬的 max-per-cell）时，法线变化剧烈；线光源的 RGB 分量不同，会把边界渲染成彩虹 halo。

现有可调手段（脚本已提供 CLI）：

- hole filling：`--mpm-height-fill-holes on|off` + `--mpm-height-fill-holes-iters`（`example/mpm_fem_rgb_compare.py:2963`）
- smoothing：`--mpm-height-smooth on|off` + `--mpm-height-smooth-iters`（`example/mpm_fem_rgb_compare.py:2971`）
- outlier clip（最后一道防线，默认关闭）：`--mpm-height-clip-outliers on|off` + `--mpm-height-clip-outliers-min-mm`（`example/mpm_fem_rgb_compare.py:2987`）

建议的调参顺序（KISS 优先）：

1) 保持 `clamp_indenter=on`；
2) 先开 `fill_holes`（增加 iters），再视情况加 `smooth_iters`；
3) 若仍有 halo，再考虑对 footprint 外极端负值做显式离群剔除（例如阈值/分位裁剪后再 inpaint）。

---

## 5. C. Marker “向左滑但点行为怪”的归因与排查

### C1) marker 语义：非 warp 就会“几乎不变形”

事实：

- `--mpm-marker` 默认是 `warp`（`example/mpm_fem_rgb_compare.py:2955`），会让 marker 参与面内位移；若设为 `static`，纹理不会随面内位移变形（只会随高度变化影响阴影）。
- warp 模式会用 `uv_disp_mm` 做非均匀纹理 warp（见 `warp_marker_texture()`：`example/mpm_fem_rgb_compare.py:460-549`）。

结论：

- 若你在看“位移/剪切”，必须用 `--mpm-marker warp`；否则看到“点不动”是预期行为，不是 bug。

### C2) UV 位移场：必须绑定“当前顶面”而不是“初始顶面一层粒子”

证据：

- `extract_surface_fields()` 里明确写了：若只追踪初始顶面，滑移时 UV 会接近 0，marker 看起来像贴在屏幕上不动（`example/mpm_fem_rgb_compare.py:1201-1203`）。

### C3) 翻转/符号修正很容易“翻两次”（导致方向怪/位移抵消）

当前链路里至少有三处会影响 x 方向符号：

1) 高度场翻转：`set_height_field()` 对 `height_field_mm` 做 `_mpm_flip_x_field`（`example/mpm_fem_rgb_compare.py:1773`）
2) UV 翻转：`set_uv_displacement()` 对 `uv_disp_mm` 做 `_mpm_flip_x_field`（`example/mpm_fem_rgb_compare.py:1671`）；`mpm_uv_disp_u_negate` 作为约定字段仍会写入 manifest（当前默认 `False`，见 `example/mpm_fem_rgb_compare.py:3322`）。
3) warp 额外 flip：`warp_marker_texture(... flip_x=..., flip_y=...)`（`example/mpm_fem_rgb_compare.py:465`）

风险表现：

- “压头向左滑，但 marker 像向右/不动/局部抽风”
- 只有边缘几列出现异常（常与 remap 出界叠加，见下一条）

建议的排查方法：

- 用 `--mpm-debug-overlay uv|warp` 快速目视位移/warp 强度是否集中在接触附近（`example/mpm_fem_rgb_compare.py:1503-1636`）。
- 开 `--export-intermediate` 抽查 `uv_disp_mm[...,0]` 在 slide phase 的符号是否与轨迹一致。
- 若确认方向相反，优先从“多次翻转”入手：把翻转集中在一层（KISS），避免同时在 **场翻转**、**UV 场翻转**、**warp flip_x** 多处都修正同一轴。

### C4) 左边缘“拖影/拉丝”的典型原因：cv2.remap 出界 + BORDER_REFLECT101

证据：

- cv2 路径仍使用 `cv2.remap(... borderMode=cv2.BORDER_REFLECT101)`（`example/mpm_fem_rgb_compare.py:532`），但已在 remap 前对 `map_x/map_y` 做 `clip` 与 numpy fallback 对齐（`example/mpm_fem_rgb_compare.py:530`）。

触发条件：

- 若 `map_x/map_y` 出界（常见于 flip/符号错或局部位移尖峰），现在会被 clip 到边界；因此更应关注“位移场异常/翻转链路”本身，而非边界采样策略。

建议：

- 若仍出现边缘拉丝，优先结合 `--mpm-debug-overlay uv|warp` 与 `--export-intermediate` 检查 `uv_disp_mm` 是否在边界存在异常尖峰。

---

## 6. D. “FEM 帧号重复”说明

这通常是 UI 循环播放导致的预期行为：

- 交互 loop 里对 `frame_idx` 做了取模循环（`example/mpm_fem_rgb_compare.py:2626`），因此帧号会从 0 重新开始。
- 这不代表仿真重复或数据错误，对当前“伪影/对齐”问题影响较小。

---

## 7. 建议排查顺序（Checklist）

按以下顺序能最快把问题归因到“参数/尺度/高度场/marker”中的某一类：

1) **锁定基线命令**：使用 `--save-dir` 固化输出与 `run_manifest.json`，避免“你以为参数一致但实际不一致”。
2) **先对齐摩擦**：`--fric 0.4`，确认启动日志 `aligned=true`。
3) **先去掉非物理叠色**：`--mpm-depth-tint off`，只看形变+光照是否仍有暗盘。
4) **收敛高度场伪影**：保持 `--mpm-height-clamp-indenter on`；逐步调 `fill_holes/smooth`。
   - 若确认 footprint 外出现极端负值：开启 `--mpm-height-clip-outliers on --mpm-height-clip-outliers-min-mm 5.0` 作为最后防线。
5) **再看 marker 语义**：切 `--mpm-marker warp`，再用 `--mpm-debug-overlay uv|warp` 判断“方向/尺度/边界”问题来自哪里。
6) **最后再讨论物理差异**：在上述非力学因素都收敛后，再看 MPM vs FEM 是否仍存在系统性差异。

---

## 8. 可选的工程化改进建议（不改变默认行为的前提下）

> 下面是“更易排查、更不容易翻两次”的改进方向；建议以开关形式引入，避免静默改变旧 demo。

1) **把 marker warp 的翻转约定显式化**：建议把实际生效的 flip 组合持续落盘到 manifest/conventions，并在 `tuning_notes.md` 记录一次调参结论，避免“翻两次”难以追责。
2) **让 cv2 与 numpy 的边界行为一致**：已在 cv2 remap 前对 `map_x/map_y` 做 clip（`example/mpm_fem_rgb_compare.py:530`），避免边缘拉丝干扰归因。
3) **footprint 外离群值剔除**：已提供可回退开关 `--mpm-height-clip-outliers`（默认 off；`example/mpm_fem_rgb_compare.py:2987`），用于把 footprint 外极端负值先置为 NaN 再交给 fill_holes/smooth 收敛。

---

## 9. 本次调查如何体现 KISS / DRY / SOLID / 安全优先

- KISS：优先通过“基线对齐 + 开关收敛”把问题拆成可验证的小变量，避免一次性改动多个链路导致不可归因。
- DRY：复用现有 `VecTouchSim/SensorScene` 的光照/相机/mesh 分辨率（对比脚本已按该方向实现），避免再造一套渲染链路。
- SOLID：将“仿真（MPM/FEM）”与“渲染（RGB/marker/tint）”通过明确接口（height_field/uv_disp）解耦，问题定位可在中间量层面闭环。
- 安全优先：不建议为了“看起来像”去关掉关键接触逻辑或引入未审计的外部依赖；所有收敛手段优先使用可回退开关与可审计输出（manifest/intermediate）。

---

## 10. 基于 `output/rgb_compare/baseline` 的实测结论（当前最关键卡点）

> 本节直接以你这次跑出来的 baseline 产物为证据，回答“现在主要差异到底是什么类型的问题”。  
> 结论先行：这次 baseline 的摩擦/尺度已对齐，但 **FEM vs MPM 在 X 方向存在接近“水平镜像”的不一致**；marker 的“该动不动/不该动乱动”大概率是这个镜像不一致叠加高度场/位移场离群值引起的。

### 10.1 baseline 配置确认：摩擦/尺度已对齐（排除 A 类问题）

证据来自 `output/rgb_compare/baseline/run_manifest.json`：

- 摩擦：`resolved.friction` 显示 FEM/MPM 均为 `0.4` 且 `aligned=true`（已排除“摩擦不一致导致粘滑窗口不同”的主干因素）。
- 尺度：`resolved.scale` 显示 `gel_size_mm == cam_view_mm == [17.3, 29.15]` 且 `consistent=true`（已排除“相机正交视野与 gel 尺寸不一致导致隐式缩放差”）。
- marker 模式：`resolved.render.mpm_marker == "warp"`，`resolved.render.mpm_depth_tint == false`（本次 marker 位移是启用的，且已关闭深度 tint 叠色干扰）。
- 轨迹阶段：`trajectory.phase_ranges_frames` 为 `press:0-29 / slide:30-77 / hold:78-85`，与你对比的帧 `0075/0080/0085`（slide 末段与 hold）一致。

### 10.2 最核心现象：同帧接触圈左右“镜像”（X 方向翻转不一致）

现象（目视）：

- FEM：接触圈在画面右侧（例：`output/rgb_compare/baseline/fem_0075.png`、`fem_0085.png`）
- MPM：同帧接触圈在画面左侧（例：`output/rgb_compare/baseline/mpm_0075.png`、`mpm_0085.png`）

定量（差分质心，复盘思路：与 frame0 做差 → 取差分的 99 分位阈值 → 对高差分区域做加权质心）：

- frame 75：FEM `cx≈314.99px`，MPM `cx≈71.17px`，`mirror(FEM)=399-cx≈84.01px`（差≈-12.84px）
- frame 80：FEM `cx≈325.04px`，MPM `cx≈69.60px`，`mirror(FEM)≈73.96px`（差≈-4.36px）
- frame 85：FEM `cx≈325.71px`，MPM `cx≈69.69px`，`mirror(FEM)≈73.29px`（差≈-3.60px）

结论：MPM 的主要接触区域位置 **非常接近 FEM 的水平镜像**（不是小 offset/缩放误差），优先级最高应按 “X 轴翻转/坐标约定不一致” 排查。

补充（离线对齐证据，直接服务于“多翻/少翻”定位与后续收敛）：

- 使用 `example/analyze_rgb_compare_flip_alignment.py` 在 `output/rgb_compare/baseline` 上对 frame 75/80/85 计算得到：
  - `mpm_vs_fem=mirror`（3/3）
  - `uv_best=flip`（3/3；例如 frame 75：`mpm_cx≈71.17px`，`uv_delta(flip)≈4.32px`）

该结果说明：当前 baseline 下 MPM 的渲染结果与 FEM 更接近“水平镜像”关系；同时 UV 强信号区在像素坐标下更符合“需要水平翻转映射”的假设，属于优先级最高的坐标/翻转一致性问题（见第 11 节）。

### 10.3 轨迹证据：仿真里 slide 是 +x（+3mm），但渲染里方向与 FEM 不一致

证据来自 `run_manifest.json` 的 `trajectory`：

- `frame_controls`：`slide_amount_m` 从 `0.0` 增到 `0.003`（= 3mm），与 `scene_params.slide_distance_mm=3.0` 对应。
- `frame_indenter_centers_m`：x 从 `0.0110509m` 增到 `0.0140509m`（+3mm）。

这说明 **仿真坐标系中的滑移方向是 +x**。如果你在图像坐标里把“向左”当作滑移方向，那么“仿真 +x → 画面向左”本身是可能成立的（取决于渲染坐标约定）；但当前的关键问题是 **FEM 与 MPM 对同一条轨迹的投影方向不一致**，才会出现“同帧接触圈左右镜像”。

### 10.4 marker 异常：warp 已开启，但中间量显示存在离群值风险（会诱发局部乱动/拉丝）

1) 本次 baseline 的 marker warp 是启用的：`resolved.render.mpm_marker == "warp"`。

2) `intermediate/frame_0075.npz / frame_0080.npz / frame_0085.npz` 的统计显示：

- `uv_disp_mm`：p90≈`0.095/0.132/0.161`mm，max≈`0.876/1.030/1.132`mm（量级足以驱动可见 warp，但若方向/翻转不一致，会表现为“该动不动/动反了”）。
- `height_field_mm`：min≈`-2.97mm`（且 p01≈`-2.8mm` 量级），而本次 `press_depth_mm=1.0mm`。  
  这类 **明显越过几何允许的负深值** 是高度场/表面选取异常的强信号，会同时污染法线与 UV 绑定的“表面粒子”选择，从而诱发 marker 的局部尖峰、局部拉丝或“非接触区诡异滑移”。

### 10.5 建议的最短排查闭环（先对齐方向，再看物理差异）

1) **先做 press-only（slide=0）**：确认 FEM/MPM 的接触圈中心在同一侧；只要 press 都左右不一致，后面的 slide/marker 都不可归因。
2) **打开可视化辅助**：建议用 `--mpm-show-indenter --mpm-debug-overlay uv`（或 `warp`）跑一组输出，用 overlay 直接确认“仿真 +x”在画面里到底对应左/右。
3) **把 X 翻转收敛到“只发生一次”**（代码层面的约定统一，而不是在多个环节叠加修正）：
   - FEM 侧的 `SensorScene.vis_depth_mesh` 使用 `x_range=(gel_w/2,-gel_w/2)` 且不对 `depth` 额外做 `[:,::-1]`（`xengym/render/sensorScene.py:197-322`）。
   - MPM 侧目前同时存在 `mesh x_range 反向 + _mpm_flip_x_field + warp flip_x/flip_y + overlay x_mm negate`（`example/mpm_fem_rgb_compare.py:1501-1835`）。
   - 建议优先以 “让 MPM 的最终输出与 FEM 同帧同侧” 为验收目标，逐步去掉/合并重复的翻转来源，并把最终生效的约定固化进 manifest（避免后续“翻两次”复发）。
4) **若仍有 halo/暗盘/marker 尖峰**：在保持 `clamp_indenter=on` 的前提下，按顺序尝试：`fill_holes` → `smooth` → 最后才开 `clip_outliers`（例如 `--mpm-height-clip-outliers on --mpm-height-clip-outliers-min-mm 2.0`）把 `height_field_mm < -2mm` 的极端值先置为 NaN 再 inpaint（见 `example/mpm_fem_rgb_compare.py:1278-1317`）。
5) **验收标准（建议）**：
   - 同一帧 `fem_00xx.png` 与 `mpm_00xx.png` 的接触圈出现在同一侧（至少不再互为镜像）。
   - `metrics.json` 的后段（slide/hold）`mae` 明显下降（当前 frame 75/80/85 的 mae≈28.65/28.96/29.11，属于“系统性错位”级别）。

---

## 11. 坐标系与翻转链路对照表（FEM vs MPM）

> 目的：把“到底在哪里翻转、翻了几次、谁在修正同一根轴”说清楚，避免后续修复出现“翻两次/翻三次”导致的镜像、抵消与边缘伪影。

### 11.1 关键坐标系与正方向（约定/事实）

- **图像像素坐标**：RGB 图像 (H,W)，x 向右为 +，y 向下为 +（约定，OpenCV/PIL 一致）。
- **gel 平面坐标（mm）**：在 `warp_marker_texture()` 中定义 `uv_disp_mm` 的约定：u=+x 向右，v=+y 向上（`example/mpm_fem_rgb_compare.py:465`）。
- **场数组索引**：`height_field_mm` 为 `(n_row,n_col)`，`uv_disp_mm` 为 `(n_row,n_col,2)`；row/col 与 gel(mm)/像素的对应关系由 mesh 的 `x_range/y_range` 决定（见下一节）。

### 11.2 FEM（SensorScene）渲染链路的“硬约定”

FEM 的最终 RGB 来自 `SensorScene.vis_depth_mesh`（depth 渲染模式下先生成纹理再 map 到深度 mesh）：

- **mesh 坐标范围**：`x_range=(gel_w/2, -gel_w/2)`，`y_range=(gel_h, 0)`（`xengym/render/sensorScene.py:197`）。
- **depth mesh texcoords**：`gen_texcoords(140,80, v_range=(1,0))`，u_range 使用默认 `(0,1)`（即 v 翻转、u 不翻转）（`xengym/render/sensorScene.py:15`、`xengym/render/sensorScene.py:205`）。
- **FEM mesh texcoords（另一条链路）**：`vis_fem_mesh` 使用 `u_range=(1,0)`（即 u 翻转）（`xengym/render/sensorScene.py:179`）。

结论：FEM 并不是“u/v 都翻”这种一刀切；不同 mesh/贴图链路的 texcoords 翻转约定并不相同，这会直接影响你在 MPM 侧用“跟随 SensorScene 约定”的判断是否成立。

### 11.3 MPM（MPMSensorScene）渲染链路的“硬约定”

MPM 的最终 RGB 由 `MPMSensorScene` 直接渲染 `surf_mesh`：

- **mesh 坐标范围**：`x_range=(gel_w/2, -gel_w/2)`，`y_range=(gel_h,0)`（`example/mpm_fem_rgb_compare.py:1501`、`example/mpm_fem_rgb_compare.py:1567`）。
- **height_field flip（可开关）**：由 `--mpm-render-flip-x on|off` 控制；默认 `off`（对齐 FEM，修复同帧左右镜像），开启则复现旧行为（legacy）（`example/mpm_fem_rgb_compare.py:1773`、`example/mpm_fem_rgb_compare.py:1783`、`example/mpm_fem_rgb_compare.py:563`）。
- **uv_disp flip（可开关）**：同样由 `--mpm-render-flip-x` 控制（`example/mpm_fem_rgb_compare.py:1671`、`example/mpm_fem_rgb_compare.py:1684`）。
- **marker warp 的额外 flip**：
  - `warp_marker_texture(... flip_x, flip_y)` 在 mm→px 后对 `dx_px/dy_px` 做符号修正（`example/mpm_fem_rgb_compare.py:465`、`example/mpm_fem_rgb_compare.py:501`）。
  - `MPMSensorScene` 默认 `self._warp_flip_x=True`、`self._warp_flip_y=True`（`example/mpm_fem_rgb_compare.py:1561`）。
- **indenter overlay 的 x_mm 修正（可开关）**：随 `--mpm-render-flip-x` 同步（`example/mpm_fem_rgb_compare.py:1830`）。

补充：`run_manifest.json.resolved.conventions` 会记录 `mpm_height_field_flip_x/mpm_uv_disp_flip_x/mpm_warp_flip_x/mpm_warp_flip_y/mpm_overlay_flip_x_mm` 等字段（见 `output/rgb_compare/baseline/run_manifest.json:1`）。目前 `height/uv/overlay` 的 flip_x 已由 `--mpm-render-flip-x` 驱动并可审计；`warp flip_x/flip_y` 仍为内部固定约定（见后续 marker 收敛项）。

### 11.4 X 方向“重复翻转”风险点清单（按发生层归类）

同一根 X 轴目前至少可能被以下环节重复修正：

1) **mesh 层**：`x_range=(gel_w/2, -gel_w/2)`（FEM/MPM 都存在）
2) **field 层**：`_mpm_flip_x_field(height_field_mm)`（`example/mpm_fem_rgb_compare.py:1783`）
3) **uv 层**：`_mpm_flip_x_field(uv_disp_mm)`（`example/mpm_fem_rgb_compare.py:1678`）
4) **warp 层**：`warp_marker_texture(... flip_x=True)` 对 `dx_px` 取负（`example/mpm_fem_rgb_compare.py:501`）
5) **overlay 层**：`_mpm_flip_x_mm` 对标量坐标取负（`example/mpm_fem_rgb_compare.py:1830`）

一旦其中某一层与 FEM 的约定不同（或者 MPM 内部出现“多翻/少翻”），就会出现你看到的：同帧接触圈左右镜像、marker 位移方向反/抵消、边缘点异常拉丝等问题。

### 11.5 收敛原则（供后续修复使用）

- **原则 1：单一真相源**：X 方向翻转只允许在一个层面发生一次（建议优先以 mesh 层的 `x_range/texcoords` 为真相源），其它层面不得再对同一轴做“补救式修正”。
- **原则 2：可审计**：最终生效的翻转约定必须写入 `run_manifest.json.resolved.conventions`，并在 `tuning_notes.md` 留痕。
- **原则 3：先对齐方向再调伪影**：先解决“同帧同侧”（press-only/sliding 方向一致），再处理高度场 outlier/holes/smooth，避免两类变量耦合导致误归因。
