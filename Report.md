# MPM vs FEM 传感器 RGB “看起来不一样”全面调查报告

> 目标：把 **非力学因素**（参数/尺度/坐标/渲染/marker 语义）从 **物理差异** 中剥离出来，让 “不一样” 可被可靠归因；并给出一套可执行的基线与排查 checklist。

## 0. 调查范围与证据来源

本次调查基于仓库现有实现的静态审阅（未在本环境跑 Taichi/ezgl 实际渲染链路）：

- 对比脚本：`example/mpm_fem_rgb_compare.py`
- FEM 深度/摩擦语义：`xengym/fem/simulation.py`
- VecTouchSim/SensorScene（基线渲染风格与 texcoords 约定）：`xengym/render/sensorScene.py`
- 光源配置（彩虹 halo 的颜色来源）：`xengym/assets/data/light.txt`
- 轻量回归/防回退：`quick_test.py`

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
- `--mpm-marker warp`：让 marker 参与面内位移（否则“静态贴图”，见 `example/mpm_fem_rgb_compare.py:2753`）。
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

- `--mpm-marker` 默认是 `static`（`example/mpm_fem_rgb_compare.py:2753`），static 模式下纹理不会随面内位移变形（只会随高度变化影响阴影）。
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
