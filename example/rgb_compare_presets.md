# MPM vs FEM RGB Compare：可复现预设命令（Presets）

本文用于“固化”对比脚本 `example/mpm_fem_rgb_compare.py` 的常用复现组合，确保每次排查/对齐/回归都能生成**可审计**的输出产物（`run_manifest.json/metrics/intermediate/帧图`），并可用 `example/validate_rgb_compare_output.py` 做一致性校验。

## 运行模式说明

- 不带 `--save-dir` 时默认进入交互 UI（interactive）。  
- 当提供 `--save-dir` 时，脚本默认以 **headless batch** 模式运行并退出（不进入 UI）。  
- 如需边预览边落盘，额外加 `--interactive`。

## 默认值（容易误解）

- `--mpm-marker` 默认 `warp`（点阵会参与面内位移 warp，因此会随滑移变形）。  
- `--fem-marker` 默认 `on`；若只想看形变/光照，可用 `--fem-marker off --mpm-marker off`。

## 预设 1：press-only（只压入不滑移）

用于先对齐“同帧同侧”（排除滑移方向与翻转耦合）。

```bash
python example/mpm_fem_rgb_compare.py --mode raw --steps 60 --record-interval 5 --press-mm 1.0 --slide-mm 0 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --mpm-show-indenter --export-intermediate --save-dir output/rgb_compare/press_only
```

## 预设 2：baseline（压入 + 3mm 滑移 + hold）

用于复现系统性差异与指标对比。

```bash
python example/mpm_fem_rgb_compare.py --mode raw --steps 60 --record-interval 5 --press-mm 1.0 --slide-mm 3.0 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --mpm-show-indenter --export-intermediate --save-dir output/rgb_compare/baseline_short
```

> 说明：如果你需要完整 baseline（与报告中的 86 帧一致），移除 `--steps 60` 让脚本回到默认 `press_steps + slide_steps + hold_steps`。

## 预设 3：debug（overlay + 位移可视化）

用于快速判定 “仿真 +x” 在图像上对应的左右方向，以及 warp 位移量级是否异常。

```bash
python example/mpm_fem_rgb_compare.py --mode raw --steps 60 --record-interval 5 --press-mm 1.0 --slide-mm 3.0 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --mpm-show-indenter --mpm-debug-overlay uv --export-intermediate --save-dir output/rgb_compare/baseline_debug_short
```

`--mpm-debug-overlay` 可选：`uv|warp`（见 `example/mpm_fem_rgb_compare.py` 的参数说明）。

## 输出校验（强烈建议每次都跑）

```bash
python example/validate_rgb_compare_output.py --save-dir <上面生成的目录> --require-alignment --require-indenter-overlay
```

该校验会确认：

- `run_manifest.json` 存在且 `trajectory.total_frames > 0`；
- `metrics.csv/metrics.json` 行数与 `total_frames` 一致；
- 若导出 intermediate，至少存在 `intermediate/frame_0000.npz`；
- 若环境支持 cv2，帧图 `fem_*.png/mpm_*.png` 数量与 `total_frames` 一致。

## baseline 验收口径（关键帧 + 指标）

为避免“看起来不一样但不知道差在哪”，建议把验收口径固定为：

- **关键帧**（默认轨迹/record_interval=5 下）：
  - `press_end=29`
  - `slide_mid=53`
  - `hold_end=85`
  - 以及人工对齐用帧：`75/80/85`
- **核心关注点**：
  - 高度场是否存在离群值导致的 halo/dark blob 风险（影响主观判断）。
  - `uv_disp_mm` 在接触区的 coverage 与分布是否“稀疏 + 尖峰”（对应 marker “不动 + 抽风”）。
  - flip/alignment 判定是否稳定（避免坐标系翻转造成的系统性错位）。

## 分析复跑（生成/更新 analysis_latest.csv 与 alignment_flip_latest.csv）

对一个既有输出目录（例如 `output/rgb_compare/baseline`），可直接用离线脚本生成/更新分析产物：

```bash
python example/analyze_rgb_compare_intermediate.py --save-dir output/rgb_compare/baseline --out output/rgb_compare/baseline/analysis_latest.csv
python example/analyze_rgb_compare_flip_alignment.py --save-dir output/rgb_compare/baseline --frames 75,80,85 --out output/rgb_compare/baseline/alignment_flip_latest.csv
python example/analyze_rgb_compare_uv_disp_contact.py --save-dir output/rgb_compare/baseline --frames 75,80,85
```

产物说明：

- `analysis_latest.csv`：对抽样帧的高度场/位移场统计与现象标签（halo_risk/edge_streak_risk 等）。
- `alignment_flip_latest.csv`：对关键帧的 direct vs mirror 与 uv_grid flip 判定，用于快速确认“坐标/翻转”一致性。
- `uv_disp_contact_stats.csv` / `uv_disp_contact_diagnostics.md`：对关键帧接触区的 `uv_disp_mm` 做 coverage/分位数/尖峰位置统计，用于解释“多数不动 + 少量抽风”。

## 离线：height_clip_outliers 对 halo 的量化影响

当环境无法运行 taichi/ezgl 渲染链路时，仍可基于已保存的 `intermediate/frame_XXXX.npz` 离线模拟 `mpm_height_clip_outliers` 的裁剪效果，并量化 halo_risk 相关统计变化：

```bash
python example/analyze_rgb_compare_height_clip_outliers_effect.py --save-dir output/rgb_compare/baseline --frames 75,80,85 --clip-min-mm 2.0
```

产物：

- `height_clip_outliers_effect.csv`：关键帧 clip 前后对比表（min/p1/grad_p99/tags/outlier_count）。
- `height_clip_outliers_effect.md`：对比摘要与结论说明（用于记录“未下降原因”）。

## 离线：warp 出界比例诊断（oob_ratio）

用于判断 warp remap 是否存在较大比例的出界采样（可能导致边缘拉丝/短横线）：

```bash
python example/analyze_rgb_compare_warp_oob.py --save-dir output/rgb_compare/baseline --frames 75,80,85
```

产物：

- `warp_oob_stats.csv`：关键帧 remap 出界像素数与比例（OOB）。

## 离线：surface fields 后处理消融（fill_holes/smooth）

当你怀疑 `uv_disp_mm` 的 coverage/尖峰与高度场 halo 主要由表层场提取/后处理引入（而非物理参数）时，可基于已有输出目录做消融对比：

```bash
python example/analyze_rgb_compare_surface_fields_ablation.py --save-dirs ^
  output/rgb_compare/retest-2025-12-28_04-48-25-k800 ^
  output/rgb_compare/retest-2025-12-28_11-48-18-k800-filloff ^
  output/rgb_compare/retest-2025-12-28_05-11-10-k800-nosmooth ^
  --out Report_surface_fields_ablation.md
```

产物：

- `Report_surface_fields_ablation.md`：>=4 组组合（fill_holes/smooth）对比表 + 推荐设置。

## 坐标/翻转约定（只翻一次）与自动检查

为避免同一轴在多层重复修正（导致“看起来像镜像/方向错”），建议把约定固定为：

- `uv_disp_mm`：u=+x 向右，v=+y 向上（传感器平面坐标）。
- “是否需要水平翻转”只在一处处理：要么由 `--mpm-render-flip-x`（场/渲染一致翻转）负责，要么由 warp 的 `--mpm-warp-flip-x/--mpm-warp-flip-y` 负责。
- 追溯来源：`run_manifest.json` 的 `run_context.resolved.conventions` 与 `scene_params` 会记录上述开关，便于回溯。

可用 alignment 工具对关键帧做自动检查（失败即提示可能存在多翻/漏翻）：

```bash
python example/analyze_rgb_compare_flip_alignment.py --save-dir output/rgb_compare/baseline --frames 75,80,85 --require-mpm-vs-fem direct --require-uv-best noflip --out output/rgb_compare/baseline/alignment_flip_latest.csv
```
