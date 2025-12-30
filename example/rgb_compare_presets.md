# MPM vs FEM RGB Compare：可复现预设命令（Presets）

本文用于“固化”对比脚本 `example/mpm_fem_rgb_compare.py` 的常用复现组合，确保每次排查/对齐/回归都能生成**可审计**的输出产物（`run_manifest.json/metrics/intermediate/帧图`），并可用 `example/validate_rgb_compare_output.py` 做一致性校验。

## 运行模式说明

- 当提供 `--save-dir` 时，脚本默认以 **headless batch** 模式运行并退出（不进入 UI）。  
- 如需交互预览，额外加 `--interactive`。

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
