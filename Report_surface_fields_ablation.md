# Report: Surface Fields Post-Processing Ablation (fill_holes / smooth)

本报告基于已保存的 `intermediate/frame_*.npz` 离线统计：

- `finite_ratio_avg/motion_ratio_avg/spike_ratio_avg/spike_boundary_ratio_avg`：来自接触区 `uv_disp_mm`（coverage/尖峰）。
- `grad_p99_avg`：对 `height_field_mm` 的梯度幅值 p99；当 `smooth=true` 时，先按 `mpm_height_smooth_iters` 做 3x3 box blur 再计算（用于近似 halo 风险）。

## Inputs

- frame_slots: `press_end,slide_mid,hold_end`
- motion_eps_mm: `0.01`
- spike_min_mm: `0.2`

### Source runs
- `output/rgb_compare/retest-2025-12-28_04-48-25-k800`
  - surface: fill_holes=`True` (iters=`10`), smooth=`True` (iters=`2`)
  - frames_used: `7,13,21` phases: `press,slide,hold`
  - cmd: `python example/mpm_fem_rgb_compare.py --mode raw --record-interval 20 --indenter-type sphere --fric 0.4 --mpm-k-normal 800 --mpm-k-tangent 400 --fem-marker off --mpm-marker off --mpm-depth-tint off --export-intermediate --save-dir output/rgb_compare/retest-2025-12-28_04-48-25-k800`
- `output/rgb_compare/retest-2025-12-28_11-48-18-k800-filloff`
  - surface: fill_holes=`False` (iters=`10`), smooth=`True` (iters=`2`)
  - frames_used: `3,6,10` phases: `press,slide,hold`
  - cmd: `python example/mpm_fem_rgb_compare.py --mode raw --record-interval 40 --indenter-type sphere --fric 0.4 --mpm-k-normal 800 --mpm-k-tangent 400 --fem-marker off --mpm-marker off --mpm-depth-tint off --mpm-height-fill-holes off --mpm-height-smooth on --mpm-height-smooth-iters 2 --export-intermediate --save-dir output/rgb_compare/retest-2025-12-28_11-48-18-k800-filloff`
- `output/rgb_compare/retest-2025-12-28_05-11-10-k800-nosmooth`
  - surface: fill_holes=`True` (iters=`10`), smooth=`False` (iters=`2`)
  - frames_used: `7,13,21` phases: `press,slide,hold`
  - cmd: `python example/mpm_fem_rgb_compare.py --mode raw --record-interval 20 --indenter-type sphere --fric 0.4 --mpm-k-normal 800 --mpm-k-tangent 400 --fem-marker off --mpm-marker off --mpm-depth-tint off --mpm-height-smooth off --export-intermediate --save-dir output/rgb_compare/retest-2025-12-28_05-11-10-k800-nosmooth`

## Ablation table (>=4 combinations)

| fill_holes | smooth | note | finite_ratio_avg | motion_ratio_avg | spike_ratio_avg | spike_boundary_ratio_avg | grad_p99_avg | source |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| true (iters=10) | true (iters=2) | actual | 1 | 0.738783 | 0.195063 | 0 | 0.189212 | `output/rgb_compare/retest-2025-12-28_04-48-25-k800` |
| true (iters=10) | false (iters=2) | actual | 1 | 0.738783 | 0.195063 | 0 | 0.226979 | `output/rgb_compare/retest-2025-12-28_05-11-10-k800-nosmooth` |
| false (iters=10) | true (iters=2) | actual | 1 | 0.720775 | 0.177032 | 0 | 0.186413 | `output/rgb_compare/retest-2025-12-28_11-48-18-k800-filloff` |
| false (iters=10) | false (iters=2) | derived | 1 | 0.720775 | 0.177032 | 0 | 0.226514 | `output/rgb_compare/retest-2025-12-28_11-48-18-k800-filloff` |

## Recommendation

- recommended: fill_holes=`False` (iters=`10`), smooth=`True` (iters=`2`)
- basis: prefer lower spike_ratio/boundary_ratio (uv stability) + lower grad_p99 (halo proxy), while keeping finite_ratio high.

## Notes

- `smooth` 属于渲染前的高度场平滑，不会改变 `intermediate/height_field_mm` 的导出内容；因此这里采用“离线 box blur 后再统计 grad_p99”的方式评估其对 halo 的影响。
- 若需要人工对照帧图，可在对应 `save_dir` 下抽样查看 `mpm_*.png` 与表中指标是否一致。
