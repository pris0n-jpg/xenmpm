# RGB Compare Sweep Report (press=1mm)

本报告基于已存在的 `output/rgb_compare/*` 目录做离线汇总（无需 taichi/ezgl）。
指标说明：

- `u_p50_avg/u_p90_avg`：接触区 `uv_disp_mm[...,0]` 分位数（帧 75/80/85 平均）。
- `|uv|_p50_avg/|uv|_p90_avg`：接触区位移幅值分位数（帧 75/80/85 平均）。
- `grad_p99_avg`：`height_field_mm` 梯度幅值的 p99（帧 75/80/85 平均），作为 halo_risk 的代理量。

## Summary Table

| name | press_mm | slide_mm | k_n | k_t | mu_s | mu_k | ogden_mu | ogden_kappa | u_p50_avg | u_p90_avg | |uv|_p50_avg | |uv|_p90_avg | grad_p99_avg | save_dir |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 3 | 800 | 400 | 0.4 | 0.4 | [2500.0] | 300000 | -0.00022711 | 0.208907 | 0.00533165 | 0.239488 | 0.467513 | output/rgb_compare/baseline |
| tune_kt2000 | 1 | 3 | 800 | 2000 | 0.4 | 0.4 | [2500.0] | 300000 | 7.53599e-05 | 0.198071 | 0.00540892 | 0.23946 | 0.467548 | output/rgb_compare/tune_kt2000 |
| tune_kt800 | 1 | 3 | 800 | 800 | 0.4 | 0.4 | [2500.0] | 300000 | -2.95112e-05 | 0.193795 | 0.00464951 | 0.23113 | 0.466358 | output/rgb_compare/tune_kt800 |
| tune_mu2 | 1 | 3 | 800 | 400 | 2 | 1.5 | [2500.0] | 300000 | -0.000252892 | 0.246647 | 0.00589878 | 0.283726 | 0.475384 | output/rgb_compare/tune_mu2 |
| tune_mu1 | 1 | 3 | 800 | 400 | 1 | 1 | [2500.0] | 300000 | -0.000277457 | 0.212792 | 0.0059384 | 0.246571 | 0.463242 | output/rgb_compare/tune_mu1 |
| tune_kn4000_kt2000 | 1 | 3 | 4000 | 2000 | 0.4 | 0.4 | [2500.0] | 300000 | -0.000806906 | 0.0726765 | 0.00692234 | 0.118534 | 0.675008 | output/rgb_compare/tune_kn4000_kt2000 |

## Recommendation

- baseline: `output/rgb_compare/baseline` u_p50_avg=`-0.00022711` grad_p99_avg=`0.467513`
- recommended: `output/rgb_compare/tune_kt2000` u_p50_avg=`7.53599e-05` (Δ=`0.00030247`) grad_p99_avg=`0.467548`

复跑命令（来自 run_manifest.json argv）：

```bash
python example\mpm_fem_rgb_compare.py --mode raw --record-interval 5 --press-mm 1.0 --slide-mm 3.0 --fric 0.4 --mpm-marker warp --mpm-depth-tint off --export-intermediate --mpm-k-tangent 2000 --save-dir output/rgb_compare/tune_kt2000
```
