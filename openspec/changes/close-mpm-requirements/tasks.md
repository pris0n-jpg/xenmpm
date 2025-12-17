# Tasks for close-mpm-requirements

1. [x] Baseline review
   - Read current mpm CLI/output/validation scripts; list existing scenes/outputs (ΔE_proj_step/E_proj_cum, contact curves).
   - Inventory stability checks (Drucker/time-step/contact stiffness) and their default behavior.
2. [x] Verification suite (FR-5 coverage)
   - Add scenes: 单轴拉伸、纯剪切+客观性（叠加刚体旋转）、能量守恒/投影收敛、GelSlim/incipient slip、Hertz/收敛。
   - Add corresponding CLI entry points and config templates.
   - Add post-processing scripts to plot required curves (应力-应变、客观性对比、能量收敛、切向力-位移、误差-步长/网格).
   - Validate each scene runs end-to-end (small particle count) and emits CSV/plots.
3. [x] Stability & guardrails
   - Default启用 Drucker-type 约束与路径扫描（可配置强/弱模式），在启动时阻断不稳定参数。
   - 时间步/接触刚度/粘性时间尺度检查，提供警告或阻断级别。
   - Update README/CLI help to reflect默认检查与禁用开关。
4. [x] 能量与输出完整性
   - 确保 ΔE_proj_step、E_proj_cum、E_viscous_*、切向力-位移、收敛数据统一写入 CSV。
   - 补充绘图/验证脚本生成图像，并在文档中列出期望曲线与合格标准。
5. [x] 自动微分/手工伴随扩展
   - 评估可行的 Maxwell/体粘性梯度方案：小规模数值差分 P_total 或保守阻断。
   - 实现可配置模式：纯 Ogden 快速路径、全 P_total 数值梯度（受规模/性能限制）、不支持时明确阻断。
   - 提供梯度验证脚本（对材料参数/初始状态在小规模场景下对比数值/解析），更新 README。
6. [x] 文档与示例
   - 更新 README/FINAL_STATUS 类文档，明确支持矩阵、限制、性能提示。
   - 新增示例运行指南（包含参数可调建议）和已知限制说明。
7. [x] Testing & validation
   - 为新增检查/模式添加单元或脚本式测试（覆盖 strict/warn 行为、模式切换、场景输出字段）。
   - 记录测试命令与结果，确保 openspec validate 通过。
