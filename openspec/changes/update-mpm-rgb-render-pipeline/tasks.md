## 1. 调研与对齐
- [ ] 1.1 复核现有 MPM 渲染链路与坐标映射
  - **Done when**: 明确记录 `(height_field row/col)` → `(mesh y/x)` → `(texture u/v)` 的翻转关系，并能用一个“右滑”场景断言方向正确。

## 2. 位移场提取（MPM → uv grid）
- [ ] 2.1 在 `example/mpm_fem_rgb_compare.py` 中实现顶面面内位移场提取
  - **Done when**: 对每帧输出 `uv_disp_mm (Ny,Nx,2)`，并在 debug 日志中输出 min/max/mean，确保 slide 阶段 `|u|` 增大。
- [ ] 2.2 空洞填补与轻量平滑
  - **Done when**: 在默认参数下不会出现大面积空洞导致的纹理破碎；平滑强度可配置或固定为轻量（避免过度模糊）。

## 3. Marker Warp（纹理随位移场运动）
- [ ] 3.1 实现 `warp` marker 模式（优先 cv2.remap）
  - **Done when**: `--mpm-marker warp` 下，slide 阶段 marker 会出现可见平移/剪切，与位移场方向一致。
- [ ] 3.2 无 cv2 的 fallback
  - **Done when**: 在 `HAS_CV2=False` 时仍能运行（可接受较低性能），输出不崩溃。
- [ ] 3.3 加入 marker 模式 CLI（off/static/warp）
  - **Done when**: 默认保持现有 static 行为；切换模式只影响 MPM 面板。

## 4. 压头可视化（可选）
- [ ] 4.1 实现 MPM 压头 2D overlay（默认关闭）
  - **Done when**: `--mpm-show-indenter` 后，MPM 面板显示压头投影轮廓且随轨迹移动；位置与 slide/press 参数一致。
- [ ] 4.2（可选增强）3D STL 压头渲染
  - **Done when**: 在用户明确选择 3D 模式时，压头几何体可渲染且不影响现有管线；否则不启用。

## 5. Debug Overlay（可诊断性）
- [ ] 5.1 增加 `--mpm-debug-overlay {off,uv,warp}` 并实现 overlay
  - **Done when**: `uv` 显示位移强度，`warp` 显示像素偏移统计/方向提示；能用于判定翻转/尺度是否正确。

## 6. 文档与验证
- [ ] 6.1 更新 `CLAUDE.md` 增补新参数使用示例
  - **Done when**: 文档包含至少 2 个命令：marker warp + indenter overlay。
- [ ] 6.2 增加轻量回归验证脚本或测试
  - **Done when**: 在无 GUI 环境可通过 `--save-dir` 保存少量帧，并验证：
    - slide 阶段相邻帧像素差异显著（marker/warp 生效）
    - press-only 场景 marker 平移接近 0（warp 不产生系统漂移）
- [ ] 6.3 `openspec validate update-mpm-rgb-render-pipeline --strict`
  - **Done when**: 严格校验无错误。

