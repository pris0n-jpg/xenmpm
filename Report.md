# MPM vs FEM RGB 对比排查报告

> 生成时间：2025-12-26  
> 排查范围：`example/mpm_fem_rgb_compare.py`、`xengym/render/*`、`xengym/fem/simulation.py`、`xengym/assets/obj/*.STL`

## TL;DR（结论先行）

当前看到的“MPM vs FEM RGB 差异”里，**混入了大量“几何/轨迹/坐标/渲染策略不对齐”的差异**，因此还不能把差异直接归因到 MPM 本身的力学误差。

最影响对比可信度的点（按优先级）：

1. **压头几何体未对齐（基线失效）**  
   - MPM 物理压头：SDF `box`（默认）`example/mpm_fem_rgb_compare.py:148,1207-1211,1628-1666`  
   - FEM 侧压头：默认用 `circle_r4.STL` 做深度渲染/接触 `example/mpm_fem_rgb_compare.py:443-450`  
2. **`circle_r4.STL` 的“接触面”并不是半径 4mm 的圆**（而是 15mm 方形底座）  
   - 资产：`xengym/assets/obj/circle_r4.STL`（同系列 `circle_r3/r4/r5` 以及 `square_d6/d8` 也有同样底座）  
3. **摩擦系数不一致（会直接改变剪切带/粘滑表现）**  
   - FEM：`fric_coef = 0.4`（默认）`xengym/fem/simulation.py:233`  
   - MPM：`mu_s=2.0, mu_k=1.5` `example/mpm_fem_rgb_compare.py:1229-1234`  
4. **滑移方向/坐标翻转存在“多处补丁式取反”**（极易导致“强边缘”出现在相反侧）  
   - MPM 物理轨迹：`center0_x - dx_slide` `example/mpm_fem_rgb_compare.py:1363-1368`  
   - FEM 轨迹：直接使用 `+dx_slide` `example/mpm_fem_rgb_compare.py:1505-1508,1533-1534`  
   - MPM 渲染还会对 height_field 做水平翻转 `example/mpm_fem_rgb_compare.py:1035-1038`  
5. **渲染策略差异在放大观感差异**  
   - MPM 侧默认对纹理做 depth tint（压暗 G/B、增强 R）`example/mpm_fem_rgb_compare.py:987-1028,1049-1050`  
   - MPM marker 默认 `static`（不随面内位移移动），而 FEM marker 是“跟着网格/深度投影走”的（语义不同）`example/mpm_fem_rgb_compare.py:1632-1634`  
6. **log 里 FEM Frame 0~7 再出现一次不是“重复跑仿真”**  
   - UI 循环播放：`frame_idx = (frame_idx + 1) % total_frames` `example/mpm_fem_rgb_compare.py:1572`

---

## 1) 压头几何体不一致（对比基线失效）

### 1.1 代码级证据

- MPM 物理压头：默认 `--indenter-type box`，最终进入 obstacle 配置：  
  - CLI 默认值：`example/mpm_fem_rgb_compare.py:1628-1630`  
  - 写入场景参数：`example/mpm_fem_rgb_compare.py:1663-1666`  
  - 生成 obstacle：`example/mpm_fem_rgb_compare.py:1207-1211`
- FEM 侧压头：DepthRenderScene 未指定 `--object-file` 时默认加载 `circle_r4.STL`：  
  - `example/mpm_fem_rgb_compare.py:443-450`

### 1.2 影响

几何体不同会直接改变：

- 接触面积与边缘应力集中形态
- 滑移下剪切带（领先/滞后边）的空间位置
- RGB 渲染里的“亮/暗/彩边”分布

因此“RGB 不像”在当前配置下**不具备指向性**（无法判定是 MPM 物理误差还是几何差异）。

---

## 2) `circle_r4.STL` 并非“纯圆柱接触面”（底面是 15mm 方形底座）

### 2.1 资产几何数据（实测）

对 `xengym/assets/obj/circle_r4.STL` 做 STL 顶/底截面统计（单位为 m，括号内换算为 mm）：

- 全体包围盒：  
  - `bbox_min = (-0.0075, 0.0, -0.0075)`（x/z 最小约 -7.5mm，y 最小 0）  
  - `bbox_max = (+0.0075, 0.0200, +0.0075)`（x/z 最大约 +7.5mm，y 最大约 20mm）
- y 最小截面（`y=0`）的 x/z 外接范围：`±0.0075`（约 **15mm 方形底座**）  
- y 最大截面（`y≈0.02`）的 x/z 外接范围：`±0.0040`（约 **直径 8mm（r≈4mm）圆柱端面**）

同系列资产也呈现一致模式（示例）：

- `circle_r3.STL / circle_r4.STL / circle_r5.STL`：底部均为 15mm 方形底座，顶部才是 r=3/4/5mm 圆柱端面  
- `square_d6.STL / square_d8.STL / rhombus_d6.STL / tri_d6.STL`：底部同样为 15mm 方形底座，顶部才是对应形状端面

### 2.2 为什么你会看到“方框边缘”

FEM 深度渲染的“朝向”如果让 **y=0 的底座面**朝向相机/硅胶，则接触轮廓更像方形/方环，RGB 上会出现明显方框轮廓。

进一步的代码风险点是：DepthRenderScene 构造时写了 `.rotate(...).translate(...)`，但每帧 `set_object_pose()` 会 `setTransform()` 覆盖本地变换，使得“构造时的固定旋转/偏移”在运行中并不可靠：  
`example/mpm_fem_rgb_compare.py:438-450,470-474` + `xensesdk/xensesdk/ezgl/GLGraphicsItem.py:191-200`。

### 2.3 建议验证

- 用 MeshLab / Blender 打开 `xengym/assets/obj/circle_r4.STL`，确认 tip（r≈4mm 圆）与底座（±7.5mm 方形）的位置与朝向。
- FEM 侧做一个最小实验：将 STL 绕 X 或 Z 轴旋转 180°，让“tip 端面”朝向深度相机/硅胶表面，再看 RGB 里的方框是否消失（这是最快的定位实验）。

---

## 3) 轨迹与帧对齐：数值上对齐，但“符号/位姿定义”存在高风险

### 3.1 帧数与阶段边界（为什么 log 是 86 帧）

MPM 总步数：`press_steps + slide_steps + hold_steps = 150 + 240 + 40 = 430` `example/mpm_fem_rgb_compare.py:130-133`  
记录间隔：`record_interval=5` `example/mpm_fem_rgb_compare.py:1461-1463`  
理论帧数约 `430/5 = 86`，与你观察到的日志一致。阶段边界也会对应到：

- press：约 30 帧（含边界帧）
- slide：约 48 帧
- hold：约 8 帧

此外脚本会“循环播放”，所以你会看到 FEM Frame 0~7 再出现：`example/mpm_fem_rgb_compare.py:1572`。

### 3.2 FEM/MPM 共用同一份控制信号（这是正确方向）

FEM 侧优先用 `mpm_sim.frame_controls` 驱动 `press/slide`：`example/mpm_fem_rgb_compare.py:1503-1508`。  
这能避免“同一 frame index 不同位姿”的时间错位问题（至少在数值上对齐了 `press/slide` 两个标量）。

### 3.3 高风险点：滑移方向的符号被多处“就地修正”

目前代码同时存在这些行为：

1. **MPM 物理里滑移使用 `center0_x - dx_slide`**（即 `dx_slide` 越大，压头 x 越小）：  
   `example/mpm_fem_rgb_compare.py:1363-1368`
2. **但 `frame_controls` 存的是 `(+dz, +dx_slide)`**：  
   `example/mpm_fem_rgb_compare.py:1390-1392`
3. **FEM 渲染里直接使用 `+dx_slide`**：  
   `example/mpm_fem_rgb_compare.py:1505-1508,1533-1534`
4. **MPM 渲染又会对 height_field 做水平翻转**：  
   `example/mpm_fem_rgb_compare.py:1035-1038`
5. **MPM 的压头 overlay 再额外使用 `-slide_amount` 并在像素映射时对 x 取负**：  
   `example/mpm_fem_rgb_compare.py:1555-1558,1084-1086`

这种“把翻转补丁撒在物理/渲染/overlay 多处”的实现，一旦 FEM 侧或其它中间层的坐标约定不完全一致，就会导致：

- 同样的“向右滑”在两边变成相反的“领先/滞后边”
- RGB 上的强响应出现在相反侧（与你描述的现象吻合）

### 3.4 建议验证（不改大结构，先把事实打印出来）

建议把下面信息在 **MPM & FEM 同帧**都打印/记录（只要 1 次运行就能定位方向问题）：

- MPM：每帧 `solver.obstacle_centers[1]` 的 `(x,y,z)`（尤其是 `x` 的单调性）
- FEM：每帧传入 `DepthRenderScene.set_object_pose(x,y,z)` 的 `(x,y,z)`
- 同时保存一张带压头 overlay 的图（MPM 侧已有 `--mpm-show-indenter`；FEM 可临时加一个同样风格的 2D overlay）

如果两边“同帧”物理压头的 `x` 方向相反，那么当前对比图里出现“强边缘在相反侧”就**不该归因 MPM 力学**，而应先修正轨迹符号约定。

---

## 4) 摩擦参数未对齐（会显著改变滑移视觉特征）

### 4.1 代码级证据

- FEM：默认 `self.fric_coef = 0.4` `xengym/fem/simulation.py:233`
- MPM：`mu_s=2.0, mu_k=1.5` `example/mpm_fem_rgb_compare.py:1229-1234`

### 4.2 影响

摩擦系数会直接影响：

- 能否形成“粘住被拖拽”的剪切带
- 剪切带宽度/位置、以及是否出现明显滞后区
- marker 的面内位移幅值（warp 模式下差异更大）

在几何对齐之前，摩擦不对齐会让“左右边缘/彩色带”的差异进一步放大。

---

## 5) 渲染策略差异：MPM 的 depth tint / marker 语义与 FEM 不一致

### 5.1 MPM depth tint 可能导致“大块发黑/发红”

MPM 侧会在每帧对纹理叠加按压深度的红色热度（并压暗 G/B）：  
`example/mpm_fem_rgb_compare.py:987-1028`，由 `set_height_field()` 调用：`example/mpm_fem_rgb_compare.py:1049-1050`。

这会造成一种非常典型的观感差异：深压区域出现更大面积的暗/红块（即使力学行为本身并没有这么“黑”）。

### 5.2 marker 默认模式不一致

脚本默认 `--mpm-marker static`：`example/mpm_fem_rgb_compare.py:1632-1634`。  
这意味着 MPM 侧 marker 默认不随面内位移扭曲（而 FEM 侧 marker 由 FEM 网格/纹理投影决定，本质是“跟着表面走”的）。

如果用 default 参数看 raw RGB，对比会天然偏离。

### 5.3 建议验证

为了把“渲染策略差异”从“物理差异”里剥离出来，建议做两组对照：

1. 关闭 marker / 或两边都用白底：看纯光照+形变的差异是否还存在“大块发黑”
2. MPM 使用 `--mpm-marker warp`：再对比 FEM marker（语义更接近的一组）

---

## 6) log 里 FEM Frame 0~7 重复的原因（非重复仿真）

`example/mpm_fem_rgb_compare.py` 的 UI 更新函数每次都会打印当前帧号：  
`example/mpm_fem_rgb_compare.py:1533`，随后 `frame_idx` 使用取模回卷：`example/mpm_fem_rgb_compare.py:1572`。  
因此帧号会在跑完 0~85 后重新回到 0，再打印 0~7……这是预期行为。

---

## 7) 次要但值得确认的不对齐项（可能继续“污染”差异来源）

1. **FEM 侧 log/说明与实际不一致**  
   - `--object-file` 的 help 写“default: sphere”，但 DepthRenderScene 默认走 `circle_r4.STL`：`example/mpm_fem_rgb_compare.py:1608-1610,443-450`  
   - 启动时打印 `FEM indenter: circle_r4.STL` 是硬编码：`example/mpm_fem_rgb_compare.py:1689-1691`
2. **投影尺度可能不一致**  
   - `gel_size_mm=(17.3,29.15)`，但 DepthRenderScene 使用 `cam_view_width_m=19.4mm, cam_view_height_m=30.8mm`：`example/mpm_fem_rgb_compare.py:121-122,153-154,459-463`  
   - 这可能导致“深度图像素 ↔ gel 坐标”比例失配（需结合实际相机标定确认）
3. **MPM box 尺寸与 overlay 尺寸可能不一致**  
   - 默认 box half extents 用 `indenter_radius_mm=4mm`（即 8mm 方头）`example/mpm_fem_rgb_compare.py:146-149,1184-1186`  
   - overlay 默认 size=6mm（若未从 STL 推断）`example/mpm_fem_rgb_compare.py:1093-1095`

---

## 8) 建议的修复顺序（按性价比）

1. **先修 FEM 侧压头资产/朝向**：确保真正接触的是“tip 端面”，而不是 15mm 方形底座  
2. **让两边使用同一几何与尺寸**：  
   - 要么两边都用 `box`（并明确尺寸），要么两边都用“纯 tip STL”（需要新资产或对现 STL 做裁切/翻转）  
3. **对齐摩擦系数**（至少把数量级对齐，否则剪切视觉差异会非常大）  
4. **统一“slide 正方向”的约定**：把翻转集中到明确的坐标变换函数里，避免散落的 `-dx`  
5. **对齐渲染策略**：depth tint / marker 模式先拉齐，再讨论物理差异  
6. **补充中间产物输出**（用于把“渲染差异”与“物理差异”剥离）：  
   - MPM：`height_field_mm`、`uv_disp_mm` 强度图、接触 mask（可用 `height_field<阈值` 近似）  
   - FEM：`sensor_sim.get_height()`、FEM 接触节点/力（已有接口）  

---

## 9) 仍需补充的信息（用于把问题收敛到“可复现的单点 bug”）

建议记录这几项（写进日志即可）：

- 对比截图对应的 `frame_id`（以及当次运行的 CLI 参数）
- 当帧的：
  - MPM obstacle center（x,y,z）与 `frame_controls`（dz, dx_slide）
  - FEM object_pose（x,y,z）
- 是否启用了 `--mpm-marker warp`、是否启用了 `--mpm-show-indenter`

