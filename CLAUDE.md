<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

使用conda环境xengym

## 项目概述

**Xengym + Xensesdk** - 触觉-视觉传感器仿真与开发框架

- **xengym**: 基于 IsaacGym 的机器人接触仿真器，含 FEM 有限元法引擎
- **xensesdk**: Xense 触觉传感器 SDK，支持深度/力/网格数据采集
- **calibration**: 材料参数贝叶斯优化标定工具

## 常用命令

### 环境安装
```bash
# 推荐 Python 3.9.19
conda create -n xense python=3.9.19
conda activate xense

# 可选 CUDA 支持
conda install cudatoolkit==11.8.0 cudnn==8.9.2.26

# 安装两个包（开发模式）
pip install -e .
pip install -e xensesdk/
```

### 运行演示
```bash
# 仿真演示 (IsaacGym + FEM + 3D 渲染)
xengym-demo --show-left --object-file xengym/assets/obj/circle_r4.STL

# 或直接运行
python example/demo_main.py \
  -f xengym/assets/data/fem_data_gel_2035.npz \
  -u xengym/assets/panda/panda_with_vectouch.urdf \
  -o xengym/assets/obj/letter.STL \
  -l -r

# 传感器演示（需要硬件或 .h5 数据文件）
python example/test.py
```

### 测试
```bash
# 冒烟测试
python quick_test.py

# 集成测试
python example/demo_main.py
python example/data_collection.py
```

### MPM vs FEM 比较工具
```bash
# FEM 模式 (仅加载预计算 FEM 数据)
python example/mpm_fem_compare.py --mode fem

# MPM 模式 (需要 Taichi 环境)
python example/mpm_fem_compare.py --mode mpm

# 并排比较 (生成对比曲线图到 output/mpm_fem_compare.png)
python example/mpm_fem_compare.py --mode both

# 自定义 FEM 文件
python example/mpm_fem_compare.py --fem-file path/to/fem_data.npz --mode both
```

### MPM vs FEM 传感器 RGB 比较
```bash
# 传感器 RGB 图像对比 (raw 模式 - 直接 RGB 输出)
python example/mpm_fem_rgb_compare.py --mode raw

# diff 模式 (相对参考帧的差分图像，增强形变可见性)
python example/mpm_fem_rgb_compare.py --mode diff

# 自定义按压/滑移参数
python example/mpm_fem_rgb_compare.py --press-mm 1.5 --slide-mm 5.0

# 保存帧序列到文件
python example/mpm_fem_rgb_compare.py --save-dir output/rgb_compare

# square 压头（推荐）：MPM marker 使用 warp（体现拉伸/压缩），并叠加压头投影辅助对齐
python example/mpm_fem_rgb_compare.py --mode raw --object-file xengym/assets/obj/square_d6.STL --indenter-type box --mpm-marker warp --mpm-show-indenter

# 位移场/warp 调试叠加（用于排查翻转、尺度、方向问题）
python example/mpm_fem_rgb_compare.py --mode raw --object-file xengym/assets/obj/square_d6.STL --indenter-type box --mpm-marker warp --mpm-debug-overlay warp
```

### 标定工具
```bash
# 位移平台数据采集
python calibration/collect_real_data.py --object circle_r4 --repeat 10 --port COM10

# 贝叶斯优化标定
python calibration/calibration.py
```

## 项目结构

```
xengym/                     # 核心仿真包
├── ezgym/                  # IsaacGym 包装层 (_env.py, assetFranka.py)
├── fem/                    # FEM 求解器 (simulation.py, simpleCSR.py)
├── render/                 # 3D 场景渲染 (calibScene.py, robotScene.py, sensorScene.py)
├── assets/                 # 数据资源 (URDF, STL, FEM 预计算数据)
└── main.py                 # CLI 入口 (xengym-demo)

xensesdk/                   # 传感器 SDK
├── xensesdk/
│   ├── __compile__.pyd/.so # C 扩展编译模块
│   ├── ezgl/               # OpenGL 3D 图形库
│   ├── xenseInfer/         # ONNX 推理引擎
│   └── xenseInterface/     # 传感器接口 + GUI
└── setup.py

calibration/                # 标定工具
├── calibration.py          # 贝叶斯优化主程序
├── bayesian_demo.py        # 高斯过程 + 获取函数
├── fem_processor.py        # FEM 批处理和缓存
├── collect_real_data.py    # 位移平台数据采集
└── stage_control_*.py      # 位移平台控制

example/                    # 示例脚本
└── test*.py                # 功能测试
```

## 核心 API

### Xensesdk 传感器
```python
from xensesdk import Sensor

sensor = Sensor.create('OP000064')  # 按 SN 创建
rectify, depth, force = sensor.selectSensorInfo(
    Sensor.OutputType.Rectify,      # 校正图像 (700, 400, 3)
    Sensor.OutputType.Depth,        # 深度图 (700, 400) mm
    Sensor.OutputType.Force         # 三维力 (35, 20, 3)
)
sensor.release()
```

### Xengym 仿真
```python
from xengym import Xensim
sim = Xensim(fem_path, urdf_path, object_path)
# IsaacGym + FEM 联合仿真
```

## 代码风格

- Python 3.9+，四空格缩进，PEP 8 命名
- `snake_case` 函数，`PascalCase` 类，大写常量
- 包内使用显式相对导入，避免通配符导入
- 文档字符串简洁英文，行内注释仅用于非显然的数学/标定逻辑
- 文件编码统一 UTF-8（无 BOM）

## OpenSpec 工作流

本项目使用 OpenSpec 进行规范驱动开发：
- 涉及新功能、破坏性变更、架构调整时，先查阅 `openspec/AGENTS.md`
- 使用 `openspec list` 查看进行中的变更
- 使用 `openspec validate [change-id] --strict` 验证提案

## 提交规范

- 格式: `<type>: <message>` (如 `feat: add diff image api`, `fix: handle empty asset path`)
- 确保每次提交构建通过且冒烟测试正常
- PR 需包含: 行为摘要、测试证据、资产变更说明

## 注意事项

- 大型原始采集数据存储在仓库外，仅提交复现所需的处理后资产
- STL 文件描述性命名 (`<shape>_<metric>.STL`)
- 新增 FEM 或标定变更时，在 `calibration/` 下记录基线并说明数据来源

回复用中文,but thinking in English
