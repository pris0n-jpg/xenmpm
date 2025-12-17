# 位移平台数据采集说明文档

## 概述

`collect_real_data.py` 是一个用于位移平台的真实数据采集脚本，替代了原有的 ABB 机器人采集方案（`abb_real_data.py`）。由于真实 ABB 机器人精度不足，改用位移平台进行更精确的数据采集。

## 主要变更

### 1. 硬件替换
- **原方案**: ABB 工业机器人（6自由度）
- **新方案**: 位移平台（主要使用 Z 轴进行压入，XY 轴在开始时设定好）

### 2. 通信方式
- **原方案**: TCP/IP 网络通信（pyabb库）
- **新方案**: 串口通信（serial库，波特率115200）

### 3. 运动控制
- **Z 轴（压入方向）**: 使用 `A` 命令控制（对应位移平台的 M0 电机）
- **X/Y 轴**: 实验开始前手动调整到合适位置
- **旋转轴**: 不使用

### 4. 保持不变的部分
- ✅ ATI 力传感器接口（IP: 192.168.1.10）
- ✅ Xense 触觉传感器接口
- ✅ 数据格式（与 calibration.py 完全兼容）
- ✅ 接触检测逻辑
- ✅ 零接触验证
- ✅ 轨迹配置文件（traj.json）

## 核心类说明

### StageController（位移平台控制器）

```python
class StageController:
    """位移平台控制器（替代ABB机器人）"""
    
    def __init__(self, port="COM10", baudrate=115200)
    def move_z_absolute(self, z_mm: float)  # Z轴绝对位置移动
    def move_z_relative(self, dz_mm: float)  # Z轴相对位置移动
    def home(self)  # 归零操作
    def disconnect(self)  # 断开串口
```

**串口命令格式**:
- `A{position}` - 移动 Z 轴到绝对位置（单位：mm）
- 示例: `A10.50\r\n` - 移动到 10.50mm

### StageDataCollector（数据采集器）

与 `ABBDataCollector` 接口基本一致，主要区别：
- 初始化时使用 `port` 参数指定串口，而非 `pose0` 机器人位姿
- 运动控制仅使用 Z 轴

## 使用方法

### 基本用法

```bash
# 采集 circle_r4 物体数据，重复10次，使用 COM10 串口
python collect_real_data.py --object circle_r4 --repeat 10 --port COM10
```

### 命令行参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--object` | 物体名称（必需） | - | `circle_r4` |
| `--repeat` | 每条轨迹重复次数 | 1 | `10` |
| `--port` | 位移平台串口 | `COM10` | `COM11` |
| `--config` | 自定义配置文件 | None | `config.yaml` |
| `--storage` | 数据存储文件 | `data/real_calibration_data.pkl` | - |
| `--overwrite` | 覆盖旧数据（标志） | False | 添加此标志 |
| `--dry-run` | 仅验证配置（标志） | False | 添加此标志 |

### 完整示例

```bash
# 示例1: 标准采集
python collect_real_data.py --object circle_r4 --repeat 10

# 示例2: 指定串口
python collect_real_data.py --object circle_r5 --repeat 5 --port COM11

# 示例3: 覆盖模式（清除旧数据）
python collect_real_data.py --object circle_r4 --repeat 10 --overwrite

# 示例4: 仅验证配置
python collect_real_data.py --object circle_r4 --dry-run
```

## 操作流程

### 1. 硬件准备

1. **连接位移平台**
   - 确保位移平台电源已打开
   - 使用 USB 转串口线连接计算机
   - 记录串口号（Windows: COMx, Linux: /dev/ttyUSBx）

2. **连接 ATI 力传感器**
   - 配置 IP: 192.168.1.10
   - 确保网络连通性

3. **连接 Xense 触觉传感器**
   - USB 连接
   - 确保驱动已安装

4. **手动调整 XY 位置**
   - 将触觉传感器移动到物体正上方
   - 确保 Z 轴下压时能准确接触物体中心

### 2. 软件准备

1. **检查轨迹配置**
   ```bash
   # 查看 calibration/obj/traj.json
   cat calibration/obj/traj.json
   ```

2. **验证配置（可选）**
   ```bash
   python collect_real_data.py --object circle_r4 --dry-run
   ```

### 3. 数据采集

1. **启动采集**
   ```bash
   python collect_real_data.py --object circle_r4 --repeat 10 --port COM10
   ```

2. **采集过程监控**
   - 观察终端输出的力值
   - 确认接触检测正常
   - 注意零接触验证结果

3. **异常处理**
   - 力过大自动停止
   - 零接触验证失败自动重试
   - 达到最大重试次数后强制保存

### 4. 数据验证

```bash
# 可视化采集数据
python calibration/data/visualize_real_calibration_data.py
```

## 配置参数说明

### 默认配置（可通过 YAML 文件自定义）

```yaml
contact_threshold: -0.025       # 接触力阈值（N）
approach_speed: 1.0             # 接近速度（mm/s）
press_speed: 0.5                # 压入速度（mm/s）
max_force: -1.0                 # 最大允许力（N）
data_frames: 40                 # 每步采集帧数
frame_interval: 0.1             # 帧间隔（秒）
step_settle_time: 0.3           # 步间等待时间（秒）
safe_offset_mm: 5.0             # 安全抬起高度（mm）
zero_contact_tolerance: 0.25    # 零接触验证容差（25%）
```

## 数据格式

输出数据格式与 `abb_real_data.py` 完全一致：

```python
{
    "物体名": {
        "traj_0_run0": {
            "step_000": {
                "marker_displacement": np.array,  # (20, 11, 2)
                "force_xyz": np.array,            # (3,)
                "metadata": {
                    "trajectory": "traj_0",
                    "run_id": 0,
                    "step_index": 0,
                    "commanded_delta_mm": (dx, dy, dz),
                    "timestamp": "2025-01-09T12:00:00"
                },
                "depth_field": None
            },
            ...
        },
        ...
    }
}
```

## 常见问题

### 1. 串口连接失败

**问题**: `RuntimeError: 无法连接到位移平台串口`

**解决方案**:
```bash
# Windows: 检查设备管理器中的 COM 端口
# Linux: 检查串口权限
sudo chmod 666 /dev/ttyUSB0

# 或添加用户到 dialout 组
sudo usermod -a -G dialout $USER
```

### 2. 接触检测失败

**问题**: `RuntimeError: 未检测到接触`

**原因**: 
- XY 位置偏移
- 力传感器未去皮
- 接触阈值设置不当

**解决方案**:
1. 手动调整 XY 位置
2. 重新去皮: `self.ati.tare()`
3. 调整 `contact_threshold` 参数

### 3. 零接触验证失败

**问题**: 反复重试零接触验证

**原因**:
- 初始接触力过小或过大
- 力变化不均匀

**解决方案**:
1. 调整接触位置，确保首次接触力合适（~0.03-0.10N）
2. 调整 `zero_contact_tolerance` 参数
3. 使用 `--overwrite` 模式清除异常数据

### 4. Z 轴运动异常

**问题**: 位移平台不响应或运动不准确

**解决方案**:
1. 检查串口波特率（应为 115200）
2. 手动发送命令测试: `echo -e "A10\r\n" > /dev/ttyUSB0`
3. 检查位移平台固件版本

## 与 ABB 版本的对比

| 特性 | ABB 版本 | 位移平台版本 |
|------|----------|--------------|
| 定位精度 | ±0.1mm | ±0.01mm |
| 重复性 | 中 | 高 |
| XYZ 控制 | 全自动 | Z 自动，XY 手动 |
| 通信方式 | TCP/IP | 串口 |
| 速度 | 快 | 中 |
| 成本 | 高 | 低 |
| 适用场景 | 大范围运动 | 精密压入测试 |

## 注意事项

1. **安全第一**
   - 首次使用时设置较小的 `max_force`
   - 随时准备按位移平台的急停按钮
   - 注意力传感器量程（通常±10N）

2. **数据质量**
   - 确保 XY 位置准确，避免偏心压入
   - 定期检查传感器标定
   - 环境温度稳定（避免热漂移）

3. **维护**
   - 定期清洁触觉传感器表面
   - 检查位移平台螺丝紧固
   - 备份采集数据

## 后续处理

采集完成后，数据可用于：

1. **标定**
   ```bash
   python calibration/calibration.py
   ```

2. **可视化**
   ```bash
   python calibration/data/visualize_real_calibration_data.py
   python calibration/data/visualize_force_comparison.py
   ```

3. **拟合**
   ```bash
   python calibration/data/fit_real_calibration_data.py
   ```

## 技术支持

如有问题，请检查：
1. `/calibration/abb_real_data.py` - 原版 ABB 采集脚本（参考）
2. `/calibration/标定台资料/` - 位移平台相关资料
3. `/calibration/STAGE_MIGRATION_GUIDE.md` - 迁移指南

---

**版本**: 1.0  
**日期**: 2025-01-09  
**作者**: 基于 abb_real_data.py 改编