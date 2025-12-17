# 采集流程对比：ABB vs 位移平台

## 核心采集流程对比

### 1. 初始化阶段

| 步骤 | ABB版本 (`abb_real_data.py`) | 位移平台版本 (`collect_real_data.py`) |
|------|-------------------------------|----------------------------------------|
| **硬件初始化** | ABBRobot (TCP/IP, 192.168.125.1) | StageController (串口, COM10, 115200) |
| **归位操作** | `robot.moveCart(pose0_init)` | `stage.home()` (发送 `A0` 命令) |
| **位置设定** | 6自由度位姿 `[x,y,z,qw,qx,qy,qz]` | Z轴归零，XY手动预设 |
| **ATI传感器** | ✅ 相同 (`ATISensor(ip="192.168.1.10")`) | ✅ 相同 |
| **触觉传感器** | ✅ 相同 (`TactileSensor()`) | ✅ 相同 |

### 2. 接触检测流程

**ABB版本**:
```python
def move_to_contact(self):
    # 1. 移动到预设高度
    self.move_to_xyz(556.58, -199.08, 115)
    # 2. 设置慢速
    self._safe_set_velocity(approach_speed, approach_speed)
    # 3. 相对移动下降
    while not is_contact:
        fz = self.get_ati_data()[2]
        if fz <= self.cont_th:  # -0.03N
            self.z_cont = self._safe_get_cartesian().z
            is_contact = True
        self.relative_move(z=0.02)  # 每次2mm
```

**位移平台版本**:
```python
def move_to_contact(self):
    # 1. 移动到安全高度
    safe_z = self.z_cont + 5.0 if self.z_cont else 10.0
    self.stage.move_z_absolute(safe_z)
    # 2. 缓慢下降
    step_size = 0.05  # 每次0.05mm (更精细)
    while not is_contact:
        fz = self.get_ati_data()[2]
        if fz <= self.cont_th:  # -0.03N
            self.z_cont = self.stage.current_z
            is_contact = True
        self.stage.move_z_relative(step_size)
```

**关键差异**:
- ✅ 检测逻辑完全相同（力阈值、循环结构）
- ✅ 步进量：ABB 2mm → 位移平台 0.05mm（**提升40倍精度**）
- ✅ 位置记录：ABB使用机器人坐标 → 位移平台使用Z轴坐标

### 3. 轨迹执行流程

**通用流程** (两个版本完全一致):
```
1. move_to_safe_height()          # 抬起到安全高度
2. move_to_contact()               # 下降到接触位置
3. for each step in trajectory:
   a. move_delta_xyz(dx, dy, dz)  # 执行增量运动
   b. _collect_current_step_data() # 采集数据
   c. _validate_zero_contact()     # step2后验证零接触
4. move_to_safe_height()          # 采集完成后抬起
```

**运动实现差异**:

ABB版本 (`move_delta_xyz`):
```python
def move_delta_xyz(self, dx=0, dy=0, dz=0):
    cp = self._safe_get_cartesian()
    target_pose = Affine(x=cp.x + dx, y=cp.y + dy, z=cp.z + dz, ...)
    self.robot.moveCart(target_pose)
    while self.robot.moving:
        time.sleep(self.step_settle_time)
```

位移平台版本 (`move_delta_xyz`):
```python
def move_delta_xyz(self, dx=0, dy=0, dz=0):
    """位移平台主要使用Z轴"""
    if dz != 0:
        self.stage.move_z_relative(dz)
    time.sleep(self.step_settle_time)
```

**关键差异**:
- ❌ ABB：XYZ三轴全自动控制
- ✅ 位移平台：**仅Z轴自动控制，忽略dx/dy**（符合设计要求）

### 4. 数据采集逻辑

**完全相同** (两个版本逐字相同):
```python
def _collect_current_step_data(self, metadata: Optional[Dict] = None) -> Dict:
    force_data_list = []
    marker_disp = self.sensor.get_data()  # 触觉marker位移
    for _ in range(self.data_frames):     # 默认40帧
        force_xyz = self.get_sensor_force_xyz()
        force_data_list.append(force_xyz)
        time.sleep(self.frame_interval)   # 默认0.1s
    avg_force = np.mean(force_data_list, axis=0)
    
    return {
        'marker_displacement': marker_disp.astype(np.float32),  # (20,11,2)
        'force_xyz': avg_force.astype(np.float32),              # (3,)
        'metadata': metadata or {},
        'depth_field': None
    }
```

### 5. 零接触验证

**完全相同** (公式和逻辑一致):
```python
def _validate_zero_contact(self, traj_data, traj_name) -> bool:
    step0_force = traj_data['step_000']['force_xyz'][2]
    step1_force = traj_data['step_001']['force_xyz'][2]
    step2_force = traj_data['step_002']['force_xyz'][2]
    
    delta1 = step1_force - step0_force
    delta2 = step2_force - step1_force
    delta_sum = delta1 + delta2
    
    # 验证公式：diff = |2*step0/(delta1+delta2) - 1|
    diff = abs(2 * step0_force / delta_sum - 1)
    
    return diff <= tolerance  # 默认25%
```

### 6. 重试机制

**完全相同**:
- ✅ 最大重试次数：150次/轨迹
- ✅ 零接触验证失败 → 自动重试
- ✅ step2后立即验证，失败则中止当前采集
- ✅ 最后一次尝试强制完成整条轨迹
- ✅ 实时保存机制（每条轨迹采集后立即写入）

### 7. 数据存储

**完全相同**:
```python
# 数据格式
{
    "物体名": {
        "traj_0_run0": {
            "step_000": {
                "marker_displacement": np.array,  # (20, 11, 2)
                "force_xyz": np.array,            # (3,)
                "metadata": dict,
                "depth_field": None
            },
            ...
        },
        ...
    }
}

# 存储文件
calibration/data/real_calibration_data.pkl
```

## 核心流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      初始化阶段                              │
├─────────────────────┬───────────────────────────────────────┤
│ ABB版本             │ 位移平台版本                           │
│ - 机器人TCP/IP连接  │ - 串口连接(COM10, 115200)             │
│ - 移动到初始位姿    │ - Z轴归零(A0命令)                      │
│ - 6轴位置控制       │ - 仅Z轴自动控制                        │
└─────────────────────┴───────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   ATI+触觉传感器初始化                       │
│              (两个版本完全相同)                              │
│  - ATI IP: 192.168.1.10                                    │
│  - Xense传感器 USB连接                                      │
│  - 传感器去皮(tare)                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      接触检测                                │
├─────────────────────┬───────────────────────────────────────┤
│ ABB: 2mm步进        │ 位移平台: 0.05mm步进 (精度提升40倍)   │
│ 力阈值: -0.03N      │ 力阈值: -0.03N (相同)                 │
└─────────────────────┴───────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   轨迹执行循环                               │
│  for each trajectory:                                       │
│    for run in repeat_count:                                 │
│      1. move_to_safe_height()                              │
│      2. move_to_contact()                                  │
│      3. for each step:                                     │
│         - move_delta_xyz(dx,dy,dz)                         │
│         - _collect_current_step_data()                     │
│         - [step2后] _validate_zero_contact()               │
│      4. move_to_safe_height()                              │
│      5. _save_single_trajectory() (实时保存)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  数据采集(每一步)                            │
│              (两个版本完全相同)                              │
│  - 采集40帧力数据(0.1s间隔)                                 │
│  - 采集marker位移数据                                       │
│  - 计算平均力                                                │
│  - 组装数据结构                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 零接触验证(step2后)                          │
│              (两个版本完全相同)                              │
│  diff = |2*F0/(F1-F0 + F2-F1) - 1|                         │
│  if diff > 25%: 重试                                        │
│  else: 继续采集                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    数据存储                                  │
│              (两个版本完全相同)                              │
│  - 格式: {object: {traj_X_runY: {step_ZZZ: {...}}}}       │
│  - 文件: calibration/data/real_calibration_data.pkl       │
│  - 实时保存 + cleanup补遗                                   │
└─────────────────────────────────────────────────────────────┘
```

## 主要差异总结

| 特性 | ABB版本 | 位移平台版本 | 影响 |
|------|---------|--------------|------|
| **硬件通信** | TCP/IP (pyabb) | 串口 (serial, 115200) | 接口不同，逻辑相同 |
| **运动控制** | 6轴全自动 (XYZ+姿态) | Z轴自动，XY手动 | 简化控制，提升Z精度 |
| **接触检测步进** | 2mm | 0.05mm | **精度提升40倍** |
| **定位精度** | ±0.1mm | ±0.01mm | **精度提升10倍** |
| **数据采集逻辑** | ✅ 相同 | ✅ 相同 | 