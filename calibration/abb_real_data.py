#!/usr/bin/env python3
"""
真实机器人数据采集脚本
用于为calibration.py采集真实的触觉传感器数据
兼容calibration.py的数据格式要求

数据格式（支持多次采集）：
{
    "物体名": {
        "traj_0_run0": {  # 支持多次运行：traj_name_runX
            "step_000": {
                "marker_displacement": np.array,  # (20, 11, 2) marker位移
                "force_xyz": np.array,            # (3,) 三维力
                "metadata": dict,                 # 轨迹/步信息（含run_id）
                "depth_field": None
            },
            ...
        },
        "traj_0_run1": {...},  # 同一轨迹的第二次采集
        ...
    }
}
"""

import argparse
import cv2
import sys
import os
import time
from time import sleep
import numpy as np
import pandas as pd
from datetime import datetime
from threading import Thread
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
import pickle
import copy

from pyabb import ABBRobot, Logger, Affine
from pyati.ati_sensor import ATISensor
from xensesdk import Sensor
from xensesdk import ExampleView

# 添加项目路径
PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJ_DIR))

from example.MarkerInterp import MarkerInterpolator

logger = Logger(log_level='DEBUG', name="ABB_Real_Data", log_path=None)

TIME_STAMP = str(datetime.now().strftime('%y_%m_%d__%H_%M_%S'))


def _load_available_objects(traj_path: Path) -> List[str]:
    if not traj_path.exists():
        return []
    try:
        with open(traj_path, 'r', encoding='utf-8') as fp:
            cfg = json.load(fp)
        return list(cfg.keys()) if isinstance(cfg, dict) else []
    except Exception:
        return []


class TactileSensor():
    """真实触觉传感器管理类，适配calibration数据格式"""
    def __init__(self):
        """初始化xense触觉传感器"""
        self.sensor = Sensor.create(0)
        marker_init = self.sensor.selectSensorInfo(Sensor.OutputType.Marker2DInit)
        self.marker_interpolator = MarkerInterpolator(marker_init)

    def get_data(self):
        """获取传感器数据"""
        marker_2D = self.sensor.selectSensorInfo(Sensor.OutputType.Marker2D)
        marker_displacement = self.marker_interpolator.interpolate(marker_2D)
        return marker_displacement

    def release(self):
        """释放传感器"""
        self.sensor.release()


class ABBDataCollector():
    """ABB机器人数据采集器"""

    def _safe_set_velocity(self, v_tcp, v_ori):
        """安全设置机器人速度，处理通信返回None的情况"""
        try:
            self.robot.set_velocity(v_tcp, v_ori)
        except AttributeError as e:
            if "'NoneType' object has no attribute 'decode'" in str(e):
                logger.warning(f"机器人通信返回None，忽略set_velocity({v_tcp}, {v_ori})错误")
            else:
                raise

    def _safe_get_cartesian(self):
        """安全获取机器人当前位置，处理通信返回None的情况"""
        max_retries = 3
        retry_delay = 0.5  # 秒

        for attempt in range(max_retries):
            try:
                return self.robot.get_cartesian()
            except (TypeError, AttributeError) as e:
                if "a bytes-like object is required, not 'NoneType'" in str(e) or \
                   "'NoneType' object has no attribute" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"机器人通信返回None，获取位置失败，{retry_delay}秒后重试 ({attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"机器人通信返回None，获取位置失败，已达最大重试次数")
                        # 返回一个默认位置，避免程序崩溃
                        return Affine(x=556.58, y=-199.08, z=115, a=0, b=1, c=0)
                else:
                    raise

    def __init__(self, pose0, object_name="cube", config_file=None, storage_file=None, repeat_count=1, overwrite=False):
        """
        初始化数据采集器

        Args:
            pose0: 初始位置 [x, y, z, qw, qx, qy, qz]
            object_name: 物体名称
            config_file: 配置文件路径
            storage_file: 数据存储文件路径
            repeat_count: 每条轨迹重复采集次数
            overwrite: 是否覆盖之前的所有运行记录
        """
        self.object_name = object_name
        self.repeat_count = max(1, repeat_count)  # 至少执行1次
        self.overwrite = overwrite
        self.max_retry_per_trajectory = 150  # 每条轨迹最大重试次数

        # 加载配置
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        self.step_settle_time = float(self.config.get('step_settle_time', 0.3))
        self.safe_offset_mm = float(self.config.get('safe_offset_mm', 1.0))
        self.frame_interval = float(self.config.get('frame_interval', 0.1))
        self.data_frames = int(self.config.get('data_frames', 30))
        self.trajectory_config = self._load_trajectory_config()

        # 数据汇总文件
        default_storage = PROJ_DIR / "calibration" / "data" / "real_calibration_data.pkl"
        self.storage_file = Path(storage_file) if storage_file else default_storage
        self.storage_file.parent.mkdir(exist_ok=True, parents=True)

        # 初始化机器人
        self.robot = ABBRobot(
            ip="192.168.125.1",
            port_motion=5000,
            port_logger=5001,
            port_signal=5002,
        )
        logger.warning("Connect to Server")
        self.robot.initialize()

        # 设置运动参数
        self.robot.set_acceleration(0.5, 0.5)
        self._safe_set_velocity(20, 20)

        pose0_init = [574.33, -176.67, 194.89, 0, 1, 0, 0]  # 默认初始位置
        self.robot.moveCart(pose0_init)
        self._check_joint_limit()
        time.sleep(1)

        # 移动到初始位置
        self.pose0 = pose0
        self.robot.moveCart(self.pose0)

        time.sleep(1)

        logger.info(f"init pose: {self._safe_get_cartesian()}")
        logger.info(f"init velocity: {self.robot.get_velocity()}")

        # 初始化ATI传感器
        self.ati = ATISensor(ip="192.168.1.10", filter_on=False)
        time.sleep(2)
        self.ati.tare()

        # 初始化触觉传感器
        self.sensor = TactileSensor()
        self.rot_sensor = (Affine(a=180)*Affine(a=-90,c=180).inverse()*Affine(a=-45)).rotation()
        # self.View = ExampleView(self.sensor.sensor)
        # self.View2d = self.View.create2d(Sensor.OutputType.Difference, Sensor.OutputType.Depth)
        # def callback():
        #     src, diff, depth = self.sensor.sensor.selectSensorInfo(
        #         Sensor.OutputType.Rectify,
        #         Sensor.OutputType.Difference,
        #         Sensor.OutputType.Depth
        #     )
        #     marker_img = self.sensor.sensor.drawMarkerMove(src)
        #     self.View2d.setData(Sensor.OutputType.Difference, diff)
        #     self.View2d.setData(Sensor.OutputType.Depth, depth)
        # self.View.setCallback(callback)
        # self.View.show()

        # 接触检测参数
        self.z_cont = None  # 接触位置，将在运行时确定
        self.cont_th = self.config.get('contact_threshold', -0.03)

        # 存储采集的数据 (calibration.py格式)
        self.calibration_data = {}

    def _get_default_config(self):
        """获取默认配置"""
        return {
            'contact_threshold': -0.025,
            'approach_speed': 1,  # mm/s
            'press_speed': 1,      # mm/s
            'max_force': -1,      # N
            'data_frames': 40,     # 每步采集数据帧数
            'frame_interval': 0.1,  # 帧间隔时间 s
            'step_settle_time': 0.3,  # 每步运动后的等待时间 s
            'safe_offset_mm': 8.0,    # 安全抬起高度 mm
            'zero_contact_tolerance': 0.25  # 零接触验证容差（25%）
        }

    def _load_trajectory_config(self) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
        """读取轨迹配置"""
        traj_path = PROJ_DIR / "calibration" / "obj" / "traj.json"
        if not traj_path.exists():
            logger.warning(f"未找到轨迹配置文件: {traj_path}")
            return {}

        try:
            with open(traj_path, 'r', encoding='utf-8') as fp:
                raw_config = json.load(fp)
        except Exception as exc:
            logger.error(f"轨迹配置解析失败: {exc}")
            return {}

        normalized: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

        for obj_name, traj_dict in raw_config.items():
            if not isinstance(traj_dict, dict):
                continue

            obj_trajs: Dict[str, List[Dict[str, float]]] = {}

            for traj_name, steps_payload in traj_dict.items():
                steps: List[Dict[str, float]] = []

                if isinstance(steps_payload, list):
                    for entry in steps_payload:
                        if isinstance(entry, dict):
                            dx = float(entry.get("x", 0.0))
                            dy = float(entry.get("y", 0.0))
                            dz = float(entry.get("z", 0.0))
                            steps.append({"x": dx, "y": dy, "z": dz})
                        elif isinstance(entry, (list, tuple)) and len(entry) == 3:
                            dx, dy, dz = map(float, entry)
                            steps.append({"x": dx, "y": dy, "z": dz})
                elif isinstance(steps_payload, dict) and {"x", "y", "z"} <= steps_payload.keys():
                    x_seq = steps_payload.get("x", [])
                    y_seq = steps_payload.get("y", [])
                    z_seq = steps_payload.get("z", [])
                    for dx, dy, dz in zip(x_seq, y_seq, z_seq):
                        steps.append({"x": float(dx), "y": float(dy), "z": float(dz)})

                if steps:
                    obj_trajs[traj_name] = steps

            if obj_trajs:
                normalized[obj_name] = obj_trajs

        if not normalized:
            logger.warning("轨迹配置中没有可用的轨迹")

        return normalized

    def _check_joint_limit(self):
        """检查关节限位"""
        current_joint = self.robot.get_joint()
        if current_joint[5] > 180:
            self.robot.moveJoint(
                current_joint[0], current_joint[1], current_joint[2],
                current_joint[3], current_joint[4], current_joint[5] - 360
            )
        elif current_joint[5] < -180:
            self.robot.moveJoint(
                current_joint[0], current_joint[1], current_joint[2],
                current_joint[3], current_joint[4], current_joint[5] + 360
            )

    def get_robot_xyz(self):
        """获取机器人当前位置"""
        pose = self._safe_get_cartesian()
        return pose.x, pose.y, pose.z

    def get_ati_data(self):
        """获取ATI传感器数据"""
        return self.ati.data.copy()

    def get_sensor_force_xyz(self):
        force_xyz = self.get_ati_data()[0:3]
        return self.rot_sensor @ force_xyz

    def move_to_xyz(self, x, y, z):
        """移动到指定位置"""
        cp = self._safe_get_cartesian()
        target_pose = Affine(x=x, y=y, z=z, a=cp.a, b=cp.b, c=cp.c)
        self.robot.moveCart(target_pose)
        while self.robot.moving:
            time.sleep(self.step_settle_time)

    def move_delta_xyz(self, dx=0, dy=0, dz=0):
        """移动到指定位置"""
        cp = self._safe_get_cartesian()
        target_pose = Affine(x=cp.x + dx, y=cp.y + dy, z=cp.z + dz, a=cp.a, b=cp.b, c=cp.c)
        self.robot.moveCart(target_pose)
        while self.robot.moving:
            time.sleep(self.step_settle_time)


    def relative_move(self, x=0, y=0, z=0, Rz=0, Ry=0, Rx=0):
        """相对移动"""
        cp = self._safe_get_cartesian()
        target_pose = (Affine(x=cp.x, y=cp.y, z=cp.z, a=cp.a, b=cp.b, c=cp.c) *
                      Affine(x=x, y=y, z=z, a=Rz, b=Ry, c=Rx))
        self.robot.moveCart(target_pose)
        while self.robot.moving:
            time.sleep(0.01)


    def move_to_contact(self):
        """移动到刚好接触的位置"""
        logger.info("开始寻找接触位置...")
        self._safe_set_velocity(20, 20)
        cp = self._safe_get_cartesian()
        if self.z_cont is not None:
            self.move_to_xyz(556.58, -199.08, self.z_cont + 0.8)
        else:
            self.move_to_xyz(556.58, -199.08, 115)  # 默认位置
        # pose_contact = (556.58, -199.08, 115)
        # self.robot.moveCart([*pose_contact,0,1,0,0])
        # self.move_to_xyz(556.58, -199.14, 114)
        time.sleep(1)
        # 设置较慢的接近速度
        self._safe_set_velocity(self.config['approach_speed'], self.config['approach_speed'])

        is_contact = False

        while not is_contact:
            # 安全检测
            fz = self.get_ati_data()[2]
            if fz <= self.config['max_force']:
                logger.error(f'力过大，退出: {fz}N')
                raise RuntimeError(f'Force too large: {fz}N')

            fz_current = self.get_ati_data()[2]
            # logger.debug(f'ATI Z方向力: {fz_current}N')

            # 检测接触
            if fz_current <= self.cont_th:
                self.z_cont = self._safe_get_cartesian().z
                logger.info(f"检测到接触，接触位置: {self.z_cont}mm")
                is_contact = True
                break

            # 向下移动
            # self.move_delta_xyz(dz=-0.01)
            self.relative_move(z=0.02)
            time.sleep(0.2)

        if not is_contact:
            raise RuntimeError("未检测到接触")

        self.relative_move(z=-0.1)
        # self.move_to_xyz(556.58, -199.08, self.z_cont)
        if np.abs(self.z_cont -113.54) >= 2:
            logger.info(f"修正接触位置: {self.z_cont} -> 113.54 mm")
            self.z_cont = 113.54
        self.robot.moveCart([556.58, -199.08, self.z_cont+0.15, 0, 1, 0, 0])
        time.sleep(0.5)

        # 恢复正常速度
        self._safe_set_velocity(self.config['press_speed'], self.config['press_speed'])

    def collect_calibration_data(self) -> Dict[str, Dict[str, Dict]]:
        """按轨迹采集真实触觉数据（支持多次重复采集）"""
        logger.info(f"开始采集 {self.object_name} 的轨迹数据（重复 {self.repeat_count} 次）...")

        trajectories = self.trajectory_config.get(self.object_name, {})
        if not trajectories:
            logger.warning(f"物体 {self.object_name} 未在轨迹配置中找到，跳过")
            self.calibration_data[self.object_name] = {}
            return {self.object_name: {}}

        # 如果是覆盖模式，在采集前就清空该物体的旧数据
        if self.overwrite:
            storage = self._load_storage()
            if self.object_name in storage:
                old_count = len(storage[self.object_name])
                logger.warning(f"🗑️  覆盖模式：清空 {self.object_name} 的 {old_count} 条旧记录")
                storage[self.object_name] = {}
                self._save_storage(storage)
            else:
                logger.info(f"覆盖模式：{self.object_name} 无旧数据")

        object_data: Dict[str, Dict[str, Dict]] = {}

        # 多次重复采集
        for run_idx in range(self.repeat_count):
            logger.info(f"===== 开始第 {run_idx + 1}/{self.repeat_count} 轮采集 =====")

            for traj_name, steps in trajectories.items():
                retry_count = 0
                success = False

                while retry_count < self.max_retry_per_trajectory and not success:
                    try:
                        # 获取已存在的运行编号，自动递增
                        next_run_id = self._get_next_run_id(traj_name)
                        traj_key_with_run = f"{traj_name}_run{next_run_id}"

                        retry_info = f" (重试 {retry_count}/{self.max_retry_per_trajectory})" if retry_count > 0 else ""
                        logger.info(f"执行 {traj_key_with_run}{retry_info}")

                        # 判断是否为最后一次尝试
                        is_last_attempt = (retry_count == self.max_retry_per_trajectory - 1)
                        traj_data = self._execute_trajectory(traj_name, steps, run_id=next_run_id, is_last_attempt=is_last_attempt)

                        if traj_data is not None:
                            # 检查是否验证失败
                            validation_failed = traj_data.pop('_validation_failed', False)

                            if validation_failed and retry_count < self.max_retry_per_trajectory - 1:
                                # 验证失败且未达最大重试次数，重试
                                retry_count += 1
                                logger.warning(f"⚠️  准备重新采集 {traj_key_with_run}...")
                                time.sleep(2)
                            else:
                                # 验证通过 或 达到最大重试次数（强制保存）
                                object_data[traj_key_with_run] = traj_data
                                if validation_failed:
                                    logger.warning(f"⚠️  {traj_key_with_run} 达到最大重试次数，保存当前数据")
                                else:
                                    logger.info(f"✓ {traj_key_with_run} 采集完成并通过验证")

                                # 立即保存到存储文件，防止意外中断导致数据丢失
                                self._save_single_trajectory(traj_key_with_run, traj_data)
                                success = True
                        else:
                            # traj_data 为 None 说明轨迹为空
                            retry_count += 1
                            if retry_count < self.max_retry_per_trajectory:
                                logger.warning(f"⚠️  轨迹为空，准备重试...")
                                time.sleep(2)
                            else:
                                logger.error(f"❌ {traj_key_with_run} 轨迹为空且达到最大重试次数，跳过")

                    except Exception as exc:
                        logger.error(f"轨迹 {traj_key_with_run} (尝试{retry_count+1}) 执行失败: {exc}")
                        import traceback
                        traceback.print_exc()
                        retry_count += 1
                        if retry_count < self.max_retry_per_trajectory:
                            time.sleep(2)

            if run_idx < self.repeat_count - 1:
                logger.info(f"第 {run_idx + 1} 轮完成，准备下一轮...")
                time.sleep(1)

        self.move_to_safe_height()
        self.calibration_data[self.object_name] = object_data
        logger.info(f"物体 {self.object_name} 采集完成，共 {len(object_data)} 条记录")
        return {self.object_name: object_data}

    def _get_next_run_id(self, traj_name: str) -> int:
        """获取下一个可用的运行编号（同时检查存储文件和内存中的数据）"""
        existing_run_ids = []

        # 1. 从存储文件中读取已有的 run_id
        storage = self._load_storage()
        obj_data = storage.get(self.object_name, {})
        for key in obj_data.keys():
            if key.startswith(f"{traj_name}_run"):
                try:
                    run_id = int(key.split("_run")[-1])
                    existing_run_ids.append(run_id)
                except ValueError:
                    pass

        # 2. 从内存中的 calibration_data 读取本次运行已采集的 run_id
        memory_obj_data = self.calibration_data.get(self.object_name, {})
        for key in memory_obj_data.keys():
            if key.startswith(f"{traj_name}_run"):
                try:
                    run_id = int(key.split("_run")[-1])
                    existing_run_ids.append(run_id)
                except ValueError:
                    pass

        return max(existing_run_ids, default=-1) + 1

    def _validate_zero_contact(self, traj_data: Dict[str, Dict], traj_name: str) -> bool:
        """
        验证是否从接近零接触开始采集

        检查前三步的法向力（force[2]）：
        - step0: 初始接触力
        - delta1 = step1 - step0
        - delta2 = step2 - step1

        验证公式：diff = |2*step0/(delta1+delta2) - 1|
        如果 diff ≤ 容差（默认25%），则认为接近零接触

        Args:
            traj_data: 轨迹数据字典
            traj_name: 轨迹名称（用于日志）

        Returns:
            bool: 是否通过零接触验证
        """
        tolerance = self.config.get('zero_contact_tolerance', 0.25)

        # 至少需要3步数据
        if len(traj_data) < 3:
            logger.warning(f"{traj_name}: 数据不足3步，跳过零接触验证")
            return True

        try:
            # 提取前三步的法向力 force_xyz[2]
            step0_force = traj_data['step_000']['force_xyz'][2]
            step1_force = traj_data['step_001']['force_xyz'][2]
            step2_force = traj_data['step_002']['force_xyz'][2]

            # 计算力的变化
            delta1 = step1_force - step0_force  # step1相对step0的变化
            delta2 = step2_force - step1_force  # step2相对step1的变化
            delta_sum = delta1 + delta2

            # 初始力极小（< 0.01N）时直接重试
            if abs(step0_force) < 0.01:
                logger.warning(
                    f"{traj_name}: ✗ 初始力过小 ({step0_force:.4f}N < 0.01N)，需要重试"
                )
                return False

            # 避免除零
            if abs(delta_sum) < 1e-6:
                logger.warning(
                    f"{traj_name}: ✗ 力变化过小 (Δ1+Δ2={delta_sum:.6f}N)，需要重试"
                )
                return False

            # 计算差异：diff = |2*step0/(delta1+delta2) - 1|
            diff = abs(2 * step0_force / delta_sum - 1)

            # 判断是否在容差范围内
            if diff <= tolerance:
                logger.info(
                    f"{traj_name}: ✓ 零接触验证通过 "
                    f"[step0={step0_force:.4f}N, Δ1={delta1:.4f}N, Δ2={delta2:.4f}N, "
                    f"差异={diff:.1%}, 容差={tolerance:.0%}]"
                )
                return True
            else:
                logger.warning(
                    f"{traj_name}: ✗ 零接触验证失败 "
                    f"[step0={step0_force:.4f}N, Δ1={delta1:.4f}N, Δ2={delta2:.4f}N, "
                    f"差异={diff:.1%}, 容差={tolerance:.0%}]"
                )
                return False

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"{traj_name}: 零接触验证数据提取失败: {e}")
            return False

    def _execute_trajectory(self, trajectory_name: str, steps: List[Dict[str, float]], run_id: int = 0, is_last_attempt: bool = False) -> Optional[Dict[str, Dict]]:
        """
        执行单条轨迹并采集每一步的数据（采集前3步后立即验证零接触）

        Args:
            trajectory_name: 轨迹名称
            steps: 轨迹步骤列表
            run_id: 运行编号
            is_last_attempt: 是否为最后一次尝试（最后一次时即使验证失败也继续采集完整轨迹）

        Returns:
            成功时返回轨迹数据字典，失败时返回带 '_validation_failed' 标记的字典或 None
        """
        if not steps:
            return None

        logger.info(f"执行轨迹 {trajectory_name} (run{run_id})，共 {len(steps)} 步")

        self.move_to_safe_height()
        time.sleep(self.step_settle_time)
        self.move_to_contact()
        time.sleep(self.step_settle_time)

        trajectory_data: Dict[str, Dict] = {}
        traj_key_with_run = f"{trajectory_name}_run{run_id}"

        for idx, step in enumerate(steps):
            dx = float(step.get('x', 0.0))
            dy = float(step.get('y', 0.0))
            dz = float(step.get('z', 0.0))

            self.move_delta_xyz(dx=dx, dy=dy, dz=dz)

            metadata = {
                'trajectory': trajectory_name,
                'run_id': run_id,
                'step_index': idx,
                'commanded_delta_mm': (dx, dy, dz),
                'timestamp': datetime.now().isoformat()
            }

            step_data = self._collect_current_step_data(metadata=metadata)
            if step_data:
                step_key = f"step_{idx:03d}"
                trajectory_data[step_key] = step_data

            # 在采集完step2后立即进行零接触验证
            if idx == 2:  # step_002 刚采集完
                if not self._validate_zero_contact(trajectory_data, traj_key_with_run):
                    if is_last_attempt:
                        # 最后一次尝试：验证失败但继续采集完整轨迹
                        logger.warning(f"⚠️  {traj_key_with_run} 零接触验证失败（最后一次尝试，继续采集）")
                        trajectory_data['_validation_failed'] = True
                    else:
                        # 非最后一次：立即中止采集并返回失败标记
                        logger.warning(f"⚠️  {traj_key_with_run} 零接触验证失败，立即中止当前采集")
                        trajectory_data['_validation_failed'] = True
                        return trajectory_data  # 立即返回，触发重试

        time.sleep(self.step_settle_time)

        logger.info(f"轨迹 {trajectory_name} (run{run_id}) 完成，采集 {len(trajectory_data)} 条数据")
        return trajectory_data

    def _collect_current_step_data(self, metadata: Optional[Dict] = None) -> Dict:
        """采集当前姿态下的数据"""
        try:
            force_data_list = []
            marker_disp = self.sensor.get_data()
            for _ in range(self.data_frames):
                force_xyz = self.get_sensor_force_xyz()
                force_data_list.append(force_xyz)
                time.sleep(self.frame_interval)
            avg_force = np.mean(force_data_list, axis=0)

            data = {
                'marker_displacement': marker_disp.astype(np.float32),
                'force_xyz': avg_force.astype(np.float32),
                'metadata': metadata or {},
                'depth_field': None
            }

            logger.debug(f"step {metadata['step_index'] if metadata else 'unknown'}: force={avg_force}")
            return data

        except Exception as e:
            logger.error(f"当前步数据采集失败: {e}")
            raise

    def move_to_safe_position(self):
        """移动到安全位置"""
        self._safe_set_velocity(20, 20)
        safe_z = self.pose0[2] + max(self.safe_offset_mm, 50)
        current_pose = self._safe_get_cartesian()
        self.move_to_xyz(current_pose.x, current_pose.y, safe_z)
        time.sleep(1)

    def move_to_safe_height(self):
        """移动到安全高度（相对于接触位置）"""
        self._safe_set_velocity(20, 20)
        if self.z_cont is not None:
            safe_z = self.z_cont + self.safe_offset_mm
        else:
            safe_z = self.pose0[2]

        current_pose = self._safe_get_cartesian()
        self.move_to_xyz(556.58, -199.08, safe_z)
        time.sleep(0.5)

    def _load_storage(self) -> Dict:
        if not self.storage_file.exists():
            return {}
        try:
            with open(self.storage_file, 'rb') as fp:
                data = pickle.load(fp)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.error(f"读取汇总文件失败: {exc}")
            return {}

    def _save_storage(self, data: Dict):
        with open(self.storage_file, 'wb') as fp:
            pickle.dump(data, fp)
        logger.info(f"汇总数据已写入: {self.storage_file}")

    def _save_single_trajectory(self, traj_key_with_run: str, traj_data: Dict):
        """
        立即保存单条轨迹数据到存储文件

        Args:
            traj_key_with_run: 轨迹键名（格式：traj_0_run0）
            traj_data: 轨迹数据字典
        """
        try:
            storage = self._load_storage()
            storage.setdefault(self.object_name, {})

            # 保存单条轨迹
            storage[self.object_name][traj_key_with_run] = copy.deepcopy(traj_data)

            # 立即写入文件
            self._save_storage(storage)
            logger.info(f"💾 已立即保存: {self.object_name}/{traj_key_with_run} ({len(traj_data)} steps)")

        except Exception as e:
            logger.error(f"立即保存轨迹失败: {e}")
            # 不抛出异常，继续采集

    def save_calibration_data(self):
        """
        将采集结果写入统一汇总文件（作为最终确认，实际数据已在采集时实时保存）
        此方法主要用于 cleanup 时的最终检查和补遗
        """
        if not self.calibration_data:
            logger.info("所有数据已在采集时实时保存")
            return self.storage_file

        storage = self._load_storage()
        saved_count = 0

        for obj_name, obj_data in self.calibration_data.items():
            storage.setdefault(obj_name, {})

            for traj_key_with_run, traj_steps in obj_data.items():
                # 检查是否已保存（实时保存时已写入）
                if traj_key_with_run not in storage[obj_name]:
                    storage[obj_name][traj_key_with_run] = copy.deepcopy(traj_steps)
                    saved_count += 1
                    logger.debug(f"补遗保存: {obj_name}/{traj_key_with_run}")

        if saved_count > 0:
            self._save_storage(storage)
            logger.info(f"✓ cleanup 补遗保存 {saved_count} 条轨迹记录")
        else:
            logger.info(f"✓ 所有 {len(self.calibration_data.get(self.object_name, {}))} 条轨迹已在采集时实时保存")

        return self.storage_file

    def cleanup(self):
        """清理资源"""
        logger.info("清理资源...")

        # 移动到安全位置
        self.move_to_safe_position()

        # 保存数据
        if self.calibration_data:
            self.save_calibration_data()

        # 释放传感器
        try:
            self.sensor.release()
        except:
            pass

        # 关闭机器人马达
        try:
            time.sleep(1)
            self.robot.sig_motor_off()
        except:
            pass

        logger.info("清理完成")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABB真实触觉数据采集（支持多次重复采集）")
    parser.add_argument("--object", required=True, default="circle_r3", help="需要采集的物体名称，与traj.json保持一致")
    parser.add_argument("--pose", nargs=7, type=float, metavar=('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'),
                        help="机器人初始位姿，未提供时使用脚本内默认")
    parser.add_argument("--config", type=str, default=None, help="自定义采集配置文件路径")
    parser.add_argument("--storage", type=str, default=None, help="统一汇总数据文件路径")
    parser.add_argument("--repeat", type=int, default=10, help="每条轨迹重复采集次数（默认1次）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖模式：删除该物体的所有旧运行记录（默认为追加模式）")
    parser.add_argument("--dry-run", action="store_true", help="仅验证配置与轨迹，不执行采集")
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    traj_path = PROJ_DIR / "calibration" / "obj" / "traj.json"
    available_objects = _load_available_objects(traj_path)

    if available_objects and args.object not in available_objects:
        logger.error(f"物体 {args.object} 不在轨迹配置中。可用物体: {available_objects}")
        return

    pose0_default = [556.58, -199.08, 114.10 + 20, 0, 1, 0, 0]

    pose0 = args.pose if args.pose else pose0_default

    collector = ABBDataCollector(
        pose0=pose0,
        object_name=args.object,
        config_file=args.config,
        storage_file=args.storage,
        repeat_count=args.repeat,
        overwrite=args.overwrite
    )

    if args.dry_run:
        logger.info("dry-run 模式：仅检查配置，不执行运动")
        logger.info(f"可用轨迹: {list(collector.trajectory_config.get(args.object, {}).keys())}")
        collector.cleanup()
        return

    try:
        calibration_data = collector.collect_calibration_data()

        logger.info("=" * 60)
        logger.info("标定数据采集完成")
        for obj_name, obj_data in calibration_data.items():
            logger.info(f"物体: {obj_name}, 总记录数: {len(obj_data)}")
            for traj_key, steps in obj_data.items():
                logger.info(f"  {traj_key}: {len(steps)} steps")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"数据采集过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        collector.cleanup()


if __name__ == '__main__':
    main()
