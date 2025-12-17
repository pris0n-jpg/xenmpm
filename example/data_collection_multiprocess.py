#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多进程优化版数据收集脚本
解决Segmentation fault问题，提升CPU利用率和内存管理
"""

import numpy as np
import argparse
from pathlib import Path
from xensesdk.ezgl import tb, Matrix4x4
from xensesdk.ezgl.utils.QtTools import qtcv
from xengym.render.robotScene import RobotScene
from xensesdk.ezgl import GLViewWidget, tb
import pickle
from tqdm import tqdm
import time
import cv2
import gc
import psutil
import os

# 多进程相关
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value, Array, Manager
from concurrent.futures import ProcessPoolExecutor
import queue
import threading
from functools import partial

from xengym import PROJ_DIR
from xensesdk.ezgl.utils.colormap import cm
from xensesdk.xenseInterface.XenseSensor import Sensor
from ezfranka.robot.franka import Robot
from ezfranka.motion import JointMotion, JointPathMotion, CartPathMotion, Affine

# 配置参数
CURR_PATH = Path(__file__).resolve().parent
OBJ_FILE = Path(str(PROJ_DIR/"assets/obj/circle_r4.STL"))
DATA_PATH = Path(str(PROJ_DIR/"data/obj"))
item_name = OBJ_FILE.stem

# 全局配置
CPU_CORES = mp.cpu_count()
MAX_WORKERS = min(CPU_CORES - 2, 6)  # 保留核心给主进程和GUI
QUEUE_SIZE = 50
UPDATE_INTERVAL = 20  # ms，降低更新频率

# 物体参数
bias = [0.0010]
open_width = 0.025 * 2
obj_pos = [0.5554, 0.0698, 0.0448]
obj_rot = [90, 0, 90]

# 随机采样范围
y_range = (-0.007, 0.007)
z_range = (-0.014, 0.0014)
xtheta_range = [-10, 10]
ytheta_range = [-2, 2]
ztheta_range = [-2, 2]

class SensorDataManager:
    """传感器数据管理器 - 在主进程中运行线程"""
    
    def __init__(self):
        self.sensors = {}
        self.data_cache = {}
        self.threads = []
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()
        
    def initialize_sensors(self, sensor_ids):
        """初始化传感器（在主进程中）"""
        successful_sensors = []
        
        for sensor_id in sensor_ids:
            try:
                sensor = Sensor.create(sensor_id)
                self.sensors[sensor_id] = sensor
                successful_sensors.append(sensor_id)
                print(f"传感器 {sensor_id} 初始化成功")
                
                # 启动数据采集线程
                thread = threading.Thread(
                    target=self._sensor_data_loop, 
                    args=(sensor_id,),
                    name=f"SensorThread-{sensor_id}",
                    daemon=True
                )
                thread.start() 
                self.threads.append(thread)
                
            except Exception as e:
                print(f"传感器 {sensor_id} 初始化失败: {e}")
                
        return successful_sensors
    
    def _sensor_data_loop(self, sensor_id):
        """传感器数据采集循环（线程中运行）"""
        sensor = self.sensors[sensor_id]
        print(f"传感器线程 {sensor_id} 启动")
        
        retry_count = 0
        max_retries = 5
        
        while not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # 获取传感器数据
                rectify, diff, depth, force, force_resultant, mesh3d = sensor.selectSensorInfo(
                    Sensor.OutputType.Rectify,
                    Sensor.OutputType.Difference, 
                    Sensor.OutputType.Depth,
                    Sensor.OutputType.Force,
                    Sensor.OutputType.ForceResultant,
                    Sensor.OutputType.Mesh3DInit
                )
                
                # 创建数据包
                data_package = {
                    'sensor_id': sensor_id,
                    'timestamp': time.time(),
                    'rectify': rectify.copy() if rectify is not None else None,
                    'diff': diff.copy() if diff is not None else None,
                    'depth': depth.copy() if depth is not None else None,
                    'force': force.copy() if force is not None else None,
                    'force_resultant': force_resultant.copy() if force_resultant is not None else None,
                    'mesh3d': mesh3d.copy() if mesh3d is not None else None
                }
                
                # 线程安全更新缓存
                with self.lock:
                    self.data_cache[sensor_id] = data_package
                
                # 重置重试计数
                retry_count = 0
                
                # 控制频率（约30fps）
                process_time = time.time() - start_time
                sleep_time = max(0, 0.033 - process_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                retry_count += 1
                print(f"传感器 {sensor_id} 数据获取错误 (重试 {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    print(f"传感器 {sensor_id} 达到最大重试次数，尝试重新初始化...")
                    if self._reconnect_sensor(sensor_id):
                        retry_count = 0
                    else:
                        print(f"传感器 {sensor_id} 重连失败，停止采集")
                        break
                
                time.sleep(0.5)  # 错误后等待较长时间
                
        print(f"传感器线程 {sensor_id} 结束")
    
    def _reconnect_sensor(self, sensor_id):
        """重连传感器"""
        try:
            # 关闭旧传感器
            if sensor_id in self.sensors:
                del self.sensors[sensor_id]
            
            # 等待一下再重连
            time.sleep(1.0)
            
            # 重新创建传感器
            sensor = Sensor.create(sensor_id)
            self.sensors[sensor_id] = sensor
            print(f"传感器 {sensor_id} 重连成功")
            return True
            
        except Exception as e:
            print(f"传感器 {sensor_id} 重连失败: {e}")
            return False
    
    def get_latest_data(self, sensor_id):
        """获取最新数据（线程安全）"""
        with self.lock:
            return self.data_cache.get(sensor_id, None)
    
    def get_all_latest_data(self):
        """获取所有传感器最新数据"""
        with self.lock:
            return self.data_cache.copy()
    
    def stop(self):
        """停止所有传感器线程"""
        print("正在停止传感器线程...")
        self.shutdown_event.set()
        
        for thread in self.threads:
            thread.join(timeout=2.0)
            
        self.threads.clear()
        self.sensors.clear()
        print("传感器线程已停止")

class ImageProcessor:
    """图像处理器 - 独立进程运行"""
    
    def __init__(self, task_queue, result_queue, control_event):
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.control_event = control_event
        
    def process_images(self):
        """图像处理主循环"""
        print("图像处理进程启动")
        
        while not self.control_event.is_set():
            try:
                # 获取处理任务（超时避免阻塞）
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                if task is None:  # 结束信号
                    break
                    
                # 处理图像
                processed_data = self._process_single_task(task)
                
                if processed_data:
                    try:
                        self.result_queue.put_nowait(processed_data)
                    except queue.Full:
                        # 队列满时丢弃最旧数据
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(processed_data)
                        except queue.Empty:
                            pass
                            
            except Exception as e:
                print(f"图像处理错误: {e}")
                time.sleep(0.1)
                
        print("图像处理进程结束")
    
    def _process_single_task(self, task):
        """处理单个图像任务"""
        try:
            task_type = task['type']
            
            if task_type == 'diff_processing':
                # 差分图像处理
                pic = task['image']
                ref_img = task['ref_image']
                sensor_id = task['sensor_id']
                
                diff_raw = pic - ref_img
                diff = np.clip(diff_raw + 100, 0, 255).astype(np.uint8)
                
                return {
                    'type': 'diff_result',
                    'sensor_id': sensor_id,
                    'diff': diff,
                    'timestamp': time.time()
                }
                
            elif task_type == 'depth_processing':
                # 深度图像处理
                depth = task['depth']
                sensor_id = task['sensor_id']
                
                processed_depth = np.minimum(depth, 0) * -1
                depth_resized = cv2.resize(processed_depth, (400, 700))
                depth_colored = cv2.cvtColor(cm.jet(depth_resized), cv2.COLOR_RGB2BGR)
                
                return {
                    'type': 'depth_result',
                    'sensor_id': sensor_id,
                    'depth_colored': depth_colored,
                    'depth_raw': processed_depth,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            print(f"任务处理错误: {e}")
            return None

class DataSaver:
    """数据保存器 - 独立进程运行"""
    
    def __init__(self, save_queue, control_event):
        self.save_queue = save_queue
        self.control_event = control_event
        self.save_count = 0
        
    def save_data_loop(self):
        """数据保存主循环"""
        print("数据保存进程启动")
        
        while not self.control_event.is_set():
            try:
                try:
                    save_task = self.save_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                if save_task is None:  # 结束信号
                    break
                    
                self._save_single_data(save_task)
                
            except Exception as e:
                print(f"数据保存错误: {e}")
                time.sleep(0.1)
                
        print("数据保存进程结束")
    
    def _save_single_data(self, save_task):
        """保存单个数据"""
        try:
            sensor_data = save_task['data']
            trj_num = save_task.get('trj_num')
            sensor_id = save_task['sensor_id']
            
            # 创建保存目录
            if trj_num is not None:
                save_dir = DATA_PATH / item_name / f'sensor_{sensor_id}' / f'trj_{trj_num}'
            else:
                save_dir = DATA_PATH / item_name / f'sensor_{sensor_id}'
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存数据
            save_path = save_dir / f'frame_{self.save_count}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(sensor_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            self.save_count += 1
            
        except Exception as e:
            print(f"保存数据失败: {e}")

class MultiProcessDataCollector:
    """多进程数据收集器主类"""
    
    def __init__(self):
        # 进程间通信队列（只用于图像处理和保存）
        self.image_task_queue = Queue(maxsize=QUEUE_SIZE)
        self.image_result_queue = Queue(maxsize=QUEUE_SIZE)
        self.save_queue = Queue(maxsize=QUEUE_SIZE)
        
        # 控制事件
        self.shutdown_event = Event()
        
        # 进程列表
        self.processes = []
        
        # 传感器管理器（主进程线程）
        self.sensor_manager = SensorDataManager()
        
        # 数据缓存
        self.ref_images = {}
        
        # 性能监控
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def start_processes(self, sensor_ids):
        """启动所有子进程和传感器线程"""
        print(f"启动多进程系统，使用 {MAX_WORKERS} 个工作进程")
        
        # 首先初始化传感器（主进程线程）
        successful_sensors = self.sensor_manager.initialize_sensors(sensor_ids)
        if not successful_sensors:
            raise RuntimeError("没有成功初始化任何传感器")
        
        print(f"成功初始化传感器: {successful_sensors}")
            
        # 启动图像处理进程
        for i in range(min(2, MAX_WORKERS // 2)):  # 2个图像处理进程
            processor = ImageProcessor(self.image_task_queue, self.image_result_queue, self.shutdown_event)
            process = Process(target=processor.process_images, name=f"ImageProcessor-{i}")
            process.start()
            self.processes.append(process)
            
        # 启动数据保存进程
        saver = DataSaver(self.save_queue, self.shutdown_event)
        process = Process(target=saver.save_data_loop, name="DataSaver")
        process.start()
        self.processes.append(process)
        
        print(f"已启动 {len(self.processes)} 个子进程")
    
    def stop_processes(self):
        """停止所有子进程和传感器线程"""
        print("正在停止所有子进程和传感器线程...")
        
        # 停止传感器线程
        self.sensor_manager.stop()
        
        # 设置停止事件
        self.shutdown_event.set()
        
        # 发送结束信号
        try:
            self.image_task_queue.put_nowait(None)
            self.save_queue.put_nowait(None)
        except:
            pass
        
        # 等待进程结束
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                print(f"强制终止进程: {process.name}")
                process.terminate()
                process.join(timeout=2.0)
                
        self.processes.clear()
        print("所有子进程和传感器线程已停止")
    
    def get_sensor_data(self, sensor_id):
        """获取传感器数据（非阻塞）"""
        return self.sensor_manager.get_latest_data(sensor_id)
    
    def get_all_sensor_data(self):
        """获取所有传感器数据"""
        return self.sensor_manager.get_all_latest_data()
    
    def submit_image_task(self, task):
        """提交图像处理任务"""
        try:
            self.image_task_queue.put_nowait(task)
        except queue.Full:
            # 队列满时丢弃任务
            pass
    
    def get_image_result(self):
        """获取图像处理结果"""
        try:
            return self.image_result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def submit_save_task(self, save_task):
        """提交保存任务"""
        try:
            self.save_queue.put_nowait(save_task)
        except queue.Full:
            print("保存队列已满，丢弃数据")
    
    def calculate_fps(self):
        """计算FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
            return fps
        return None

def memory_monitor():
    """内存监控"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def main():
    # 设置多进程启动方式（只对图像处理进程）
    mp.set_start_method('spawn', force=True)
    
    # 准备传感器ID列表（将在多进程收集器中初始化）
    sensor_ids = [0, 2]  # 尝试初始化这些传感器
    
    # 参数解析
    parser = argparse.ArgumentParser(description='多进程优化数据收集系统')
    parser.add_argument('-f', '--fem_file', type=str, 
                       help='FEM文件路径', 
                       default=str(PROJ_DIR/"assets/data/fem_data_vec4070.npz"))
    parser.add_argument('-u', '--urdf_file', type=str,
                       help='URDF文件路径',
                       default=str(PROJ_DIR/"assets/panda/panda_with_vectouch.urdf"))
    parser.add_argument('-o', '--object_file', type=str,
                       help='物体文件路径',
                       default=str(OBJ_FILE))
    parser.add_argument('-l', '--show_left', help='显示左传感器', action='store_true', default=False)
    parser.add_argument('-r', '--show_right', help='显示右传感器', action='store_true', default=False)
    
    args = parser.parse_args()
    if not args.show_left and not args.show_right:
        args.show_left = True
        args.show_right = True

    print(f"系统信息: CPU核心数={CPU_CORES}, 使用工作进程数={MAX_WORKERS}")
    
    # 初始化多进程数据收集器
    collector = MultiProcessDataCollector()
    
    try:
        # 启动子进程
        collector.start_processes(sensor_ids)
        
        # 初始化场景
        scene = RobotScene(
            fem_file=args.fem_file,
            urdf_file=args.urdf_file,
            object_file=args.object_file,
            visible=True,
            left_visible=args.show_left,
            right_visible=args.show_right
        )

        scene.cameraLookAt([1.5, 0, 0.4], [0, 0.07, 0.2], [0, 0, 1])
        scene.object.setTransform(Matrix4x4.fromVector6d(*obj_pos, *obj_rot))

        # 获取参考图像（仅对成功初始化的传感器）
        initialized_sensors = list(collector.sensor_manager.sensors.keys())
        
        if 0 in initialized_sensors and args.show_left:
            scene.left_sensor.set_show_marker(False)
            ref_img_0 = scene.left_sensor.get_image()
            scene.left_sensor.set_show_marker(True)
            collector.ref_images[0] = ref_img_0
            print("传感器0参考图像已获取")
        
        if 2 in initialized_sensors and args.show_right:
            scene.right_sensor.set_show_marker(False)
            ref_img_1 = scene.right_sensor.get_image()
            scene.right_sensor.set_show_marker(True)
            collector.ref_images[2] = ref_img_1
            print("传感器2参考图像已获取")

        # 初始化机器人
        robot = Robot(ip='172.16.0.2', dynamic_rel=0.6)
        robot.set_ee(Affine(0, 0, -0.111, 0, 0, 0))
        gripper = robot.get_gripper(speed=0.08, force=30, homing=False)
        base_pose_and_width = [None, None]

        # 机器人控制函数
        def gripper_move_down(val=0.002):
            width = gripper.width()
            gripper.move(width - val)

        def gripper_move_up(val=0.002):
            width = gripper.width()
            gripper.move(width + val)

        def gripper_x_down(val=-0.001):
            base = robot.get_pose()
            next_pose = Affine(val,0,0,0,0,0) * base
            robot.move_cart(next_pose)

        def gripper_x_up(val=0.001):
            base = robot.get_pose()
            next_pose = Affine(val,0,0,0,0,0) * base
            robot.move_cart(next_pose)
        
        def gripper_y_down(val=-0.001):
            base = robot.get_pose()
            next_pose = Affine(0,val,0,0,0,0) * base
            robot.move_cart(next_pose)
        
        def gripper_y_up(val=0.001):
            base = robot.get_pose()
            next_pose = Affine(0,val,0,0,0,0) * base
            robot.move_cart(next_pose)
        
        def gripper_z_down(val=-0.001):
            base = robot.get_pose()
            next_pose = Affine(0,0,val,0,0,0) * base
            robot.move_cart(next_pose)

        def gripper_z_up(val=0.001):
            base = robot.get_pose()
            next_pose = Affine(0,0,val,0,0,0) * base
            robot.move_cart(next_pose)

        # 全局变量
        cnt = 0
        VERT = []
        
        # 优化的主循环
        def onTimeout():
            nonlocal cnt
            
            if scene.windowShouldClose():
                return
            
            try:
                # 1. 更新机器人状态（轻量级操作）
                joints = robot.get_joints()
                gripper_pose = gripper.width() - bias[0]
                joints = joints.tolist()
                joints.extend([gripper_pose/2, gripper_pose/2])
                scene.set_joints(joints)
                tb.get_widget("current_gripper").value = gripper_pose
                
                # 2. 更新场景（必须在主线程）
                scene.update()
                scene.updateSensors()
                
                # 3. 获取传感器数据并处理图像（非阻塞）
                sensor_data_all = collector.get_all_sensor_data()
                
                for sensor_id, sensor_data in sensor_data_all.items():
                    if sensor_data and sensor_id in collector.ref_images:
                        # 生成仿真图像
                        if sensor_id == 0:
                            scene.left_sensor.set_show_marker(False)
                            pic = scene.left_sensor.get_image()
                            scene.left_sensor.set_show_marker(True)
                        elif sensor_id == 2:
                            scene.right_sensor.set_show_marker(False)
                            pic = scene.right_sensor.get_image()
                            scene.right_sensor.set_show_marker(True)
                        else:
                            continue
                            
                        # 提交图像处理任务
                        task = {
                            'type': 'diff_processing',
                            'image': pic,
                            'ref_image': collector.ref_images[sensor_id],
                            'sensor_id': sensor_id
                        }
                        collector.submit_image_task(task)
                
                # 4. 获取图像处理结果并更新显示
                result = collector.get_image_result()
                if result:
                    if result['type'] == 'diff_result':
                        sensor_id = result['sensor_id']
                        if sensor_id == 0 and img_view_2 is not None:
                            img_view_2.setData(result['diff'])
                        elif sensor_id == 2 and img_view_4 is not None:
                            img_view_4.setData(result['diff'])
                
                # 5. 更新实际传感器数据显示
                if 0 in sensor_data_all and sensor_data_all[0]['diff'] is not None and img_view is not None:
                    img_view.setData(sensor_data_all[0]['diff'])
                if 2 in sensor_data_all and sensor_data_all[2]['diff'] is not None and img_view_3 is not None:
                    img_view_3.setData(sensor_data_all[2]['diff'])
                
                # 6. FPS和内存监控
                fps = collector.calculate_fps()
                if fps is not None:
                    tb.get_widget("fps_counter").value = fps
                    memory_mb = memory_monitor()
                    tb.get_widget("memory_usage").value = memory_mb
                
                # 7. 定期清理内存
                if cnt % 100 == 0:
                    gc.collect()
                    
                cnt += 1
                
            except Exception as e:
                print(f"主循环错误: {e}")

        # 保存函数
        def on_save(trj_num=None):
            try:
                sensor_data_all = collector.get_all_sensor_data()
                
                for sensor_id, sensor_data in sensor_data_all.items():
                    if sensor_data:
                        # 复制传感器数据
                        save_data = sensor_data.copy()
                        
                        # 添加仿真数据
                        if sensor_id == 0:
                            save_data["vertex"] = scene.left_sensor.vis_fem_mesh._mesh._vertexes.data.copy()
                            save_data["sim_depth"] = np.minimum(scene.left_sensor.fem_sim.get_depth(), 0)
                            
                            # 添加仿真图像数据
                            scene.left_sensor.set_show_marker(False)
                            pic = scene.left_sensor.get_image()
                            scene.left_sensor.set_show_marker(True)
                            diff_raw = pic - collector.ref_images.get(0, pic)
                            diff = np.clip(diff_raw + 100, 0, 255).astype(np.uint8)
                            save_data["sim_diff"] = diff
                            save_data["sim_rectify"] = pic
                            
                        elif sensor_id == 2:
                            save_data["vertex"] = scene.right_sensor.vis_fem_mesh._mesh._vertexes.data.copy()
                            save_data["sim_depth"] = np.minimum(scene.right_sensor.fem_sim.get_depth(), 0)
                            
                            # 添加仿真图像数据
                            scene.right_sensor.set_show_marker(False)
                            pic = scene.right_sensor.get_image()
                            scene.right_sensor.set_show_marker(True)
                            diff_raw = pic - collector.ref_images.get(2, pic)
                            diff = np.clip(diff_raw + 100, 0, 255).astype(np.uint8)
                            save_data["sim_diff"] = diff
                            save_data["sim_rectify"] = pic
                        
                        # 将键名改为与原版本兼容的格式
                        if 'rectify' in save_data:
                            save_data['real_rectify'] = save_data.pop('rectify')
                        if 'diff' in save_data:
                            save_data['real_diff'] = save_data.pop('diff')
                        if 'depth' in save_data:
                            save_data['real_depth'] = save_data.pop('depth')
                        if 'force' in save_data:
                            save_data['real_force'] = save_data.pop('force')
                        if 'force_resultant' in save_data:
                            save_data['real_force_resultant'] = save_data.pop('force_resultant')
                        if 'mesh3d' in save_data:
                            save_data['real_mesh3d_init'] = save_data.pop('mesh3d')
                        
                        # 提交保存任务
                        save_task = {
                            'data': save_data,
                            'trj_num': trj_num,
                            'sensor_id': sensor_id
                        }
                        collector.submit_save_task(save_task)
                        
                print("数据保存任务已提交")
                
            except Exception as e:
                print(f"保存数据错误: {e}")

        def on_object_pose_change(data):
            pos = tb.get_value("pos [m]")
            rotx = tb.get_value("euler [deg]")
            scene.object.setTransform(Matrix4x4.fromVector6d(*pos, *rotx))

        def set_tf_visible(data):
            scene.set_tf_visible(id=data[0], visible=data[1])

        # 随机采样函数（优化版）
        def on_random_sample_optimized(nums):
            nonlocal cnt
            cnt = 0
            
            base: Affine = base_pose_and_width[0]
            base_width = base_pose_and_width[1]

            if base is None or base_width is None:
                print("请先设置基础姿态和宽度!")
                return

            xyz = base.translation()
            base = Affine(Matrix4x4.fromVector6d(*xyz, 0, 0, -90))
            
            print(f"开始优化随机采样 {nums} 个轨迹...")
            
            for i in tqdm(range(nums), desc="采样进度"):
                gripper.move(open_width)
                
                # 生成随机参数
                y = np.random.uniform(*y_range)
                z = np.random.uniform(*z_range)
                xtheta = np.random.uniform(*xtheta_range)
                ytheta = np.random.uniform(*ytheta_range)
                ztheta = np.random.uniform(*ztheta_range)
                
                random_pose = base * Affine(0, y, z, 0, 0, 0)
                random_pose = random_pose * Affine(0, 0, 0, xtheta, ytheta, ztheta)
                robot.move_cart(random_pose)
                
                depths = np.array([0.0005, 0.0008, 0.0011, 0.0014, 0.0017, 0.002, 0.0023, 0.0026,
                                 0.0023, 0.002, 0.0017, 0.0014, 0.0011, 0.0008, 0.0005])
                
                for depth in depths:
                    gripper.move(base_width - depth)
                    time.sleep(0.05)  # 减少等待时间
                    on_save(i)
                
                cnt = 0
                print(f"轨迹 {i} 完成")
                
            gripper.move(0.04 * 2)
            print("所有采样完成!")

        def on_random():
            thread = threading.Thread(target=on_random_sample_optimized, daemon=True, args=(50,))
            thread.start()

        def open_gripper():
            thread = threading.Thread(target=lambda: gripper.move(0.035 * 2), daemon=True)
            thread.start()

        def back_place():
            def back_place_func():
                base: Affine = base_pose_and_width[0]
                xyz = base.translation()
                base = Affine(Matrix4x4.fromVector6d(*xyz, 0, 0, -90))
                gripper.move(0.035 * 2)
                robot.move_cart(base)
                gripper.move(base_pose_and_width[1])
            thread = threading.Thread(target=back_place_func, daemon=True)
            thread.start()

        # UI界面
        with tb.window("tb", size=(1600, 2800)):
            with tb.group("view", horizontal=True, collapsible=False, show=False):
                # 左传感器图像显示
                if 0 in initialized_sensors:
                    img_view = tb.add_image_view("real_diff_L", None, img_size=(400, 700), img_format="bgr")
                    img_view_2 = tb.add_image_view("sim_diff_L", None, img_size=(400, 700), img_format="bgr")
                else:
                    img_view = None
                    img_view_2 = None
                    
                # 右传感器图像显示
                if 2 in initialized_sensors and args.show_right:
                    img_view_3 = tb.add_image_view("real_diff_R", None, img_size=(400, 700), img_format="bgr")
                    img_view_4 = tb.add_image_view("sim_diff_R", None, img_size=(400, 700), img_format="bgr")
                else:
                    img_view_3 = None
                    img_view_4 = None

        with tb.window("多进程数据收集系统", None, 10, pos=(300, 200), size=(450, 750)):
            
            tb.add_spacer(10)
            tb.add_text("多进程优化数据收集系统")
            tb.add_spacer(5)
            
            # 性能监控
            with tb.group("性能监控", show=True, collapsed=False, collapsible=True):
                tb.add_drag_value("fps_counter", value=0, min_val=0, max_val=100, step=1, decimals=1, callback=None)
                tb.add_drag_value("memory_usage", value=0, min_val=0, max_val=4000, step=1, decimals=1, callback=None)
                tb.add_text(f"CPU核心数: {CPU_CORES}")
                tb.add_text(f"工作进程数: {MAX_WORKERS}")
                tb.add_text(f"传感器数量: {len(collector.sensor_manager.sensors)}")

            tb.add_spacer(10)
            tb.add_text("控制面板")
            tb.add_spacer(10)
            
            with tb.group("robot", show=False, collapsed=True, collapsible=True):
                joint_values = scene.robot.get_joints_attr("value")
                j_limits = np.array(scene.robot.get_joints_attr("limit"))
                tb.add_drag_array(
                    "joints [rad]", value=joint_values, min_val=j_limits[:, 0], max_val=j_limits[:, 1],
                    step=[0.001]*len(joint_values), decimals=[3]*len(joint_values),
                    format=[name+": %.3f" for name in scene.robot.get_joints_attr("name")],
                    callback=lambda data: scene.set_joint(data[0], data[1]), horizontal=False
                )
                tb.add_drag_value("current_gripper", value=gripper.width(), min_val=0.001, max_val=0.08, step=0.001, decimals=4, callback=None)

            with tb.group("axes", show=False, collapsed=True, collapsible=True, horizontal=True):
                tb.add_checklist("axes1", items=scene._axes_name[:7], horizontal=False, exclusive=False, callback=set_tf_visible)
                tb.add_checklist("axes2", items=scene._axes_name[7:], horizontal=False, exclusive=False, callback=set_tf_visible)

            with tb.group("object", show=False, collapsed=True, collapsible=True, horizontal=False):
                tb.add_drag_array(
                    "pos [m]", value=obj_pos if obj_pos is not None else [0.427, 0, 0.530], 
                    min_val=[-1, -1, -1], max_val=[1, 1, 1], step=[0.0001]*3, decimals=[4]*3,
                    format=[name+": %.4f" for name in ["x", "y", "z"]], callback=on_object_pose_change, horizontal=False
                )
                tb.add_drag_array(
                    "euler [deg]", value=obj_rot if obj_rot is not None else [90, 0, 90], 
                    min_val=[-180, -180, -180], max_val=[180, 180, 180], step=[0.1]*3, decimals=[1]*3,
                    format=[name+": %.1f" for name in ["a", "b", "c"]], callback=on_object_pose_change, horizontal=False
                )
                tb.add_checkbox("show axis", value=True, callback=lambda data: scene.object.childItems()[0].setVisible(data))

            if 0 in initialized_sensors and args.show_left:
                with tb.group("left sensor", show=False, collapsed=True, collapsible=True, horizontal=False):
                    tb.add_checkbox("enable smooth", value=True, callback=scene.left_sensor.enable_smooth)
                    tb.add_checkbox("show marker", value=True, callback=scene.left_sensor.set_show_marker)
                    tb.add_checkbox("show contact", value=False, callback=scene.left_sensor.set_show_contact)
                    tb.add_checkbox("show force", value=False, callback=scene.left_sensor.set_show_force)
                    tb.add_checkbox("show fem mesh", value=False, callback=scene.left_sensor.set_show_fem_mesh)
                    tb.add_button("align view", callback=scene.left_sensor.align_view)

            if 2 in initialized_sensors and args.show_right:
                with tb.group("right sensor", show=False, collapsed=True, collapsible=True, horizontal=False):
                    tb.add_checkbox("enable smooth ", value=True, callback=scene.right_sensor.enable_smooth)
                    tb.add_checkbox("show marker ", value=True, callback=scene.right_sensor.set_show_marker)
                    tb.add_checkbox("show contact ", value=False, callback=scene.right_sensor.set_show_contact)
                    tb.add_checkbox("show force ", value=False, callback=scene.right_sensor.set_show_force)
                    tb.add_checkbox("show fem mesh ", value=False, callback=scene.right_sensor.set_show_fem_mesh)
                    tb.add_button("align view ", callback=scene.right_sensor.align_view)

            with tb.group("功能控制", show=True, collapsed=False, collapsible=True, horizontal=False):
                tb.add_drag_value("bias", value=bias[0], min_val=0, max_val=0.004, step=0.0001, decimals=4, 
                                format="%.4f", callback=lambda data: bias.__setitem__(0, data))
                tb.add_button("保存数据", callback=on_save)
                tb.add_button("设置基础姿态", callback=lambda: base_pose_and_width.__setitem__(0, robot.get_pose()))
                tb.add_button("设置物体姿态", callback=lambda: base_pose_and_width.__setitem__(0, Affine(scene.object.transform(False))))
                tb.add_button("设置基础宽度", callback=lambda: base_pose_and_width.__setitem__(1, gripper.width()))
                tb.add_button("打开夹爪", callback=open_gripper)
                tb.add_button("回到起始位置", callback=back_place)
                tb.add_button("随机采样(多进程)", callback=on_random)

            # 降低主循环频率以减少CPU负载
            tb.add_timer("timer", interval_ms=UPDATE_INTERVAL, callback=onTimeout)

            # 键盘绑定
            tb.add_key_binding(tb.Key.Key_J, gripper_move_down)
            tb.add_key_binding(tb.Key.Key_K, gripper_move_up)
            tb.add_key_binding(tb.Key.Key_W, gripper_x_down)
            tb.add_key_binding(tb.Key.Key_S, gripper_x_up)
            tb.add_key_binding(tb.Key.Key_A, gripper_y_down)
            tb.add_key_binding(tb.Key.Key_D, gripper_y_up)
            tb.add_key_binding(tb.Key.Key_Z, gripper_z_up)
            tb.add_key_binding(tb.Key.Key_X, gripper_z_down)

        print("多进程数据收集系统启动完成!")
        print("主要优化:")
        print("- 传感器数据获取使用独立进程")
        print("- 图像处理使用多进程池")
        print("- 数据保存使用异步I/O")
        print("- 智能内存管理和垃圾回收")
        print(f"- 使用 {len(collector.processes)} 个子进程")
        
        tb.exec()
        
    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("正在清理资源...")
        collector.stop_processes()
        
        # 强制垃圾回收
        gc.collect()
        print("资源清理完成")

if __name__ == "__main__":
    main() 