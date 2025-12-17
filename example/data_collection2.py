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
import queue
import threading

from xengym import PROJ_DIR
import cv2
from xensesdk.ezgl.utils.colormap import cm
CURR_PATH = Path(__file__).resolve().parent

VERT = []
FLAG = 0
from xensesdk.xenseInterface.XenseSensor import Sensor

from ezfranka.robot.franka import Robot
from ezfranka.motion import JointMotion, JointPathMotion, CartPathMotion, Affine
from threading import Thread

from MarkerInterp import MarkerInterpolator

OBJ_FILE = Path(str(PROJ_DIR/"assets/obj/circle_r4.STL"))
DATA_PATH = Path(str(PROJ_DIR/"data/obj"))
item_name = OBJ_FILE.stem

bias = [0.0010]
open_width = 0.025 * 2
cnt = 0
obj_pos = [0.5554, 0.0698, 0.0444]
obj_rot = [90, 0, 90]

# 添加全局队列用于线程间通信
render_queue = queue.Queue()
render_result_queue = queue.Queue()

y_range = (-0.007, 0.007)
z_range = (-0.014, 0.0014)
xtheta_range = [-10, 10]
ytheta_range = [-2,2]
ztheta_range = [-2,2]

def main():
    sensor_0 = Sensor.create(0)
    marker_init = sensor_0.selectSensorInfo(Sensor.OutputType.Marker2DInit)
    marker_geter = MarkerInterpolator(marker_init)
    try:
        sensor_1 = Sensor.create(2)
    except Exception as e:
        print('only 1 sensor')
        sensor_1 = None
    
    parser = argparse.ArgumentParser(description='Sim')
    # 添加参数
    # parser.add_argument('-f', '--fem_file', type=str, help='Path to the FEM file (default: %(default)s)', default=str(PROJ_DIR/"assets/data/fem_data_vec4070.npz"))
    parser.add_argument('-f', '--fem_file', type=str, help='Path to the FEM file (default: %(default)s)', default=str(PROJ_DIR/"assets/data/fem_data_gel_2035.npz"))
    parser.add_argument('-u', '--urdf_file', type=str, help='Path to the URDF file (default: %(default)s)', default=str(PROJ_DIR/"assets/panda/panda_with_vectouch.urdf"))
    parser.add_argument('-o', '--object_file', type=str, help='Path to the object file (default: %(default)s)', default=str(OBJ_FILE))
    parser.add_argument('-l', '--show_left', help='Show left sensor (default: %(default)s)', action='store_true', default=False)
    parser.add_argument('-r', '--show_right', help='Show right sensor (default: %(default)s)', action='store_true', default=False)

    args = parser.parse_args()
    if not args.show_left and not args.show_right:
        args.show_left = True
        args.show_right = True

    #  ==== xensim ====
    scene = RobotScene(
        fem_file=args.fem_file,
        urdf_file=args.urdf_file,
        object_file=args.object_file,
        visible=True,
        left_visible=args.show_left,
        right_visible=args.show_right
    )
    scene.left_sensor.set_show_marker(False)
    ref_img_0 = scene.left_sensor.get_image()
    scene.left_sensor.set_show_marker(True)
    scene.right_sensor.set_show_marker(False)
    ref_img_1 = scene.right_sensor.get_image()
    scene.right_sensor.set_show_marker(True)

    scene.cameraLookAt([1.5, 0, 0.4], [0, 0.07, 0.2], [0, 0, 1])
    
    scene.object.setTransform(Matrix4x4.fromVector6d(*obj_pos, *obj_rot))

    # 在 main() 或合适位置初始化
    scene.original_scale_factor = 1.0  # 假设初始为1

    #  ==== robot ====
    robot = Robot(ip='172.16.0.2', dynamic_rel=0.3)  # 降低动态参数
    robot.set_ee(Affine(0, 0, -0.111, 0, 0, 0))  #  设置末端执行器位置
    gripper = robot.get_gripper(speed=0.05, force=20, homing=False)  # 降低速度和力度
    base_pose_and_width = [None, None]
    
    # 添加机器人状态检查
    robot_error_count = 0
    max_robot_errors = 5

    def safe_robot_operation(operation, *args, **kwargs):
        """安全执行机器人操作，带重试机制"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                print(f"机器人操作失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5)  # 等待一下再重试
    
    def gripper_move_down(val=0.002):
        try:
            width = safe_robot_operation(gripper.width)
            safe_robot_operation(gripper.move, width - val)
        except Exception as e:
            print(f"夹爪下移失败: {e}")

    def gripper_move_up(val=0.002):
        try:
            width = safe_robot_operation(gripper.width)
            safe_robot_operation(gripper.move, width + val)
        except Exception as e:
            print(f"夹爪上移失败: {e}")

    def gripper_x_down(val=-0.001):
        try:
            base = safe_robot_operation(robot.get_pose)
            next_pose = Affine(val,0,0,0,0,0) * base
            safe_robot_operation(robot.move_cart, next_pose)
        except Exception as e:
            print(f"X轴下移失败: {e}")

    def gripper_x_up(val=0.001):
        try:
            base = safe_robot_operation(robot.get_pose)
            next_pose = Affine(val,0,0,0,0,0) * base
            safe_robot_operation(robot.move_cart, next_pose)
        except Exception as e:
            print(f"X轴上移失败: {e}")
    
    def gripper_y_down(val=-0.001):
        try:
            base = safe_robot_operation(robot.get_pose)
            next_pose = Affine(0,val,0,0,0,0) * base
            safe_robot_operation(robot.move_cart, next_pose)
        except Exception as e:
            print(f"Y轴下移失败: {e}")
    
    def gripper_y_up(val=0.001):
        try:
            base = safe_robot_operation(robot.get_pose)
            next_pose = Affine(0,val,0,0,0,0) * base
            safe_robot_operation(robot.move_cart, next_pose)
        except Exception as e:
            print(f"Y轴上移失败: {e}")
    
    def gripper_z_down(val=-0.001):
        try:
            base = safe_robot_operation(robot.get_pose)
            next_pose = Affine(0,0,val,0,0,0) * base
            safe_robot_operation(robot.move_cart, next_pose)
        except Exception as e:
            print(f"Z轴下移失败: {e}")

    def gripper_z_up(val=0.001):
        try:
            base = safe_robot_operation(robot.get_pose)
            next_pose = Affine(0,0,val,0,0,0) * base
            safe_robot_operation(robot.move_cart, next_pose)
        except Exception as e:
            print(f"Z轴上移失败: {e}")

    # object
    def on_change_scale_change(val):
        scale = tb.get_value("scale")
        Scale = np.array([scale]*3)
        Original_Scale = np.array([scene.scale_factor]*3)
        scene.object.scale(*(Scale/Original_Scale))
        scene.scale_factor = scale

    def scale_change(val):
        if val:  # True，缩小
            scale = 0.001
        else:    # False，恢复
            scale = 1.0

        # 计算缩放比例
        Scale = np.array([scale]*3)
        Original_Scale = np.array([scene.scale_factor]*3)
        scene.object.scale(*(Scale/Original_Scale))
        
        # 反向缩放axis
        axis = scene.object.childItems()[0]  # 假设axis是第一个child
        axis_scale = scene.scale_factor / scale
        axis.scale(axis_scale, axis_scale, axis_scale)

        scene.scale_factor = scale
        
    # 修改处理渲染队列的函数
    def process_render_queue():
        """在主线程中处理渲染任务"""
        try:
            while not render_queue.empty():
                task = render_queue.get_nowait()
                task_type = task['type']
                
                if task_type == 'save':
                    trj_num = task['trj_num']
                    # 在主线程中执行渲染操作
                    _on_save_render(trj_num)
                    render_result_queue.put({'status': 'completed', 'trj_num': trj_num})
                    
        except queue.Empty:
            pass
        except Exception as e:
            print(f"渲染队列处理错误: {e}")
            render_result_queue.put({'status': 'error', 'error': str(e)})

    def _on_save_render(trj_num=None):
        """在主线程中执行的渲染和保存操作"""
        global cnt
        scene.update()
        scene.updateSensors()
        
        # sensor_0
        data_0 = {}
        # real
        rectify_real, diff_real, depth_real, marker_of_real, force_real, res_force_real = sensor_0.selectSensorInfo(
            Sensor.OutputType.Rectify, 
            Sensor.OutputType.Difference,
            Sensor.OutputType.Depth,
            Sensor.OutputType.Marker2D,
            Sensor.OutputType.Force,
            Sensor.OutputType.ForceResultant
        )
        marker_real = marker_geter.target_grid + marker_geter.interpolate(marker_of_real)
        data_0["rectify_real"] = rectify_real
        data_0["diff_real"] = diff_real
        data_0["depth_real"] = depth_real
        data_0["marker_of_real"] = marker_of_real
        data_0["force_real"] = force_real
        data_0["res_force_real"] = res_force_real
        data_0["marker_real"] = marker_real
        
        # sim
        raw_pic = scene.left_sensor.get_image()
        scene.left_sensor.set_show_marker(False)
        pic = scene.left_sensor.get_image()
        scene.left_sensor.set_show_marker(True)
        diff_raw = pic - ref_img_0
        diff = np.clip(diff_raw +100, 0, 255).astype(np.uint8)
        raw_depth = scene.left_sensor.fem_sim.get_depth()
        depth = np.minimum(raw_depth, 0)*-1
        depth = cv2.resize(depth, (400,700))
        depth = cv2.cvtColor(cm.jet(depth), cv2.COLOR_RGB2BGR)
        data_0["rectify_sim"] = raw_pic
        data_0["rectify_pure_sim"] = pic
        data_0["diff_sim"] = diff
        data_0["marker_sim"] = scene.left_sensor.get_marker()
        data_0["raw_depth_sim"] = cv2.resize(raw_depth, (400,700))
        data_0["vis_depth_sim"] = depth
        data_0["force_sim"] = scene.left_sensor.get_force().reshape(35,20,3)
        data_0["vertex"] = scene.left_sensor.vis_fem_mesh._mesh._vertexes.data.copy()

        if trj_num is not None:
            save_dir = DATA_PATH / item_name / 'sensor_0' / f'trj_{trj_num}'
        else:
            save_dir = DATA_PATH / item_name / 'sensor_0'
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'frame_{cnt}.pkl', 'wb') as f:
                    pickle.dump(data_0, f)

        # sensor_1
        if sensor_1 is not None:
            data_1 = {}
            # real
            rectify_real, diff_real, depth_real, marker_of_real, force_real, res_force_real = sensor_1.selectSensorInfo(
                Sensor.OutputType.Rectify, 
                Sensor.OutputType.Difference,
                Sensor.OutputType.Depth,
                Sensor.OutputType.Marker2D,
                Sensor.OutputType.Force,
                Sensor.OutputType.ForceResultant
            )
            marker_real = marker_geter.target_grid + marker_geter.interpolate(marker_of_real)
            data_1["rectify_real"] = rectify_real
            data_1["diff_real"] = diff_real
            data_1["depth_real"] = depth_real
            data_1["marker_of_real"] = marker_of_real
            data_1["force_real"] = force_real
            data_1["res_force_real"] = res_force_real
            data_1["marker_real"] = marker_real
            
            # sim
            raw_pic = scene.right_sensor.get_image()
            scene.right_sensor.set_show_marker(False)
            pic = scene.right_sensor.get_image()
            scene.right_sensor.set_show_marker(True)
            diff_raw = pic - ref_img_0
            diff = np.clip(diff_raw +100, 0, 255).astype(np.uint8)
            raw_depth = scene.right_sensor.fem_sim.get_depth()
            depth = np.minimum(raw_depth, 0)*-1
            depth = cv2.resize(depth, (400,700))
            depth = cv2.cvtColor(cm.jet(depth), cv2.COLOR_RGB2BGR)
            data_1["rectify_sim"] = raw_pic
            data_1["rectify_pure_sim"] = pic
            data_1["diff_sim"] = diff
            data_1["marker_sim"] = scene.right_sensor.get_marker()
            data_1["raw_depth_sim"] = cv2.resize(raw_depth, (400,700))
            data_1["vis_depth_sim"] = depth
            data_1["force_sim"] = scene.right_sensor.get_force().reshape(35,20,3)
            data_1["vertex"] = scene.right_sensor.vis_fem_mesh._mesh._vertexes.data.copy()

            if trj_num is not None:
                save_dir = DATA_PATH / item_name / 'sensor_1' / f'trj_{trj_num}'
            else:
                save_dir = DATA_PATH / item_name / 'sensor_1'
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / f'frame_{cnt}.pkl', 'wb') as f:
                pickle.dump(data_1, f)

        cnt += 1
        
    # callbacks
    def onTimeout():
        nonlocal robot_error_count
        
        if scene.windowShouldClose() == False:
            # 处理渲染队列
            process_render_queue()
            
            # update robot with error handling
            try:
                joints = robot.get_joints()
                gripper_pose = gripper.width() - bias[0]
                joints = joints.tolist()
                joints.extend([gripper_pose/2, gripper_pose/2])
                scene.set_joints(joints)
                tb.get_widget("current_gripper").value = gripper_pose
                robot_error_count = 0  # 重置错误计数
            except (RuntimeError, Exception) as e:
                robot_error_count += 1
                print(f"机器人通信错误 ({robot_error_count}/{max_robot_errors}): {e}")
                
                if robot_error_count >= max_robot_errors:
                    print("机器人通信错误次数过多，程序退出")
                    return
                
                # 使用上次的关节状态，避免机器人状态更新失败
                try:
                    current_joints = scene.robot.get_joints_attr("value")
                    scene.set_joints(current_joints)
                except Exception:
                    pass
            
            scene.update()
            scene.updateSensors()
            
            # sensor 0
            try:
                diff_real_0 = sensor_0.selectSensorInfo(Sensor.OutputType.Difference)
                scene.left_sensor.set_show_marker(False)
                pic = scene.left_sensor.get_image()
                scene.left_sensor.set_show_marker(True)
                diff_raw = pic - ref_img_0
                diff = np.clip(diff_raw +100, 0, 255).astype(np.uint8)
                img_view.setData(diff_real_0)
                img_view_2.setData(diff)
            except Exception as e:
                print(f"传感器0数据获取错误: {e}")

            # sensor 1
            if sensor_1 is not None:
                try:
                    diff_real_1 = sensor_1.selectSensorInfo(Sensor.OutputType.Difference)
                    scene.right_sensor.set_show_marker(False)
                    pic = scene.right_sensor.get_image()
                    scene.right_sensor.set_show_marker(True)
                    diff_raw = pic - ref_img_1
                    diff = np.clip(diff_raw +100, 0, 255).astype(np.uint8)
                    img_view_3.setData(diff_real_1)
                    img_view_4.setData(diff)
                except Exception as e:
                    print(f"传感器1数据获取错误: {e}")



    def save_vertex():
        global FLAG
        VERT.append(
            (
                scene.left_sensor.vis_fem_mesh._mesh._vertexes.data.copy(),
                np.minimum(scene.left_sensor.fem_sim.get_depth(), 0)
            )
        )
        FLAG = 0  
        
    def replay_vertex():
        global FLAG
        FLAG = 1
        scene.left_sensor.update_replay(*VERT[-1])
        scene.left_sensor.super_update()

    def on_save(trj_num=None):
        """修改为通过队列请求渲染"""
        render_queue.put({'type': 'save', 'trj_num': trj_num})
        
        # 等待渲染完成
        try:
            result = render_result_queue.get(timeout=5.0)  # 5秒超时
            if result['status'] == 'error':
                print(f"渲染错误: {result['error']}")
        except queue.Empty:
            print("渲染超时")

    def on_object_pose_change(data):
        pos = tb.get_value("pos [m]")
        rotx = tb.get_value("euler [deg]")
        scene.object.setTransform(Matrix4x4.fromVector6d(*pos, *rotx))

    def set_tf_visible(data):
        scene.set_tf_visible(id=data[0], visible=data[1])
    
    # === show picture ===
    with tb.window("tb", size=(1600, 2800)):
        with tb.group("view", horizontal=True, collapsible=False, show=False):
            img_view = tb.add_image_view("real_diff_L" , None, img_size=(400, 700), img_format= "bgr")
            img_view_2 = tb.add_image_view("sim_diff_L" , None, img_size=(400, 700), img_format= "rgb")
            if sensor_1 is not None:
                img_view_3 = tb.add_image_view("real_diff_R" , None, img_size=(400, 700), img_format= "bgr")
                img_view_4 = tb.add_image_view("sim_diff_R" , None, img_size=(400, 700), img_format= "rgb")


    # functions
    def open_gripper():
        thread = Thread(target=open_gripper_func, daemon=True)
        thread.start()

    def open_gripper_func():
        try:
            safe_robot_operation(gripper.move, 0.035 * 2)
        except Exception as e:
            print(f"打开夹爪失败: {e}")

    def back_place():
        thread = Thread(target=back_place_func, daemon=True)
        thread.start()

    def back_place_func():
        try:
            base: Affine = base_pose_and_width[0]
            if base is None:
                print("基础位置未设置!")
                return
                
            xyz = base.translation()
            base = Affine(Matrix4x4.fromVector6d(*xyz, 0, 0, -90))
            
            safe_robot_operation(gripper.move, 0.035 * 2)
            time.sleep(0.3)
            safe_robot_operation(robot.move_cart, base)
            time.sleep(0.3)
            
            if base_pose_and_width[1] is not None:
                safe_robot_operation(gripper.move, base_pose_and_width[1])
        except Exception as e:
            print(f"回到初始位置失败: {e}")

    def test_pw():
        thread = Thread(target=on_test_pw, daemon=True, args=(1,))
        thread.start()

    def on_test_pw(nums):
        
        print("开始测试位置和宽度...")
        
        base: Affine = base_pose_and_width[0]
        base_width = base_pose_and_width[1]

        if base is None or base_width is None:
            print("请先设置基础位置和夹爪宽度!")
            return

        xyz = base.translation()
        base = Affine(Matrix4x4.fromVector6d(*xyz, 0, 0, -90))
        
        success_count = 0
        
        for i in tqdm(range(nums)):
            try:
                safe_robot_operation(gripper.move, open_width)
                time.sleep(0.2)
                
                # 减少随机范围
                y = np.random.uniform(*y_range) * 0.8
                z = np.random.uniform(*z_range) * 0.8
                xtheta = np.random.uniform(*xtheta_range) * 0.6

                random_pose = base * Affine(0, y, z, 0, 0, 0)
                random_pose = random_pose * Affine(0, 0, 0, xtheta, 0, 0)
                safe_robot_operation(robot.move_cart, random_pose, relative=False)
                time.sleep(0.3)
                
                depths = np.array([0.0017, 0.002, 0.0023, 0.0026])
                for depth in depths:
                    try:
                        target_width = base_width - depth
                        if target_width > 0.001:
                            safe_robot_operation(gripper.move, target_width)
                            time.sleep(0.15)
                    except Exception as e:
                        print(f"深度 {depth} 测试失败: {e}")
                        
                success_count += 1
                print(f"测试轨迹 {i} 完成 (成功: {success_count}/{i+1})")
                
            except Exception as e:
                print(f"测试轨迹 {i} 失败: {e}")
                try:
                    safe_robot_operation(gripper.move, open_width)
                    time.sleep(0.3)
                except Exception:
                    print("无法恢复，请检查机器人状态")
                    break
                    
        try:
            safe_robot_operation(gripper.move, open_width)
            print(f"位置宽度测试完成! 成功: {success_count}/{nums}")
        except Exception as e:
            print(f"最终恢复失败: {e}")

    def on_random_press():
        thread = Thread(target=on_random_press_sample, daemon=True, args=(100,))
        thread.start()

    def on_random_press_sample(nums):
        global cnt
        cnt = 0
        
        print("开始随机采样...")

        base: Affine = base_pose_and_width[0]
        base_width = base_pose_and_width[1]

        if base is None or base_width is None:
            print("请先设置基础位置和夹爪宽度!")
            return

        xyz = base.translation()
        base = Affine(Matrix4x4.fromVector6d(*xyz, 0, 0, -90))
        
        # 减少深度变化的频率，避免过于频繁的运动
        depths = np.array([0.0008, 0.0014, 0.002, 0.0026, 0.002, 0.0014, 0.0008])
        
        success_count = 0
        
        for i in tqdm(range(nums)):
            try:
                # 安全地打开夹爪
                safe_robot_operation(gripper.move, open_width)
                time.sleep(0.2)  # 增加等待时间
                
                # 生成随机位置，但限制范围更小
                y = np.random.uniform(*y_range) * 0.7  # 减少Y轴范围
                z = np.random.uniform(*z_range) * 0.7  # 减少Z轴范围
                xtheta = np.random.uniform(*xtheta_range) * 0.5  # 减少旋转角度
                ytheta = np.random.uniform(*ytheta_range) * 0.5
                ztheta = np.random.uniform(*ztheta_range) * 0.5
                
                # 逐步移动到目标位置
                random_pose = base * Affine(0, y, z, 0, 0, 0)
                random_pose = random_pose * Affine(0, 0, 0, xtheta, ytheta, ztheta)
                
                # 安全移动到目标位置
                safe_robot_operation(robot.move_cart, random_pose)
                time.sleep(0.3)  # 等待位置稳定
                
                # 执行深度序列
                for j, depth in enumerate(depths):
                    try:
                        target_width = base_width - depth
                        if target_width > 0.001:  # 确保夹爪宽度合理
                            safe_robot_operation(gripper.move, target_width)
                            time.sleep(0.2)  # 增加等待时间让传感器稳定
                            
                            # 请求渲染
                            render_queue.put({'type': 'save', 'trj_num': i})
                            
                    except Exception as e:
                        print(f"深度 {depth} 执行失败: {e}")
                        continue
                
                success_count += 1
                cnt = 0
                print(f"轨迹 {i} 完成 (成功: {success_count}/{i+1})")
                
            except Exception as e:
                print(f"轨迹 {i} 失败: {e}")
                # 尝试恢复到安全位置
                try:
                    safe_robot_operation(gripper.move, open_width)
                    time.sleep(0.5)
                except Exception:
                    print("无法恢复到安全位置，请手动检查机器人状态")
                    break
        
        # 最后安全地打开夹爪
        try:
            safe_robot_operation(gripper.move, 0.04 * 2)
            print(f"随机采样完成! 成功完成 {success_count}/{nums} 个轨迹")
        except Exception as e:
            print(f"最终夹爪打开失败: {e}")

    # ui
    with tb.window("Xensim", None, 10, pos=(300, 200), size=(400, 600)):

        tb.add_spacer(10)
        tb.add_text(" Settings")
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
            tb.get_widget("axes1").setStyleSheet("")
            tb.get_widget("axes2").setStyleSheet("")

        with tb.group("object", show=False, collapsed=True, collapsible=True, horizontal=False):
            tb.add_drag_array(
                "pos [m]", value=obj_pos if obj_pos is not None else [0.427, 0, 0.530], min_val=[-1, -1, -1], max_val=[1, 1, 1], step=[0.0001]*3, decimals=[4]*3,
                format=[name+": %.4f" for name in ["x", "y", "z"]], callback=on_object_pose_change, horizontal=False
            )
            tb.add_drag_array(
                "euler [deg]", value=obj_rot if obj_rot is not None else [90, 0, 90], min_val=[-180, -180, -180], max_val=[180, 180, 180], step=[0.1]*3, decimals=[1]*3,
                format=[name+": %.1f" for name in ["a", "b", "c"]], callback=on_object_pose_change, horizontal=False
            )
            # tb.add_drag_value("scale", value=1, min_val=0.001, max_val=1, step=0.001, callback=on_change_scale_change)
            # tb.add_checkbox("unit: mm", value=False, callback=scale_change)
            tb.add_checkbox("show axis", value=True, callback=lambda data: scene.object.childItems()[0].setVisible(data))

        if args.show_left:
            with tb.group("left sensor", show=False, collapsed=True, collapsible=True, horizontal=False):
                tb.add_checkbox("enable smooth", value=True, callback=scene.left_sensor.enable_smooth)
                tb.add_checkbox("show marker", value=True, callback=scene.left_sensor.set_show_marker)
                tb.add_checkbox("show contact", value=False, callback=scene.left_sensor.set_show_contact)
                tb.add_checkbox("show force", value=False, callback=scene.left_sensor.set_show_force)
                tb.add_checkbox("show fem mesh", value=False, callback=scene.left_sensor.set_show_fem_mesh)
                tb.add_button("align view", callback=scene.left_sensor.align_view)

        if args.show_right:
            with tb.group("right sensor ", show=False, collapsed=True, collapsible=True, horizontal=False):
                tb.add_checkbox("enable smooth ", value=True, callback=scene.right_sensor.enable_smooth)
                tb.add_checkbox("show marker ", value=True, callback=scene.right_sensor.set_show_marker)
                tb.add_checkbox("show contact ", value=False, callback=scene.right_sensor.set_show_contact)
                tb.add_checkbox("show force ", value=False, callback=scene.right_sensor.set_show_force)
                tb.add_checkbox("show fem mesh", value=False, callback=scene.right_sensor.set_show_fem_mesh)
                tb.add_button("align view ", callback=scene.right_sensor.align_view)

        with tb.group("Function", show=False, collapsed=True, collapsible=True, horizontal=False):
            tb.add_drag_value("bias", value=bias[0], min_val=0, max_val=0.004, step=0.0001, decimals=4, format="%.4f", callback=lambda data: bias.__setitem__(0, data))
            tb.add_button("save", callback=on_save)
            def set_base_pose():
                try:
                    pose = safe_robot_operation(robot.get_pose)
                    base_pose_and_width[0] = pose
                    print("基础位置已设置")
                except Exception as e:
                    print(f"设置基础位置失败: {e}")
                    
            def set_base_pose_obj():
                try:
                    pose = Affine(scene.object.transform(False))
                    base_pose_and_width[0] = pose
                    print("基础位置已设置为物体位置")
                except Exception as e:
                    print(f"设置物体基础位置失败: {e}")
                    
            def set_base_width():
                try:
                    width = safe_robot_operation(gripper.width)
                    base_pose_and_width[1] = width
                    print(f"基础夹爪宽度已设置: {width:.4f}")
                except Exception as e:
                    print(f"设置基础夹爪宽度失败: {e}")
            
            tb.add_button("base_pose", callback=set_base_pose)
            tb.add_button("base_pose_obj", callback=set_base_pose_obj)
            tb.add_button("base_width", callback=set_base_width)
            tb.add_button("open gripper", callback=open_gripper)
            tb.add_button("back_place", callback = back_place)
            tb.add_button("test_pose_width", callback=test_pw)
            tb.add_button("random press sample", callback=on_random_press)

        tb.add_timer("timer", interval_ms=2, callback=onTimeout)
        # tb.add_key_binding(tb.Key.Key_0, save_vertex)
        # tb.add_key_binding(tb.Key.Key_1, replay_vertex)

        tb.add_key_binding(tb.Key.Key_J, gripper_move_down)
        tb.add_key_binding(tb.Key.Key_K, gripper_move_up)
        tb.add_key_binding(tb.Key.Key_W, gripper_x_down)
        tb.add_key_binding(tb.Key.Key_S, gripper_x_up)
        tb.add_key_binding(tb.Key.Key_A, gripper_y_down)
        tb.add_key_binding(tb.Key.Key_D, gripper_y_up)
        tb.add_key_binding(tb.Key.Key_Z, gripper_z_up)
        tb.add_key_binding(tb.Key.Key_X, gripper_z_down)



    tb.exec()

if __name__ == "__main__":
    main()