import numpy as np
import argparse
from pathlib import Path
from xensesdk.ezgl import tb, Matrix4x4
from xensesdk.ezgl.utils.QtTools import qtcv
from xengym.render.robotScene import RobotScene
from xensesdk.ezgl import GLViewWidget, tb
import pickle
from MarkerInterp import MarkerInterpolator

from xengym import PROJ_DIR
import cv2
from xensesdk.ezgl.utils.colormap import cm
CURR_PATH = Path(__file__).resolve().parent

VERT = []
FLAG = 0
flag = 1
from xensesdk.xenseInterface.XenseSensor import Sensor


OBJ_FILE = Path(str(PROJ_DIR/"assets/obj/circle_r4.STL"))
DATA_PATH = Path(str(PROJ_DIR/"data/obj"))
item_name = OBJ_FILE.stem

bias = [0.0027]
open_width = 0.025
cnt = -1


def main():
    sensor_0 = Sensor.create(0)
    marker_init = sensor_0.selectSensorInfo(Sensor.OutputType.Marker2DInit)
    marker_geter = MarkerInterpolator(marker_init)
    try:
        sensor_1 = Sensor.create(2)
    except Exception as e:
        print(e)
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
    
    # 如果sensor_1不可用，强制禁用右侧传感器显示
    if sensor_1 is None:
        args.show_right = False

    #  ==== xensim ====
    scene = RobotScene(
        fem_file=args.fem_file,
        urdf_file=args.urdf_file,
        object_file=args.object_file,
        visible=True,
        left_visible=args.show_left,
        right_visible=args.show_right
    )

    scene.cameraLookAt([1.5, 0, 0.8], [0, 0, 0.4], [0, 0, 1])
    obj_pos = [0.427, 0, 0.515]
    obj_rot = [90, 0, 90]
    scene.object.setTransform(Matrix4x4.fromVector6d(*obj_pos, *obj_rot))

    # get ref img 
    scene.left_sensor.set_show_marker(False)
    ref_img = scene.left_sensor.get_image()
    scene.left_sensor.set_show_marker(True)

    # 在 main() 或合适位置初始化
    scene.original_scale_factor = 1.0  # 假设初始为1


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
        
    # callbacks
    def onTimeout():
        
        if scene.windowShouldClose() == False:

            reference_real_0 = sensor_0._dag_runner.reference_image
            # if sensor_1 is not None:
            #     reference_real_1 = sensor_1._dag_runner.reference_image
            rectify_real_0, diff_real_0, depth_real_0, marker_of_real_0, force_real_0, res_force_real_0 = sensor_0.selectSensorInfo(
                                                                        Sensor.OutputType.Rectify, 
                                                                        Sensor.OutputType.Difference, 
                                                                        Sensor.OutputType.Depth,
                                                                        Sensor.OutputType.Marker2D,
                                                                        Sensor.OutputType.Force,
                                                                        Sensor.OutputType.ForceResultant
                                                                        )
            marker_img = sensor_0.drawMarkerMove(rectify_real_0)
            raw_depth_real = depth_real_0
            depth_real_0 = cv2.cvtColor(cm.jet(depth_real_0), cv2.COLOR_RGB2BGR)
            marker_real_uv = marker_geter.interpolate(marker_of_real_0)
            marker_real = marker_geter.target_grid + marker_real_uv
            scene.update()
            scene.updateSensors()
            raw_pic = scene.left_sensor.get_image()
            scene.left_sensor.set_show_marker(False)
            pic = scene.left_sensor.get_image()
            scene.left_sensor.set_show_marker(True)
            diff_raw = pic - ref_img
            diff = np.clip(diff_raw +100, 0, 255).astype(np.uint8)
            marker_sim = scene.left_sensor.get_marker()
            marker_uv_sim = scene.left_sensor.get_marker_displacement()
            raw_depth = scene.left_sensor.fem_sim.get_depth()
            raw_depth = cv2.resize(raw_depth, (400,700))
            depth = np.minimum(scene.left_sensor.fem_sim.get_depth(), 0)*-1
            depth = cv2.resize(depth, (400,700))
            depth = cv2.cvtColor(cm.jet(depth), cv2.COLOR_RGB2BGR)
            force_sim = scene.left_sensor.get_force().reshape(35,20,3)
            force_sim_xyz = scene.left_sensor.get_force_xyz()
            height_map = scene.left_sensor.get_height()
            height_map = np.minimum(height_map, 0)*-1
            vmax, vmin = float(height_map.max()), float(height_map.min())
            height_map = np.clip(height_map, vmin, vmax)
            height_map = (255.0 * (height_map - vmin) / (vmax - vmin)).astype(np.uint8)
            height_map = cv2.applyColorMap(height_map, cv2.COLORMAP_JET)
            


            global flag
            if flag == 1:
                print('======real======')
                print('rectify_real',rectify_real_0.shape)
                print('rectify_marker_real', marker_img.shape)
                print('diff_real',diff_real_0.shape)
                print('marker_of_real',marker_of_real_0.shape)
                print('marker_real_uv',marker_real_uv.shape)
                print('marker_real',marker_real.shape)
                print('depth_real',depth_real_0.shape)
                print('force_real',force_real_0.shape)
                print('res_force_real', res_force_real_0.shape)
                print('raw_depth_real', raw_depth_real.shape)
                
                print('======sim======')
                print('rectify_sim',raw_pic.shape)
                print('rectify_pure_sim',pic.shape)
                print('diff_sim',diff.shape)
                print('marker_sim',marker_sim.shape)
                print('raw_depth_sim',raw_depth.shape)
                print('depth_sim',depth.shape)
                print('force_sim',force_sim.shape)
                print('force_sim_xyz',force_sim_xyz.shape, force_sim_xyz)
                print('height_map',height_map.shape)

                print('=====marker=====')
                # print('marker_real_x', marker_real[0,:,0])
                # print('marker_real_y', marker_real[:,0,1])
                # print('marker_sim_x', marker_sim[0,:,0])
                # print('marker_sim_y', marker_sim[:,0,1])
                # print('marker_of_mm_x', marker_geter.source_grid[0,:,0])
                # print('marker_of_mm_y', marker_geter.source_grid[:,0,1])
                # print('marker_real_mm_x', marker_geter.target_grid[0,:,0])
                # print('marker_real_mm_y', marker_geter.target_grid[:,0,1])
                # print('real_depth', raw_depth_real)
                # print('sim_depth', raw_depth)
                
                print('end')
                flag = 0

            img_view.setData(rectify_real_0)
            img_view.setData(marker_img)
            # img_view.setData(depth_real_0)
            # img_view_2.setData(raw_pic)
            # img_view_2.setData(height_map)
            img_view_2.setData(diff_real_0)

            if sensor_1 is not None:

                rectify_real_1, diff_real_1, depth_real_1 = sensor_1.selectSensorInfo(Sensor.OutputType.Rectify, Sensor.OutputType.Difference, Sensor.OutputType.Depth)
                raw_pic = scene.right_sensor.get_image()
                img_view_4.setData(raw_pic)
                img_view_3.setData(rectify_real_1)


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

    def on_save():
        global cnt
        cnt += 1
        data_0 = {}
        # data_0["vertex"] = scene.left_sensor.vis_fem_mesh._mesh._vertexes.data.copy()
        # data_0["depth"] = np.minimum(scene.left_sensor.fem_sim.get_depth(), 0)
        # data_0["real_rectify"] = sensor_0.selectSensorInfo(Sensor.OutputType.Rectify)[0]
        # data_0["sim_diff"] = sensor_0.selectSensorInfo(Sensor.OutputType.Difference)[0]
        # data_0["real_depth"] = sensor_0.selectSensorInfo(Sensor.OutputType.Depth)[0]
        # data_0["real_diff"] = sensor_0.selectSensorInfo(Sensor.OutputType.Difference)[0]
        real_marker = sensor_0.selectSensorInfo(Sensor.OutputType.Marker2D)
        data_0["marker_real"] = marker_geter.interpolate(real_marker)
        data_0["marker_sim"] = -scene.left_sensor.get_marker_displacement()[::-1,::-1,:2]

        save_dir = DATA_PATH / item_name / 'sensor_0'
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'{item_name}.pkl', 'wb') as f:
                    pickle.dump(data_0, f)
        if sensor_1 is not None:
            data_1 = {}
            data_1["vertex"] = scene.right_sensor.vis_fem_mesh._mesh._vertexes.data.copy()
            data_1["depth"] = np.minimum(scene.right_sensor.fem_sim.get_depth(), 0)
        pass


    def on_object_pose_change(data):
        pos = tb.get_value("pos [m]")
        rotx = tb.get_value("euler [deg]")
        scene.object.setTransform(Matrix4x4.fromVector6d(*pos, *rotx))

    def set_tf_visible(data):
        scene.set_tf_visible(id=data[0], visible=data[1])
    
    # === show picture ===
    with tb.window("tb", size=(1600, 2800)):
        with tb.group("view", horizontal=True, collapsible=False, show=False):
            # img_view = tb.add_image_view("src" , None, img_size=(400, 700), img_format= "rgb")
            # img_view_2 = tb.add_image_view("dep" , None, img_size=(400, 700), img_format= "bgr")
            # img_view_3 = tb.add_image_view("diff_real" , None, img_size=(400, 700), img_format= "bgr")
            # img_view_4 = tb.add_image_view("dep_real" , None, img_size=(400, 700), img_format= "bgr")
            img_view = tb.add_image_view("real_L" , None, img_size=(400, 700), img_format= "bgr")
            img_view_2 = tb.add_image_view("sim_L" , None, img_size=(400, 700), img_format= "bgr")
            if sensor_1 is not None:
                    img_view_3 = tb.add_image_view("real_R" , None, img_size=(400, 700), img_format= "bgr")
                    img_view_4 = tb.add_image_view("sim_R" , None, img_size=(400, 700), img_format= "rgb")

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
            tb.add_button("save", callback=on_save)

        tb.add_timer("timer", interval_ms=2, callback=onTimeout)
        # tb.add_key_binding(tb.Key.Key_0, save_vertex)
        # tb.add_key_binding(tb.Key.Key_1, replay_vertex)
    tb.exec()

if __name__ == "__main__":
    main()