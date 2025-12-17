import numpy as np
import argparse
from pathlib import Path
from xensesdk.ezgl import tb, Matrix4x4
from xensesdk.ezgl.utils.QtTools import qtcv
from xengym.render.robotScene import RobotScene
from xensesdk.ezgl import GLViewWidget, tb
import pickle
from xengym import PROJ_DIR
import cv2
CURR_PATH = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description='Xense Sim')
    # 添加参数
    parser.add_argument('-f', '--fem_file', type=str, help='Path to the FEM file (default: %(default)s)', default=str(PROJ_DIR/"assets/data/fem_data_vec4070.npz"))
    parser.add_argument('-u', '--urdf_file', type=str, help='Path to the URDF file (default: %(default)s)', default=str(PROJ_DIR/"assets/panda/panda_with_vectouch.urdf"))
    parser.add_argument('-o', '--object_file', type=str, help='Path to the object file (default: %(default)s)', default=str(PROJ_DIR/"assets/obj/circle_r4.STL"))
    parser.add_argument('-l', '--show_left', help='Show left sensor (default: %(default)s)', action='store_true', default=False)
    parser.add_argument('-r', '--show_right', help='Show right sensor (default: %(default)s)', action='store_true', default=False)

    args = parser.parse_args()
    if not args.show_left and not args.show_right:
        args.show_left = True

    scene = RobotScene(
        fem_file=args.fem_file,
        urdf_file=args.urdf_file,
        object_file=args.object_file,
        visible=True,
        left_visible=args.show_left,
        right_visible=args.show_right
    )

    scene.cameraLookAt([1.5, 0, 0.8], [0, 0, 0.4], [0, 0, 1])

    # get ref img 
    scene.left_sensor.set_show_marker(False)
    ref_img = scene.left_sensor.get_image()
    scene.left_sensor.set_show_marker(True)
    def on_change_scale_change(val):
        scale = tb.get_value("scale")
        scene.scale_factor = 1 / scene.scale_factor
        scene.object.scaleData(scene.scale_factor)
        scene.scale_factor = scale
        scene.object.scaleData(scene.scale_factor)
        
    # callbacks
    def onTimeout():
        if scene.windowShouldClose() == False:
            scene.update()
            scene.updateSensors()
            scene.left_sensor.set_show_marker(False)
            pic = scene.left_sensor.get_image()
            scene.left_sensor.set_show_marker(True)
            diff_raw = pic - ref_img
            # print(diff_raw.min(), diff_raw.max())
            diff = np.clip((diff_raw+100), 0, 255).astype(np.uint8)
            img_view.setData(diff)
            if tb.get_widget("save pic").value:
                save_pic(diff)
                save_pose()
                print("save ok")
                tb.set_value("save pic", False)


    def save_pic(img):
        save_file_name = tb.get_widget("file name").value
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(CURR_PATH/ "data_collect" /f'{save_file_name}.jpg', img_bgr)

    def save_pose():
        save_file_name = tb.get_widget("file name").value
        obj_pose = scene.object.transform(False)
        left_pose = scene._left_contact_cam.transform(False)
        T_left_obj = (obj_pose * left_pose.inverse())
        pose_list = T_left_obj.toVector7d().tolist()
        with open(CURR_PATH/ "data_collect" /f'{save_file_name}.pkl', 'wb') as f:
            pickle.dump(pose_list, f)

    def on_object_pose_change(data):
        pos = tb.get_value("pos [m]")
        rotx = tb.get_value("euler [deg]")
        scene.object.setTransform(Matrix4x4.fromVector6d(*pos, *rotx))

    def set_tf_visible(data):
        scene.set_tf_visible(id=data[0], visible=data[1])
    # === show picture ===
    with tb.window("tb", size=(400, 700)):
        with tb.group("view", horizontal=True, collapsible=False, show=False):
            img_view = tb.add_image_view("src" , None, img_size=(400, 700), img_format= "rgb")
            with tb.group("view", horizontal=False, collapsible=False, show=False):
                tb.add_text_editor("file name")
                tb.add_checkbox("save pic")

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
                "pos [m]", value=[0.427, 0, 0.530], min_val=[-1, -1, -1], max_val=[1, 1, 1], step=[0.0001]*3, decimals=[4]*3,
                format=[name+": %.4f" for name in ["x", "y", "z"]], callback=on_object_pose_change, horizontal=False
            )
            tb.add_drag_array(
                "euler [deg]", value=[0, 0, 0], min_val=[-180, -180, -180], max_val=[180, 180, 180], step=[0.1]*3, decimals=[1]*3,
                format=[name+": %.1f" for name in ["a", "b", "c"]], callback=on_object_pose_change, horizontal=False
            )
            tb.add_drag_value("scale", value=1, min_val=0.1, max_val=2, step=0.02, callback=on_change_scale_change)
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

        tb.add_timer("timer", interval_ms=2, callback=onTimeout)

    tb.exec()

if __name__ == "__main__":
    main()