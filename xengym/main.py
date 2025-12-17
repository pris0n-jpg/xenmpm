import numpy as np

from xensesdk.ezgl import tb, Matrix4x4
from xensesdk.ezgl.utils.QtTools import qtcv
from .render.robotScene import RobotScene


def main(args):
    
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

    # callbacks
    def onTimeout():
        if scene.windowShouldClose() == False:
            scene.update()
            scene.updateSensors()

    def on_object_pose_change(data):
        pos = tb.get_value("pos [m]")
        rotx = tb.get_value("euler [deg]")
        scene.object.setTransform(Matrix4x4.fromVector6d(*pos, *rotx))

    def set_tf_visible(data):
        scene.set_tf_visible(id=data[0], visible=data[1])

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
