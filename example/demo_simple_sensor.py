import numpy as np

from ezgl import tb, Matrix4x4
from ezgl.items.scene import Scene
from ezgl.items import GLModelItem, DepthCamera, PointLight, GLAxisItem
from ezgl.utils.QtTools import qtcv
from xengym.render import VecTouchSim
from xengym import PROJ_DIR


class DepthRender(Scene):
    """一个简单的获取物体深度值的渲染类"""
    
    def __init__(self, visible=True):
        super().__init__(600, 400, visible)
        self.cameraLookAt((0.1, 0.1, 0.1), (0, 0, 0), (0, 1, 0))
        self.cam_view_width = 0.0194  # 19.4 mm  传感器感知面积
        self.cam_view_height = 0.0308   # 30.8 mm

        # 观察的物体
        self.object = GLModelItem(PROJ_DIR/"assets/obj/letter.STL", glOptions="translucent", lights=PointLight()).translate(0, 0.02, 0)

        # 深度相机
        self.axis = GLAxisItem((0.1, 0.1, 0.1), tip_size=0.1).translate(0.05, 0.05, 0.05)
        # ortho_space: left, right, bottom, top, near, far
        self.depth_cam = DepthCamera(
            self, eye=(0,0,0), center=(0,1,0), up=(0,0,1), img_size=(100, 175), proj_type="ortho",
            ortho_space=(-self.cam_view_width/2, self.cam_view_width/2, -self.cam_view_height/2, self.cam_view_height/2, -0.005, 0.1), 
            frustum_visible=True, actual_depth=True
        )
        
    def get_depth(self):
        """获取当前深度图"""
        return self.depth_cam.render()


if __name__ == '__main__':
    
    # 创建 VecTouchSim 实例
    sensor_sim = VecTouchSim(
        depth_size=(100, 175),  # 深度图尺寸
        visible=True,
        title="VecTouch Sensor Simulation"
    )

    # 创建 DepthRender 实例, 用来获取物体的深度值
    depth_scene = DepthRender(visible=True)
    
    # callbacks
    def on_object_pose_change(data):
        pos = tb.get_value("pos [m]")
        rotx = tb.get_value("euler [deg]")
        depth_scene.object.setTransform(Matrix4x4.fromVector6d(*pos, *rotx))

    def onTimeout():
        """更新场景"""
        if depth_scene.windowShouldClose() == False:
            depth_scene.update()
            depth = depth_scene.get_depth()
            sensor_pose = depth_scene.depth_cam.transform(local=False)
            object_pose = depth_scene.object.transform(local=False)

            sensor_sim.step(object_pose, sensor_pose, depth)
            sensor_sim.update()


    # ui
    with tb.window("Xensim", None, 10, pos=(300, 200), size=(400, 600)):
        tb.add_spacer(10)
        tb.add_text(" Settings")
        tb.add_spacer(10)
        
        with tb.group("left sensor", show=False, collapsed=False, collapsible=True, horizontal=False):
            tb.add_checkbox("enable smooth", value=True, callback=sensor_sim.enable_smooth)
            tb.add_checkbox("show marker", value=True, callback=sensor_sim.set_show_marker)
            tb.add_checkbox("show contact", value=False, callback=sensor_sim.set_show_contact)
            tb.add_checkbox("show force", value=False, callback=sensor_sim.set_show_force)
            tb.add_button("align view", callback=sensor_sim.align_view)

        with tb.group("object", show=False, collapsed=False, collapsible=True, horizontal=False):
            tb.add_drag_array(
                "pos [m]", value=[0., 0.02, 0.], min_val=[-1, -1, -1], max_val=[1, 1, 1], step=[0.0001]*3, decimals=[4]*3,
                format=[name+": %.4f" for name in ["x", "y", "z"]], callback=on_object_pose_change, horizontal=False
            )
            tb.add_drag_array(
                "euler [deg]", value=[0, 0, 0], min_val=[-180, -180, -180], max_val=[180, 180, 180], step=[0.1]*3, decimals=[1]*3,
                format=[name+": %.1f" for name in ["a", "b", "c"]], callback=on_object_pose_change, horizontal=False
            )
        tb.add_timer("timer", 30, onTimeout)

    tb.exec()