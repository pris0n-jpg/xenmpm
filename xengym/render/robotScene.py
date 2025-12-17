import numpy as np
from xensesdk.ezgl import (Matrix4x4, PointLight, GLURDFItem, GLModelItem, GLAxisItem,
                  GLGridItem, DepthCamera, MeshData, GLInstancedMeshItem)
from xensesdk.ezgl.items.scene import Scene
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import load_pem_public_key
import subprocess
from cryptography.fernet import Fernet
from .sensorSim import VecTouchSim
from .. import ASSET_DIR
import time

class RobotScene(Scene):

    def __init__(
        self,
        fem_file,
        urdf_file,
        object_file,
        depth_size=(100, 175),
        win_width = 800,
        win_height = 600,
        visible = False,
        left_visible = False,
        right_visible = False,
    ):
        # if not self.verify_valid_license(sudo_password):
        #     return
        """
        仿真环境

        Parameters:
        - fem_file : Path, fem文件路径
        - urdf_file : Path, urdf文件路径, urdf 需包含名为 left_finger 和 right_finger 的两个 link
        - object_file : Path, obj文件路径, 用于仿真的物体
        - depth_size : tuple, (width, height), 深度图尺寸
        - win_width : int, optional, default: 800, 窗口宽度
        - win_height : int, optional, default: 600, 窗口高度
        - visible : bool, optional, default: False, 是否显示窗口
        - left_visible : bool, optional, default: False, 是否显示左传感器
        - right_visible : bool, optional, default: False, 是否显示右传感器
        """
        super().__init__(win_width, win_height, visible)
        # -- lights
        light = PointLight(pos=(5, 3, 4), ambient=(0.5, 0.5, 0.5), diffuse=(0.3, 0.3, 0.3), visible=False, directional=True, render_shadow=False)

        # ==== robot, object ====
        self.robot = GLURDFItem(urdf_file, lights=[light], glOptions="translucent_cull")
        self.robot.set_joints([0, -np.pi/8, 0, -5*np.pi/8, 0, np.pi/2, np.pi/4, 0.04, 0.04])
        self.object = GLModelItem(object_file, glOptions="translucent", lights=light).translate(0.427, 0, 0.530)
        self.scale_factor = 1.0
        self.object.addChildItem(GLAxisItem(size=(0.12, 0.12, 0.12), tip_size=0.08))
        # -- grid
        self.grid = GLGridItem(
            size=(11, 11), spacing=(0.5, 0.5), lineWidth=1, color=np.array([0.78, 0.71, 0.60])*2.5, lineColor=(0.4, 0.3, 0.2), lights=[light]
        ).rotate(90, 1, 0, 0)

        # ==== Sensor Scene ====
        self.left_sensor = VecTouchSim(depth_size, fem_file, visible=left_visible, title="Left Sensor")
        self.right_sensor = VecTouchSim(depth_size, fem_file, visible=right_visible, title="Right Sensor")
        self.gel_size_mm = np.array(self.left_sensor.gel_size_mm)
        self.gel_size_m = self.gel_size_mm / 1000.0  # convert mm to m
        # ==== size of sensor, unit: m, 严格等于有限元网格的尺寸 ====
        width, height = self.gel_size_m

        # ==== cameras ====
        # left contact camera
        self._left_contact_cam = DepthCamera(
            self, eye=(0,0,0), center=(0,-1,0), up=(0,0,1), img_size=depth_size, proj_type="ortho",
            ortho_space=(-width/2, width/2, -height/2, height/2, -0.005, 0.1), frustum_visible=False, actual_depth=True
        )
        self.robot.get_link("left_finger").addChildItem(self._left_contact_cam)
        self._left_contact_cam.render_group.update(self.object)
        left_contact_cam_axis = GLAxisItem(size=(0.12, 0.12, 0.12), tip_size=0.08)
        left_contact_cam_axis.setVisible(False)
        self._left_contact_cam.addChildItem(left_contact_cam_axis)

        # right contact camera
        self._right_contact_cam = DepthCamera(
            self, eye=(0,0,0), center=(0,1,0), up=(0,0,1), img_size=depth_size, proj_type="ortho",
            ortho_space=(-width/2, width/2, -height/2, height/2, -0.005, 0.1), frustum_visible=False, actual_depth=True
        )
        self.robot.get_link("right_finger").addChildItem(self._right_contact_cam)
        self._right_contact_cam.render_group.update(self.object)
        right_contact_cam_axis = GLAxisItem(size=(0.12, 0.12, 0.12), tip_size=0.08)
        right_contact_cam_axis.setVisible(False)
        self._right_contact_cam.addChildItem(right_contact_cam_axis)

        # ==== axes ====
        axes = [link.axis for link in self.robot._links.values()]
        axes.extend([left_contact_cam_axis, right_contact_cam_axis])
        self._axes_name = self.robot.get_links_name()
        self._axes_name.extend(["left_sensor", "right_sensor"])
        self._axes = dict(zip(self._axes_name, axes))

        # NOTE: scene 自动将 cam 添加到渲染组， 但不是必须的, 因为 cam 已经在 urdf item 树中
        self.item_group.removeItem(self._left_contact_cam, self._right_contact_cam)

        # ==== custom axes ====
        vert_axis, ind_axis, color_axis = MeshData.axis_mesh(0.08, 0.0016, tip_width=4, tip_length=0.15)
        self._custom_axes = GLInstancedMeshItem(MeshData.Mesh(vert_axis, ind_axis), color=color_axis, color_divisor=0)

    def updateSensors(self):
        """
        更新传感器
        """
        obj_pose = self.object.transform(False)
        if self.left_sensor.visible:
            left_pose = self._left_contact_cam.transform(False)
            depth = self._left_contact_cam.render()
            self.left_sensor.step(obj_pose, left_pose, depth)
            self.left_sensor.update()

        if self.right_sensor.visible:
            right_pose = self._right_contact_cam.transform(False)
            depth = self._right_contact_cam.render()
            self.right_sensor.step(obj_pose, right_pose, depth)
            self.right_sensor.update()

    def set_data(self, obj_pose: np.ndarray, panda_joints=None):
        pose = Matrix4x4.fromVector7d(*obj_pose)
        self.object.setTransform(pose)
        if panda_joints is not None:
            panda_joints = np.array(panda_joints)
            panda_joints[-2:] -= 0.0003  # HACK
            self.robot.set_joints(panda_joints)

    def get_joints(self):
        return self.robot.get_joints_attr("value")

    def set_joint(self, joint_id: int, joint_value: float):
        self.robot.set_joint(joint_id, joint_value)

    def set_joints(self, panda_joints: np.ndarray):
        """
        panda_joints: np.ndarray, shape=(9,), unit: rad
        """
        self.robot.set_joints(panda_joints)

    def get_tf(self, id) -> Matrix4x4:
        if isinstance(id, int):
            id = self._axes_name[id]
        assert id in self._axes_name, f"Invalid id {id}"
        return self._axes[id].transform(False)

    def set_tf_visible(self, id, visible: bool):
        if id == "object":
            self.object.childItems()[0].setVisible(visible)
            return

        if isinstance(id, int):
            id = self._axes_name[id]
        assert id in self._axes_name, f"Invalid id {id}"

        if id not in ["left_sensor", "right_sensor"]:
            self.robot.get_link(id).set_data(visible)
        else:
            self._axes[id].setVisible(visible)

    def add_axis(self, pose):
        self._custom_axes.addData(pose)

    def set_axis(self, pose):
        self._custom_axes.setData(pose)

    def load_public_key_from_file(self, file_path):
        with open(file_path, 'rb') as key_file:
            public_key = load_pem_public_key(
                key_file.read(),
            )
        return public_key

    def verify_valid_license(self, sudopassword):
        # get signature

        try:
            # decrypt info 
            certificate_path = 'certificate.xense'

            # Load signature from file
            with open(ASSET_DIR /certificate_path, 'r') as file:
                certificate_hex = file.read()
                certificate = bytes.fromhex(certificate_hex)

            public_key_session, signature = certificate.split(b'::::::')

            public_key_password = public_key_session[:44]
            public_key_byte = public_key_session[44:]

            f = Fernet(public_key_password)
            public_key = load_pem_public_key(f.decrypt(public_key_byte))

            slice_idx = 5
            key_len = 44
            data_len = 100
            real_signature = signature[:slice_idx] + signature[slice_idx+key_len+data_len:]

            # get time 
            time_key = signature[slice_idx:slice_idx+key_len]
            time_data = signature[slice_idx+key_len:slice_idx+key_len+data_len]
            f = Fernet(time_key)
            decrypted_time = f.decrypt(time_data)
            

            # get data
            serial_numbers = next(iter(self.get_serial_numbers(sudopassword).values()))

            public_key.verify(
                signature=real_signature,
                data=serial_numbers.encode(),
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                algorithm=hashes.SHA256()
            )
            date_str = decrypted_time.decode()
            time_struct = time.strptime(date_str, "%Y%m%d")
            timestamp = time.mktime(time_struct)
            if time.time() > timestamp:
                print(
                "#############################################################\n"
                "################### Xense Sim Init Fail! ####################\n"
               f"################### Expire Time {decrypted_time.decode()} ####################\n"
                "################### Please contact XenseRobotics for help ###\n"
                "#############################################################\n")
                return False
            print(
                "#############################################################\n"
                "################### Xense Sim Init Suceess! #################\n"
               f"################### Expire Time {decrypted_time.decode()} ####################\n"
                "#############################################################\n"
                "#############################################################\n")
            return True
        except Exception as e:
            print(e)
            print(
                "#############################################################\n"
                "################### Xense Sim Init Fail! ####################\n"
                "################### Please contact XenseRobotics for help ###\n"
                "#############################################################\n"
                "#############################################################\n")
            return False

    def get_serial_numbers(self, password):
        commands = {
            'Baseboard Serial Number': 'dmidecode -s baseboard-serial-number'
        }
        results = {}
        for key, command in commands.items():
            try:
                result = subprocess.run(f'echo {password} | sudo -S {command}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
                if result.stderr:
                    results[key] = "Error: " + result.stderr
                else:
                    results[key] = result.stdout.strip()
            except Exception as e:
                results[key] = str(e)
        return results


if __name__ == '__main__':
    sim = RobotScene(
        fem_file="/home/lj/workspace/xengym/xengym/assets/data/fem_data_vec4070.npz",
        urdf_file="/home/lj/workspace/xengym/xengym/assets/panda/panda_with_vectouch.urdf",
        object_file="/home/lj/workspace/xengym/xengym/assets/obj/handle.STL",
        visible=True
    )
    import time
    while not sim.windowShouldClose():
        time.sleep(0.02)
        sim.update()