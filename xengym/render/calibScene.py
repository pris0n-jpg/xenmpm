import math
import numpy as np
from xensesdk.ezgl import (Matrix4x4, PointLight, GLModelItem, GLAxisItem,
                  GLGridItem, DepthCamera, MeshData, GLInstancedMeshItem)
from xensesdk.ezgl.experimental.GLSurfMeshItem import GLSurfMeshItem
from xensesdk.ezgl.items import Material
from xensesdk.ezgl.items.scene import Scene
from .sensorSim import VecTouchSim
from .sensorScene import SensorScene
from .. import ASSET_DIR, PROJ_DIR
import time
from pathlib import Path
from typing import List, Dict, Union, Optional
import json
import cv2


class CalibrationVecTouchSim(SensorScene):
    """
    支持raw_data的标定传感器
    """

    def __init__(
        self,
        depth_size: tuple,
        fem_file: str = None,
        raw_data: Dict = None,
        visible: bool = False,
        title: str = "Calibration Sensor"
    ):
        """
        标定传感器初始化
        
        Parameters:
        - depth_size : tuple, (width, height), 深度图尺寸
        - fem_file : str, FEM文件路径 (如果raw_data为None时使用)
        - raw_data : Dict, FEM原始数据
        - visible : bool, 是否显示传感器窗口
        - title : str, 传感器标题
        """
        # self.gel_size_mm = (17.3, 29.15)
        self.gel_size_mm = (17.2, 28.5)
        self.marker_dx_dy_mm = (1.31, 1.31)
        
        # Marker网格固定为20×11，不应该根据FEM节点数改变
        self.marker_row_col = (20, 11)
        
        # 调试信息：如果有raw_data，输出top_nodes数量供参考
        if raw_data is not None:
            n_top_nodes = len(raw_data.get('top_nodes', []))
            if n_top_nodes > 0:
                print(f"📐 FEM top_nodes数量: {n_top_nodes}, Marker网格固定为: {self.marker_row_col}")
        
        # 如果没有提供raw_data，需要确保有fem_file
        if raw_data is None and fem_file is None:
            from .. import PROJ_DIR
            fem_file = str(PROJ_DIR / "assets/data/fem_data_gel_2035.npz")
        
        # 如果有raw_data，fem_file可以为None（后面会重新创建FEMSimulator）
        if raw_data is not None:
            # 使用默认fem_file先初始化，后面会被替换
            temp_fem_file = fem_file
            if temp_fem_file is None:
                from .. import PROJ_DIR
                temp_fem_file = str(PROJ_DIR / "assets/data/fem_data_gel_2035.npz")
        else:
            temp_fem_file = fem_file
        
        # 调用父类构造函数
        super().__init__(
            temp_fem_file,
            depth_size,
            self.gel_size_mm,
            marker_row_col=self.marker_row_col,
            marker_dx_dy_mm=self.marker_dx_dy_mm,
            visible=visible,
            title=title
        )
        
        # 如果提供了raw_data，重新创建FEMSimulator
        if raw_data is not None:
            from ..fem.simulation import FEMSimulator
            self.fem_sim = FEMSimulator(
                self, 
                None,  # fem_file设为None，使用raw_data
                self.marker_row_col,
                self.marker_dx_dy_mm,
                depth_size=depth_size,
                gel_size_mm=self.gel_size_mm,
                raw_data=raw_data
            )


class CalibrationScene(Scene):
    """
    单传感器标定场景（新版：物体固定原点，传感器向下按压）
    保留原接口
    """

    def __init__(
        self,
        raw_data: Optional[Dict] = None,  # FEM raw_data from fem_processor
        object_files: List[str] = None,  # 标定物体文件列表
        depth_size=(100, 175),
        win_width=800,
        win_height=600,
        visible=False,
        sensor_visible=True,
    ):
        """
        标定场景初始化（新版：传感器移动）
        
        Parameters:
        - raw_data : Dict, optional, FEM原始数据，来自fem_processor
        - object_files : List[str], 标定物体文件路径列表
        - depth_size : tuple, (width, height), 深度图尺寸
        - win_width : int, 窗口宽度
        - win_height : int, 窗口高度
        - visible : bool, 是否显示窗口
        - sensor_visible : bool, 是否显示传感器窗口
        """
        super().__init__(win_width, win_height, visible)
        
        # 设置光源
        self.light = PointLight(
            pos=(5, 3, 4), 
            ambient=(0.5, 0.5, 0.5), 
            diffuse=(0.3, 0.3, 0.3), 
            visible=False, 
            directional=True, 
            render_shadow=False
        )

        # 网格
        self.grid = GLGridItem(
            size=(11, 11), 
            spacing=(0.5, 0.5), 
            lineWidth=1, 
            color=np.array([0.78, 0.71, 0.60]) * 2.5, 
            lineColor=(0.4, 0.3, 0.2), 
            lights=[self.light]
        ).rotate(90, 1, 0, 0)

        # 初始化传感器
        self.sensor = CalibrationVecTouchSim(
            depth_size, 
            raw_data=raw_data,  # 传递raw_data
            visible=sensor_visible, 
            title="Calibration Sensor"
        )
        self.gel_size_mm = np.array(self.sensor.gel_size_mm)
        self.gel_size_m = self.gel_size_mm / 1000.0  # mm to m
        width, height = self.gel_size_m

        # 深度相机（代表传感器位姿）
        self.contact_cam = DepthCamera(
            self,
            eye=(0, 0, 0),
            center=(0, 0, 1),
            up=(1, 0, 0),
            img_size=depth_size,
            proj_type="ortho",
            ortho_space=(-width/2, width/2, -height/2, height/2, -0.005, 0.1),
            frustum_visible=False,
            actual_depth=True
        )
        # 传感器初始位姿设为一定高度，方便调试
        self.contact_cam.setTransform(Matrix4x4.fromVector6d(0, 0, 0.05, 0, 0, -90))

        # 深度相机可视化：添加坐标轴
        cam_axis = GLAxisItem(size=(0.05, 0.05, 0.05), tip_size=0.04)
        cam_axis.setVisible(True)
        self.contact_cam.addChildItem(cam_axis)

        # 深度相机可视化：蓝色平面
        cam_plane = GLSurfMeshItem(
            (40, 40),
            x_range=(-width/8, width/8),
            y_range=(-height/8, height/8),
            lights=[self.light],
            material=Material(ambient=(0.6,0.6,0.6), diffuse=(0.7,0.7,0.7), specular=(0.2,0.2,0.2))
        )
        cam_plane.applyTransform(Matrix4x4.fromVector6d(0, 0, 0.05, 0, 0, -90))
        cam_plane.setVisible(True)
        self.contact_cam.addChildItem(cam_plane)
        # 添加到场景可视化树
        self.item_group.addItem(cam_plane)

        # 传感器安全高度（向上抬起避免接触）
        self.sensor_safe_height = 0.05  # m

        # 数据存储
        self.calibration_data = {}  # 标定数据存储
        self.substep_move_mm = 0.05  # 仿真中子步长
        self.substep_settle_time = 0.04  # 子步短暂停顿
        self.trajectory_config = self._load_trajectory_config()

        # 物体管理（保持接口）：加载后固定在原点
        self.object_files = object_files or []
        self.objects = {}
        self.current_object = None
        self.current_object_name = None

        self._load_objects()

        # FEM processor（按需创建）
        self._fem_processor = None

        # 场景视角
        self.cameraLookAt([0.05, 0, 0.04], [0, 0, 0.02], [0, 0, 1])

    def _load_objects(self):
        """加载所有标定物体（固定原点）"""
        for obj_file in self.object_files:
            obj_path = Path(obj_file)
            obj_name = obj_path.stem
            
            obj_model = GLModelItem(
                obj_file, 
                glOptions="translucent", 
                lights=self.light
            )
            obj_model.addChildItem(GLAxisItem(size=(0.05, 0.05, 0.05), tip_size=0.04))

            # 物体固定在原点，轻微默认旋转与旧版一致
            initial_pose = Matrix4x4.fromVector6d(
                0.0, 0.0, 0.0,
                90, 0, -90
            )
            obj_model.setTransform(initial_pose)
            obj_model.setVisible(False)
            
            self.objects[obj_name] = obj_model
            print(f"✓ 加载标定物体: {obj_name}")

    def _load_trajectory_config(self) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
        """读取并规范化轨迹配置"""
        traj_path = PROJ_DIR.parent / "calibration" / "obj" / "traj.json"
        if not traj_path.exists():
            print(f"⚠️ 未找到轨迹配置文件: {traj_path}")
            return {}

        try:
            with open(traj_path, "r", encoding="utf-8") as fp:
                raw_config = json.load(fp)
        except Exception as exc:
            print(f"⚠️ 轨迹配置解析失败: {exc}")
            return {}

        normalized: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

        for obj_name, traj_dict in raw_config.items():
            if not isinstance(traj_dict, dict):
                continue

            obj_trajs: Dict[str, List[Dict[str, float]]] = {}

            for traj_name, steps_payload in traj_dict.items():
                steps: List[Dict[str, float]] = []

                if isinstance(steps_payload, dict) and {"x", "y", "z"} <= steps_payload.keys():
                    x_seq = steps_payload.get("x", [])
                    y_seq = steps_payload.get("y", [])
                    z_seq = steps_payload.get("z", [])
                    for dx, dy, dz in zip(x_seq, y_seq, z_seq):
                        steps.append({
                            "x": float(dx),
                            "y": float(dy),
                            "z": float(dz)
                        })
                elif isinstance(steps_payload, list):
                    for entry in steps_payload:
                        if isinstance(entry, dict):
                            dx = float(entry.get("x", 0.0))
                            dy = float(entry.get("y", 0.0))
                            dz = float(entry.get("z", 0.0))
                        elif isinstance(entry, (list, tuple)) and len(entry) == 3:
                            dx, dy, dz = entry
                            dx, dy, dz = float(dx), float(dy), float(dz)
                        else:
                            continue

                        steps.append({"x": dx, "y": dy, "z": dz})

                if steps:
                    obj_trajs[traj_name] = steps

            if obj_trajs:
                normalized[obj_name] = obj_trajs

        if not normalized:
            print("⚠️ 轨迹配置中没有可用的轨迹步骤")

        return normalized

    def set_current_object(self, object_name: str):
        """设置当前活动物体"""
        if object_name not in self.objects:
            raise ValueError(f"物体 '{object_name}' 未找到")
        
        if self.current_object is not None:
            self.current_object.setVisible(False)
        
        self.current_object = self.objects[object_name]
        self.current_object_name = object_name
        self.current_object.setVisible(True)
        
        # 更新深度相机的渲染组
        self.contact_cam.render_group.update(self.current_object)
        
        print(f"✓ 切换到物体: {object_name}")

    # 新增：移动传感器（DepthCamera）到指定z
    def _set_sensor_z(self, target_z: float):
        tf = self.contact_cam.transform(False)
        x, y = tf.xyz[0], tf.xyz[1]
        a, b, c = tf.euler
        new_pose = Matrix4x4.fromVector6d(x, y, target_z, a, b, c)
        self.contact_cam.setTransform(new_pose)

    def _move_sensor_by_delta(self, dx_mm: float, dy_mm: float, dz_mm: float):
        """根据增量(单位mm)移动传感器"""
        tf = self.contact_cam.transform(False)
        base_x, base_y, base_z = tf.xyz
        a, b, c = tf.euler
        new_pose = Matrix4x4.fromVector6d(
            base_x + dx_mm / 1000.0,
            base_y + dy_mm / 1000.0,
            base_z + dz_mm / 1000.0,
            a, b, c
        )
        self.contact_cam.setTransform(new_pose)

    def move_sensor_to_contact(self):
        """改为移动传感器到刚好接触位置（接口名保持不变）"""
        if self.current_object is None:
            raise ValueError("未选择当前物体")
        
        # 确保渲染组包含当前物体
        self.contact_cam.render_group.update(self.current_object)
        
        # 当前传感器z
        sensor_tf = self.contact_cam.transform(False)
        sensor_z = sensor_tf.xyz[2]
        
        # 渲染深度，计算最小深度
        depth = self.contact_cam.render()
        valid = depth > -1000
        if not np.any(valid):
            return sensor_z
        min_depth = np.min(depth[valid])
        
        # 将最小深度调为接触（约0）
        contact_z = sensor_z - float(min_depth)
        self._set_sensor_z(contact_z)
        
        # 更新传感器与仿真
        self.update_sensor()
        
        # 微调验证
        final_depth = self.contact_cam.render()
        valid2 = final_depth > -1000
        if np.any(valid2):
            min_d2 = np.min(final_depth[valid2])
            if abs(min_d2) > 1e-3:
                # 小幅度修正
                self._set_sensor_z(contact_z - min_d2 * 0.8)
                self.update_sensor()
                contact_z = self.contact_cam.transform(False).xyz[2]
        
        # 记录并返回接触高度（传感器z）
        self.sensor_contact_z = contact_z
        return contact_z
    
    def move_sensor_to_safe_height(self):
        """改为将传感器抬至安全高度（接口名保持不变）"""
        # 以当前x,y与姿态不变，仅调整z
        self._set_sensor_z(self.sensor_safe_height)
        self.update_sensor()

    def press_sensor_to_depth(self, target_depth_mm: float):
        """
        改为将传感器按压到指定深度（接口与参数保持不变）
        target_depth_mm: 目标按压深度（相对接触位姿）
        """
        if self.current_object is None:
            raise ValueError("未选择当前物体")
        
        # 若尚未找到接触，先对齐接触
        contact_z = getattr(self, 'sensor_contact_z', None)
        if contact_z is None:
            contact_z = self.move_sensor_to_contact()
        
        # 按压深度：向下移动传感器（-Z）
        target_z = contact_z - target_depth_mm / 1000.0
        self._set_sensor_z(target_z)
        
        # 更新传感器
        self.update_sensor()

    def _execute_trajectory(self, trajectory_name: str, steps: List[Dict[str, float]]) -> Dict[str, Dict]:
        """按照给定轨迹执行传感器运动并采集每一步的数据"""
        if self.current_object is None:
            raise ValueError("未选择当前物体")

        if not steps:
            return {}

        # print(f"▶️  执行轨迹: {trajectory_name} (共 {len(steps)} 步)")

        # 起始前确保回到安全高度并取得接触位姿
        self.move_sensor_to_safe_height()
        self.move_sensor_to_contact()

        trajectory_data: Dict[str, Dict] = {}

        for idx, step in enumerate(steps):
            dx = float(step.get("x", 0.0))
            dy = float(step.get("y", 0.0))
            dz = float(step.get("z", 0.0))

            max_delta = max(abs(dx), abs(dy), abs(dz))
            if max_delta <= 0:
                substeps = 1
            else:
                substeps = max(1, math.ceil(max_delta / self.substep_move_mm))

            inc_dx = dx / substeps
            inc_dy = dy / substeps
            inc_dz = dz / substeps

            remaining_dx, remaining_dy, remaining_dz = dx, dy, dz

            for sub_idx in range(substeps):
                if sub_idx == substeps - 1:
                    step_dx = remaining_dx
                    step_dy = remaining_dy
                    step_dz = remaining_dz
                else:
                    step_dx = inc_dx
                    step_dy = inc_dy
                    step_dz = inc_dz

                self._move_sensor_by_delta(step_dx, step_dy, step_dz)
                self.update_sensor()

                remaining_dx -= step_dx
                remaining_dy -= step_dy
                remaining_dz -= step_dz

            # 最后一子步完成后再额外等待以稳定
            self.update_sensor()

            metadata = {
                "trajectory": trajectory_name,
                "step_index": idx,
                "commanded_delta_mm": (dx, dy, dz)
            }

            step_data = self._collect_current_depth_data(metadata=metadata)
            if step_data:
                step_key = f"step_{idx:03d}"
                trajectory_data[step_key] = step_data

        self.move_sensor_to_safe_height()

        # print(f"✅  轨迹 {trajectory_name} 完成，采集 {len(trajectory_data)} 条数据")
        return trajectory_data
    
    def progressive_press_to_depth(self, target_depth_mm: float, step_size_mm: float = 0.02, 
                                 stabilization_time: float = 0.1) -> Dict:
        """
        渐进按压到指定深度，并在过程中采集数据（移动传感器）
        """
        if self.current_object is None:
            raise ValueError("未选择当前物体")
        
        contact_z = self.move_sensor_to_contact()
        
        current_depth = 0.0
        collected_data = []
        
        while current_depth < target_depth_mm:
            next_depth = min(current_depth + step_size_mm, target_depth_mm)
            next_z = contact_z - next_depth / 1000.0
            self._set_sensor_z(next_z)
            self.update_sensor()
            
            current_data = self._collect_current_depth_data(next_depth)
            if current_data:
                collected_data.append(current_data)
            current_depth = next_depth
        
        return collected_data[-1] if collected_data else {}
    
    def _collect_current_depth_data(self, depth_mm: Optional[float] = None, metadata: Optional[Dict] = None) -> Dict:
        """
        采集当前深度的数据
        """
        try:
            depth_field = self.contact_cam.render()
            data: Dict[str, Union[np.ndarray, Dict, float, tuple]] = {
                "depth_field": depth_field.astype(np.float32)
            }
            # Marker位移数据 (20x11x2)
            marker_3d = self.sensor.get_marker_displacement()  # shape: (20, 11, 3)
            marker_displacement = -marker_3d[::-1, ::-1, :2]
            # marker_displacement = -marker_displacement[::-1, ::-1, :]
            data['marker_displacement'] = marker_displacement.astype(np.float32)

            # 三维力数据 (3,)
            force_xyz = self.sensor.get_force_xyz()
            data['force_xyz'] = force_xyz.astype(np.float32)


            if depth_mm is not None:
                data['target_depth_mm'] = float(depth_mm)

            if metadata:
                data['metadata'] = metadata
            
            return data
        except Exception as e:
            print(f"  ❌ 深度 {depth_mm}mm 数据采集失败: {e}")
            return {}

    def update_sensor(self):
        """更新传感器状态（新版：传感器位姿来自contact_cam）"""
        if self.current_object is None:
            return

        obj_pose = self.current_object.transform(False)
        sensor_pose = self.contact_cam.transform(False)
        depth = self.contact_cam.render()
        self.sensor.step(obj_pose, sensor_pose, depth)
        self.sensor.update()

    def collect_data_for_depth(self, depth_mm: float, use_progressive: bool = True) -> Dict:
        """
        采集指定深度的数据
        """
        if self.current_object is None:
            raise ValueError("未选择当前物体")
        
        if use_progressive:
            return self.progressive_press_to_depth(depth_mm)
        else:
            self.press_sensor_to_depth(depth_mm)
            return self._collect_current_depth_data(depth_mm)

    def collect_calibration_data(self, object_name: str = None) -> Dict:
        """采集指定物体的全部轨迹数据"""
        if object_name is not None:
            self.set_current_object(object_name)

        if self.current_object is None:
            raise ValueError("未选择物体")

        obj_name = self.current_object_name
        trajectories = self.trajectory_config.get(obj_name, {})

        if not trajectories:
            print(f"⚠️ 物体 '{obj_name}' 未在轨迹配置中找到，跳过")
            return {obj_name: {}}

        object_data: Dict[str, Dict[str, Dict]] = {}

        for traj_name, steps in trajectories.items():
            try:
                traj_data = self._execute_trajectory(traj_name, steps)
                if traj_data:
                    object_data[traj_name] = traj_data
            except Exception as exc:
                print(f"  ❌ 轨迹 {traj_name} 执行失败: {exc}")

        self.calibration_data[obj_name] = object_data
        print(f"✓ 物体 '{obj_name}' 轨迹数据采集完成，共 {len(object_data)} 条轨迹")
        return {obj_name: object_data}
    

    def collect_all_calibration_data(self) -> Dict:
        """
        采集所有物体的标定数据
        
        Returns:
        - Dict, 完整的标定数据字典
        """
        print("🎯 开始采集所有物体的标定数据...")
        
        all_data = {}
        
        for obj_name in self.objects.keys():
            try:
                obj_data = self.collect_calibration_data(obj_name)
                all_data.update(obj_data)
                # print(f"✓ 物体 '{obj_name}' 数据采集完成")
                
            except Exception as e:
                print(f"❌ 物体 '{obj_name}' 数据采集失败: {e}")
                continue
        
        self.calibration_data = all_data
        print("🎉 所有标定数据采集完成!")
        return all_data

    def update_fem_data(self, raw_data: Dict, coef: float=0.000):
        """
        更新FEM数据（用于不同材料参数的测试）
        """
        print("🔄 更新FEM材料参数...")
        
        from ..fem.simulation import FEMSimulator
        self.sensor.fem_sim = FEMSimulator(
            self.sensor, 
            None,  # fem_file为None，使用raw_data
            self.sensor.marker_row_col,
            self.sensor.marker_dx_dy_mm,
            depth_size=(100, 175),  # 使用默认深度尺寸
            gel_size_mm=self.sensor.gel_size_mm,
            raw_data=raw_data,
            nonlinear_coef=coef
        )
        
        print("✓ FEM材料参数更新完成")
    
    def calibrate_with_parameters(self, E: float, nu: float, coef: float=0.0) -> Dict:
        """
        使用指定材料参数进行标定
        """
        # 参数验证和精度限制
        E = round(float(E), 4)
        nu = round(float(nu), 4)
        coef = round(float(coef), 3)
        
        print(f"🎯 使用材料参数 E={E:.4f}, nu={nu:.4f} 进行标定...")
        
        try:
            import sys
            from pathlib import Path
            possible_paths = [
                Path(__file__).parent.parent / "calibration",
                Path("calibration"),
                Path("../calibration"),
            ]
            
            fem_processor_found = False
            for calibration_dir in possible_paths:
                if calibration_dir.exists() and (calibration_dir / "fem_processor.py").exists():
                    if str(calibration_dir) not in sys.path:
                        sys.path.insert(0, str(calibration_dir))
                    fem_processor_found = True
                    break
            
            if not fem_processor_found:
                raise ImportError("无法找到 fem_processor 模块")
                
            from fem_processor import process_gel_data
            
            if not hasattr(self, '_fem_processor') or self._fem_processor is None:
                print("🔧 首次创建FEM processor实例...")
                self._fem_processor = process_gel_data('g1-ws', E=E, nu=nu, use_cache=True)
            else:
                print("🔄 更新现有FEM processor的材料参数...")
                self._fem_processor.update_material_properties(E=E, nu=nu)
            
            raw_data = self._fem_processor.get_data()
            
            self.update_fem_data(raw_data, coef)
            
            self.reset_scene()
            
            calibration_data = self.collect_all_calibration_data()
            
            print(f"✓ 材料参数 E={E}, nu={nu}, coef={coef} 标定完成")
            return calibration_data
            
        except Exception as e:
            print(f"❌ 标定失败: {e}")
            return {}

    def reset_scene(self):
        """重置场景状态（抬起传感器、复位FEM）"""
        self.move_sensor_to_safe_height()
        self.sensor.fem_sim.reset()
        print("✓ 场景已重置")

    def get_available_objects(self) -> List[str]:
        """获取可用物体列表"""
        return list(self.objects.keys())

    def get_calibration_data_summary(self) -> Dict:
        """获取标定数据摘要"""
        if not self.calibration_data:
            return {"message": "无标定数据"}
        
        summary: Dict[str, Dict] = {}
        for obj_name, traj_data in self.calibration_data.items():
            traj_summary: Dict[str, Dict] = {}
            for traj_name, steps in traj_data.items():
                step_count = len(steps)
                first_step = next(iter(steps.values())) if steps else {}
                traj_summary[traj_name] = {
                    "step_count": step_count,
                    "has_depth_field": bool(first_step) and "depth_field" in first_step,
                    "has_marker_displacement": bool(first_step) and "marker_displacement" in first_step,
                    "has_force": bool(first_step) and "force_xyz" in first_step
                }

            summary[obj_name] = {
                "trajectory_count": len(traj_data),
                "trajectories": traj_summary
            }

        return summary


# 便利函数

def create_calibration_scene(
    object_files: List[str],
    raw_data: Optional[Dict] = None,
    **kwargs
) -> CalibrationScene:
    """
    创建标定场景（新版：传感器移动）
    """
    return CalibrationScene(
        raw_data=raw_data,
        object_files=object_files,
        **kwargs
    )


if __name__ == '__main__':
    from pathlib import Path
    import sys
    import time
    try:
        from xensesdk.ezgl import tb
    except Exception as e:
        print(f"❌ 无法导入可视化工具 tb: {e}")
        sys.exit(1)

    # 准备物体文件
    asset_dir = Path("/home/czl/Downloads/workspace/xengym/calibration/obj")
    candidates = [
            "circle_r3.STL", 
            "circle_r4.STL", 
            "circle_r5.STL",
            "r3d5.STL",
            "r4d5.STL",
            "rhombus_d6.STL",
            "rhombus_d8.STL",
            "square_d6.STL",
            "square_d8.STL",
            "tri_d6.STL"
            ]
    
    object_files = [str(asset_dir / n) for n in candidates if (asset_dir / n).exists()]
    if not object_files:
        print("❌ 未找到可用的物体模型")
        sys.exit(1)

    # 创建场景（新版）
    possible_paths = [
                Path(__file__).parent.parent / "calibration",
                Path("calibration"),
                Path("../calibration"),
            ]
            
    fem_processor_found = False
    for calibration_dir in possible_paths:
        if calibration_dir.exists() and (calibration_dir / "fem_processor.py").exists():
            if str(calibration_dir) not in sys.path:
                sys.path.insert(0, str(calibration_dir))
            fem_processor_found = True
            break
 
    from fem_processor import process_gel_data
    fem_pro = process_gel_data('g1-ws', E=0.7966, nu=0.3523, use_cache=True)
    
    raw_data = fem_pro.get_data()
    scene = CalibrationScene(
        raw_data=raw_data,
        object_files=object_files,
        visible=True,
        sensor_visible=True,
    )
    scene.update_fem_data(raw_data, coef=-0.015)

    # 默认选择第一个物体
    first_obj = list(scene.objects.keys())[0]
    scene.set_current_object(first_obj)
    
    # 存储力数据的全局变量
    force_data = {'fx': 0.0, 'fy': 0.0, 'fz': 0.0}

    # UI 交互函数（接口保持一致）
    def ui_set_object(choice=0):
        names = list(scene.objects.keys())
        if isinstance(choice, int):
            if 0 <= choice < len(names):
                scene.set_current_object(names[choice])
        else:
            if choice in names:
                scene.set_current_object(choice)

    def ui_move_to_contact():
        try:
            scene.move_sensor_to_contact()
            # 同步更新UI滑块为新的传感器位置
            try:
                tf = scene.contact_cam.transform(False)
                xyz = list(getattr(tf, 'xyz', [0.0, 0.0, 0.05]))
                tb.set_value("pos [m]", xyz)
            except Exception as _:
                pass
        except Exception as e:
            print(f"move_to_contact 失败: {e}")

    def ui_press(depth_mm=0.2):
        try:
            scene.collect_data_for_depth(depth_mm, use_progressive=True)
        except Exception as e:
            print(f"press 失败: {e}")

    def ui_collect_all():
        try:
            data = scene.collect_all_calibration_data()
            # PROJ_DIR 指向 xengym/ 目录，需要回到父目录才是项目根
            storage_file = PROJ_DIR.parent / "calibration" / "data" / "sim_data.pkl"
            
            # 确保目录存在
            storage_file.parent.mkdir(parents=True, exist_ok=True)
            
            import pickle
            with open(storage_file, 'wb') as fp:
                pickle.dump(data, fp)
            
            print(f"✓ 数据已保存至: {storage_file}")
            print(f"📊 数据摘要:")
            for obj, depths in data.items():
                print(f"  {obj}: {list(depths.keys())}")

        except Exception as e:
            print(f"❌ collect_all 失败: {e}")
            import traceback
            traceback.print_exc()

    # 位姿滑动块（保留，用于少量校正；默认在原点）
    def on_sensor_pose_change(data=None):
        try:
            # 读取传感器当前位置与朝向，仅更新xyz
            tf = scene.contact_cam.transform(False)
            euler = tf.euler
            new_pos = tb.get_value("pos [m]")
            pose = Matrix4x4.fromVector6d(*new_pos, *euler)
            scene.contact_cam.setTransform(pose)
            scene.update_sensor()
        except Exception as e:
            print(f"传感器位姿更新失败: {e}")

    # 定时器：驱动渲染并更新力数据显示
    def on_timeout():
        if not scene.windowShouldClose():
            scene.update()
            # 更新实时力数据显示
            try:
                force_xyz = scene.sensor.get_force_xyz()
                force_data['fx'] = float(force_xyz[0])
                force_data['fy'] = float(force_xyz[1])
                force_data['fz'] = float(force_xyz[2])
                tb.set_value("force_x", force_data['fx'])
                tb.set_value("force_y", force_data['fy'])
                tb.set_value("force_z", force_data['fz'])
            except Exception as e:
                pass

    # 构建简单UI（保留旧接口与布局）
    with tb.window("Calibration (Sensor Down)", None, 10, pos=(200, 100), size=(420, 420)):
        tb.add_text("标定场景控制（传感器下压）")
        tb.add_spacer(6)
        tb.add_text("物体选择")
        names = list(scene.objects.keys())
        if names:
            tb.add_combo("object", names, callback=lambda i: ui_set_object(i))
        tb.add_spacer(6)
        tb.add_button("对齐接触", callback=lambda: ui_move_to_contact())
        tb.add_spacer(4)
        tb.add_drag_value("按压深度(mm)", value=0.2, min_val=0.1, max_val=0.8, step=0.1,
                         decimals=3, format="%.3f",
                         callback=lambda v: ui_press(v))
        tb.add_spacer(8)

        # 传感器初始位置
        try:
            tf0 = scene.contact_cam.transform(False)
            init_pos = getattr(tf0, 'xyz', [0.0, 0.0, 0.05])
        except Exception:
            init_pos = [0.0, 0.0, 0.05]
        tb.add_drag_array(
            "pos [m]",
            value=init_pos,
            min_val=[-0.05, -0.05, -0.05],
            max_val=[0.05, 0.05, 0.05],
            step=[0.0001, 0.0001, 0.0001],
            decimals=[4, 4, 4],
            format=[name+": %.4f" for name in ["x", "y", "z"]],
            callback=on_sensor_pose_change,
            horizontal=False
        )
        tb.add_spacer(6)
        tb.add_button("采集所有深度", callback=lambda: ui_collect_all())
        tb.add_spacer(8)
        
        # 实时力数据显示框
        tb.add_text("实时力数据 (Force XYZ):")
        tb.add_drag_value("force_x", value=0.0, min_val=-100.0, max_val=100.0,
                         step=0.0001, decimals=4, format="Fx: %.4f N")
        tb.add_drag_value("force_y", value=0.0, min_val=-100.0, max_val=100.0,
                         step=0.0001, decimals=4, format="Fy: %.4f N")
        tb.add_drag_value("force_z", value=0.0, min_val=-100.0, max_val=100.0,
                         step=0.0001, decimals=4, format="Fz: %.4f N")
        tb.add_spacer(6)
        
        tb.add_text("提示: 关闭GL窗口或Ctrl+C可退出")
        tb.add_timer("timer", interval_ms=16, callback=on_timeout)

    try:
        tb.exec()
    except KeyboardInterrupt:
        print("\n👋 退出") 
