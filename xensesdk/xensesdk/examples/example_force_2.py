import sys
import numpy as np
from xensesdk import ExampleView
from xensesdk import Sensor
from xensesdk.ezgl import GLViewWidget
from xensesdk.ezgl.items import GLAxisItem, GLGridItem, PointLight
from xensesdk.omni.GLOmniItem import OmniSensorItem
from xensesdk.omni.GLWedgeSensorItem import GLWedgeSensorItem
from xensesdk.omni.GLFingerTipItem import GLFingerTipItem
from xensesdk.xenseInterface.sensorEnum import SensorType
from xensesdk.ezgl.utils.colormap import cm
from xensesdk import MACHINE
from qtpy import QtCore, QtWidgets
from xensesdk.ezgl import tb

flag = 1

class DualSensorView(GLViewWidget):
    """双传感器显示视图"""
    
    def __scale_grid(self, grid):
        grid = grid.astype(np.float32) / (400, 700)
        grid = grid * [0.94, 0.93] + [0.03, 0.003]
        return grid

    def __init__(self, sensor_0, sensor_1):
        # 使用更现代的方式处理高DPI
        try:
            # 对于新版本Qt，使用setHighDpiScaleFactorRoundingPolicy
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
                QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
        except AttributeError:
            # 对于旧版本Qt，使用AA_EnableHighDpiScaling（忽略弃用警告）
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        
        self.app = QtWidgets.QApplication(sys.argv)
        
        super().__init__(parent=None)
        self.sensor_0 = sensor_0
        self.sensor_1 = sensor_1
        self.initUI()
        tb.add_timer("timer", 10, self.onTimeout, self)

    def initUI(self):
        self.camera.set_vector6d((0.3, -1.4, -11, 30, -67, -121))
        self.axis = GLAxisItem(size=(3, 3, 3))
        self.axis.setVisible(0)
        
        # 灯光设置
        self.light = PointLight(pos=(5, 10, 10), ambient=(0.6, 0.6, 0.6), diffuse=(0.6, 0.6, 0.6),
                               visible=False, directional=True, render_shadow=False)
        self.light.setShadow(-3, 3, -3, 3, -3, 3, bias=0.05)

        # 网格
        self.griditem = GLGridItem(
            size=(11, 11), spacing=(0.5, 0.5), lineWidth=1, color=np.array([0.78, 0.71, 0.60])*1.5,
            lineColor=(0.4, 0.3, 0.2), lights=[self.light]
        ).rotate(90, 1, 0, 0)

        # 创建两个传感器模型，并排显示
        self.sensor_item_0 = self._create_sensor_item(self.sensor_0).translate(-3, 0, 0)  # 左侧
        self.sensor_item_1 = self._create_sensor_item(self.sensor_1).translate(3, 0, 0)   # 右侧
        
        if MACHINE != "aarch64":
            self.addItem(self.griditem)
            self.addItem(self.axis)
        self.addItem(self.sensor_item_0)
        self.addItem(self.sensor_item_1)
    
    def _create_sensor_item(self, sensor):
        """根据传感器类型创建对应的3D模型"""
        if sensor.sensor_type == SensorType.Omni:
            return OmniSensorItem(lights=[self.light], grid_size=(100, 200))
        elif sensor.sensor_type == SensorType.VecTouch:
            return GLWedgeSensorItem(lights=[self.light], grid_size=(110, 200))
        elif sensor.sensor_type in [SensorType.OmniB]:
            return OmniSensorItem(lights=[self.light])
        elif sensor.sensor_type == SensorType.Finger:
            return GLFingerTipItem(lights=[self.light]).rotate(90,0,0,1).rotate(90,0,1,0).translate(0,0,0.1)
        else:
            raise Exception(f"No sensor type {sensor.sensor_type}!")
    
    def setCallback(self, function):
        self._callback_func = function

    def onTimeout(self):
        self._callback_func()
        self.update()
    
    def setForceFlow(self, sensor_idx, force, res_force, mesh_init):
        """设置指定传感器的力数据显示"""
        sensor_item = self.sensor_item_0 if sensor_idx == 0 else self.sensor_item_1
        
        F_len = np.linalg.norm(force, axis=2)*4
        color = cm.yellow_red(F_len)
        force[F_len < 0.05] = 0  # 滤掉杂力

        if isinstance(sensor_item, GLWedgeSensorItem):
            sensor_item.set_force(res_force[:3], res_force[3:])
            force[..., 2] = -force[..., 2] * 2
            sensor_item.set_3d_arrow(mesh_init[1:-1, 1:-1]/10, mesh_init[1:-1, 1:-1]/10 + force[1:-1, 1:-1], color[1:-1, 1:-1])

        if isinstance(sensor_item, OmniSensorItem):
            sensor_item.set_force(res_force[:3], res_force[3:])
            force = sensor_item.process_force_for_show(force)
            sensor_item.set_3d_arrow(mesh_init[1:-1, 1:-1]/10, mesh_init[1:-1, 1:-1]/10 + force[1:-1, 1:-1], color[1:-1, 1:-1])

        if isinstance(sensor_item, GLFingerTipItem):
            sensor_item.set_force(res_force[:3], res_force[3:])
            force = sensor_item.process_force_for_show(force)
            sensor_item.set_3d_arrow(mesh_init[1:-1, 1:-1]/10, mesh_init[1:-1, 1:-1]/10 + force[1:-1, 1:-1], color[1:-1, 1:-1])
            
    def setDepth(self, sensor_idx, depth):
        """设置指定传感器的深度数据显示"""
        sensor_item = self.sensor_item_0 if sensor_idx == 0 else self.sensor_item_1
        sensor_item.set_depth(depth)
    
    def create2d(self, *args):
        self._view2d = self.View2d(*args, parent=self)
        return self._view2d

    def show(self):
        self.app.exec_()
        
    class View2d():
        def __init__(self, *args, parent):
            assert all(isinstance(arg, Sensor.OutputType) for arg in args), "All arguments must be of type {Sensor.OutputType}"
            self.winlist = dict()
            
            # 为两个传感器分别计算显示尺寸
            def calculate_display_size(sensor, max_height=400):
                try:
                    rectify_w, rectify_h = sensor.rectify_size
                    scale_factor = min(max_height / rectify_h, 1.0)
                    return int(rectify_w * scale_factor), int(rectify_h * scale_factor)
                except:
                    return 240, 420  # 默认值
            
            # 计算两个传感器的显示尺寸
            img_size_0 = calculate_display_size(parent.sensor_0)
            img_size_1 = calculate_display_size(parent.sensor_1)
            
            print(f"传感器0显示尺寸: {img_size_0[0]}x{img_size_0[1]}")
            print(f"传感器1显示尺寸: {img_size_1[0]}x{img_size_1[1]}")
            
            # 计算窗口大小 - 根据图像尺寸动态调整
            window_width = max(img_size_0[0] + img_size_1[0] + 600, 1600)  # 600为3D视图预留空间
            window_height = max(max(img_size_0[1], img_size_1[1]) * 3 + 200, 900)  # 3张图片垂直排列
            
            with tb.window("Dual Sensor View", size=(window_width, window_height)):
                with tb.group("view", horizontal=True, show=False):
                    # 传感器0组
                    with tb.group("Sensor 0", horizontal=False):
                        for i, name in enumerate(args[:len(args)//2]):
                            key = f"sensor_0_{name}"
                            self.winlist[key] = tb.add_image_view(
                                f"S0_{name.name}", 
                                None, 
                                img_size=img_size_0, 
                                img_format="bgr"
                            )
                    
                    # 传感器1组
                    with tb.group("Sensor 1", horizontal=False):
                        for i, name in enumerate(args[len(args)//2:]):
                            key = f"sensor_1_{name}"
                            self.winlist[key] = tb.add_image_view(
                                f"S1_{name.name}", 
                                None, 
                                img_size=img_size_1, 
                                img_format="bgr"
                            )
                    
                    # 3D视图
                    tb.add_widget("3d_view", parent)

        def setData(self, sensor_idx, name, img):
            key = f"sensor_{sensor_idx}_{name}"
            if key in self.winlist:
                if name == Sensor.OutputType.Depth:
                    import cv2
                    img = cv2.cvtColor(cm.jet(img), cv2.COLOR_RGB2BGR)
                    self.winlist[key].setData(img * 255)
                else:
                    self.winlist[key].setData(img)

def main():
    sensor_0 = Sensor.create(0)
    sensor_1 = Sensor.create(2)
    
    # 打印传感器信息，帮助调试画面比例
    print("=== 传感器配置信息 ===")
    print(f"传感器 0:")
    print(f"  - 类型: {sensor_0.sensor_type}")
    print(f"  - Rectify尺寸: {sensor_0.rectify_size}")
    print(f"  - Grid坐标尺寸: {sensor_0.grid_coord_size}")
    print(f"  - 推理尺寸: {sensor_0.infer_size}")
    
    print(f"传感器 1:")
    print(f"  - 类型: {sensor_1.sensor_type}")
    print(f"  - Rectify尺寸: {sensor_1.rectify_size}")
    print(f"  - Grid坐标尺寸: {sensor_1.grid_coord_size}")
    print(f"  - 推理尺寸: {sensor_1.infer_size}")
    print("========================")
    
    # 使用自定义的双传感器视图
    View = DualSensorView(sensor_0, sensor_1)
    
    # 创建2D视图显示两个传感器的数据
    View2d = View.create2d(
        Sensor.OutputType.Difference, Sensor.OutputType.Depth, Sensor.OutputType.Marker2D,  # 传感器0
        Sensor.OutputType.Difference, Sensor.OutputType.Depth, Sensor.OutputType.Marker2D   # 传感器1
    )

    def callback():
        """处理两个传感器的数据"""
        # 获取传感器0的数据
        force, res_force, mesh_init, src, diff, depth = sensor_0.selectSensorInfo(
            Sensor.OutputType.Force, 
            Sensor.OutputType.ForceResultant,
            Sensor.OutputType.Mesh3DInit,
            Sensor.OutputType.Rectify, 
            Sensor.OutputType.Difference, 
            Sensor.OutputType.Depth,
        )
        
        # 获取传感器1的数据
        force_1, res_force_1, mesh_init_1, src_1, diff_1, depth_1 = sensor_1.selectSensorInfo(
            Sensor.OutputType.Force, 
            Sensor.OutputType.ForceResultant,
            Sensor.OutputType.Mesh3DInit,
            Sensor.OutputType.Rectify, 
            Sensor.OutputType.Difference, 
            Sensor.OutputType.Depth,
        )
        
        # 生成标记图像
        marker_img = sensor_0.drawMarkerMove(src)
        marker_img_1 = sensor_1.drawMarkerMove(src_1)
        
        # 设置2D视图数据
        View2d.setData(0, Sensor.OutputType.Difference, diff)
        View2d.setData(0, Sensor.OutputType.Depth, depth)
        View2d.setData(0, Sensor.OutputType.Marker2D, marker_img)
        
        View2d.setData(1, Sensor.OutputType.Difference, diff_1)
        View2d.setData(1, Sensor.OutputType.Depth, depth_1)
        View2d.setData(1, Sensor.OutputType.Marker2D, marker_img_1)
        
        # 设置3D视图的力数据显示
        View.setForceFlow(0, force, res_force, mesh_init)
        View.setDepth(0, depth)
        View.setForceFlow(1, force_1/20, res_force_1/20, mesh_init_1)
        View.setDepth(1, depth_1)
        
        # 打印两个传感器的力数据信息
        global flag
        if flag == 1:
            print("=== 双传感器力数据对比 ===")
            print(f'传感器 0 - Force shape: {force.shape}, Max force: {abs(force).max():.6f}')
            print(f'传感器 0 - Resultant force: {res_force}')
            print(f'传感器 1 - Force shape: {force_1.shape}, Max force: {abs(force_1).max():.6f}')
            print(f'传感器 1 - Resultant force: {res_force_1}')
            print("==========================")
            flag = 0
        
        # 实时打印力数据对比
        max_force_0 = abs(force).max()
        max_force_1 = abs(force_1).max()
        if max_force_0 > 0.1 or max_force_1 > 0.1:
            print(f"实时力对比 - 传感器0: {max_force_0:.3f}N, 传感器1: {max_force_1:.3f}N")

    View.setCallback(callback)
    View.show()
    
    sensor_0.release()
    sensor_1.release()
    sys.exit()

if __name__ == '__main__':
    main()