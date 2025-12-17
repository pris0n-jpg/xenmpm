#!/usr/bin/env python3
"""
位移平台XYZ三轴完全独立控制工具
基于host_machine_chinese.py改编，适配3轴位移平台
支持：粗调、微调、ATI力传感器实时显示、急停机制

功能：
- X/Y/Z轴完全独立控制
- 方向选择 + 步数控制（粗调）
- 微调功能（脉冲数 × 重复次数 × 时间间隔）
- ATI力传感器实时显示
- 急停/解锁机制
- 串口自动连接

使用: python stage_control_xyz.py
"""

import sys
import serial
import serial.tools.list_ports
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTextEdit, QLineEdit, QGridLayout, QGroupBox
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont

# 尝试导入ATI传感器
ATI_AVAILABLE = False
try:
    from pyati.ati_sensor import ATISensor
    ATI_AVAILABLE = True
except ImportError:
    print("⚠️  pyati未安装，ATI传感器功能将不可用")


class StageControlXYZ(QWidget):
    """位移平台XYZ三轴独立控制界面"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("位移平台XYZ三轴独立控制工具")
        self.serial = None
        self.emergency_locked = False
        self.ati = None
        
        # 控制组件字典
        self.axisControls = {}
        self.adjControls = {}
        
        # 初始化界面
        self.init_ui()
        
        # 串口读取定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_serial)
        
        # ATI传感器定时器
        if ATI_AVAILABLE:
            self.ati_timer = QTimer()
            self.ati_timer.timeout.connect(self.update_ati_display)
        
        # 自动连接
        self.auto_connect()
        
        # 初始化ATI传感器
        if ATI_AVAILABLE:
            self.init_ati_sensor()

    def init_ui(self):
        """初始化用户界面"""
        mainLayout = QHBoxLayout()
        
        # ========== 左侧：控制区 ==========
        leftLayout = QVBoxLayout()
        
        # 串口连接区
        portLayout = QHBoxLayout()
        self.portCombo = QComboBox()
        self.refreshPorts()
        self.refreshBtn = QPushButton("刷新串口")
        self.connectBtn = QPushButton("连接")
        self.disconnectBtn = QPushButton("断开")
        
        self.refreshBtn.clicked.connect(self.refreshPorts)
        self.connectBtn.clicked.connect(self.connect_serial)
        self.disconnectBtn.clicked.connect(self.disconnect_serial)
        
        portLayout.addWidget(QLabel("串口:"))
        portLayout.addWidget(self.portCombo)
        portLayout.addWidget(self.refreshBtn)
        portLayout.addWidget(self.connectBtn)
        portLayout.addWidget(self.disconnectBtn)
        leftLayout.addLayout(portLayout)
        
        # 粗调控制区（X/Y/Z三轴）
        moveGroup = QGroupBox("粗调控制")
        moveGrid = QGridLayout()
        axes = ['X', 'Y', 'Z']
        
        for i, axis in enumerate(axes):
            # 轴标签
            moveGrid.addWidget(QLabel(f"{axis}轴方向:"), i, 0)
            
            # 方向选择
            dir_combo = QComboBox()
            if axis == 'X':
                dir_combo.addItem("向后运动 (-)", 0)
                dir_combo.addItem("向前运动 (+)", 1)
            elif axis == 'Y':
                dir_combo.addItem("向左运动 (-)", 0)
                dir_combo.addItem("向右运动 (+)", 1)
            elif axis == 'Z':
                dir_combo.addItem("向上运动 (-)", 0)
                dir_combo.addItem("向下运动 (+)", 1)
            moveGrid.addWidget(dir_combo, i, 1)
            
            # 步数输入
            moveGrid.addWidget(QLabel("步数:"), i, 2)
            steps_input = QLineEdit("100")
            steps_input.setFixedWidth(100)
            moveGrid.addWidget(steps_input, i, 3)
            
            # 发送按钮
            send_btn = QPushButton(f"发送{axis}轴")
            send_btn.setFixedHeight(50)
            send_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
            send_btn.clicked.connect(lambda _, a=axis: self.send_axis(a))
            moveGrid.addWidget(send_btn, i, 4)
            
            # 保存控件引用
            self.axisControls[axis] = (dir_combo, steps_input, send_btn)
        
        moveGroup.setLayout(moveGrid)
        leftLayout.addWidget(moveGroup)
        
        # 微调区（2x2布局）
        adjLayout = QGridLayout()
        adjLayout.addWidget(self._build_adjust_group('X', color="#d0e8ff"), 0, 0)
        adjLayout.addWidget(self._build_adjust_group('Y', color="#d0ffd0"), 0, 1)
        adjLayout.addWidget(self._build_adjust_group('Z', color="#fff5b0"), 1, 0, 1, 2)
        leftLayout.addLayout(adjLayout)
        
        mainLayout.addLayout(leftLayout, stretch=2)
        
        # ========== 右侧：状态显示区 ==========
        rightLayout = QVBoxLayout()
        
        # 急停区
        emergencyLayout = QHBoxLayout()
        self.stopBtn = QPushButton("急停")
        self.stopBtn.setStyleSheet(
            "background-color: red; color: white; font-weight: bold; font-size: 32px;"
        )
        self.stopBtn.setFixedSize(180, 90)
        self.stopBtn.clicked.connect(self.send_stop)
        
        self.unlockBtn = QPushButton("取消急停")
        self.unlockBtn.setStyleSheet(
            "background-color: green; color: white; font-weight: bold; font-size: 32px;"
        )
        self.unlockBtn.setFixedSize(180, 90)
        self.unlockBtn.setEnabled(False)
        self.unlockBtn.clicked.connect(self.unlock_controls)
        
        emergencyLayout.addWidget(self.stopBtn)
        emergencyLayout.addWidget(self.unlockBtn)
        emergencyLayout.addStretch()
        rightLayout.addLayout(emergencyLayout)
        
        # ATI传感器显示区
        if ATI_AVAILABLE:
            atiGroup = QGroupBox("ATI力传感器")
            atiLayout = QVBoxLayout()
            
            self.force_label = QLabel("Force (N):\nFx: 0.000\nFy: 0.000\nFz: 0.000")
            self.force_label.setStyleSheet(
                "font-size: 14px; font-family: Consolas; "
                "background-color: #1e1e1e; color: #00ff00; padding: 10px;"
            )
            atiLayout.addWidget(self.force_label)
            
            self.torque_label = QLabel("Torque (Nm):\nTx: 0.000\nTy: 0.000\nTz: 0.000")
            self.torque_label.setStyleSheet(
                "font-size: 14px; font-family: Consolas; "
                "background-color: #1e1e1e; color: #ffff00; padding: 10px;"
            )
            atiLayout.addWidget(self.torque_label)
            
            atiGroup.setLayout(atiLayout)
            rightLayout.addWidget(atiGroup)
        
        # 串口反馈区
        rightLayout.addWidget(QLabel("串口反馈:"))
        self.logOutput = QTextEdit()
        self.logOutput.setReadOnly(True)
        self.logOutput.setMinimumWidth(350)
        self.logOutput.setStyleSheet(
            "background-color: black; color: #00FF00; "
            "font-family: Consolas, monospace; font-size: 11pt;"
        )
        rightLayout.addWidget(self.logOutput)
        
        # 清屏按钮
        self.clearBtn = QPushButton("清屏")
        self.clearBtn.clicked.connect(lambda: self.logOutput.clear())
        rightLayout.addWidget(self.clearBtn)
        
        mainLayout.addLayout(rightLayout, stretch=1)
        self.setLayout(mainLayout)
        
        # 设置窗口大小
        self.resize(1200, 700)

    def _build_adjust_group(self, axis, color=None):
        """构建微调控制组"""
        group = QGroupBox(f"{axis}轴微调")
        if color:
            group.setStyleSheet(f"QGroupBox {{ background-color: {color}; }}")
        
        grid = QGridLayout()
        
        # 方向选择
        grid.addWidget(QLabel("方向:"), 0, 0)
        dir_combo = QComboBox()
        if axis == 'X':
            dir_combo.addItem("向后运动 (-)", 0)
            dir_combo.addItem("向前运动 (+)", 1)
        elif axis == 'Y':
            dir_combo.addItem("向左运动 (-)", 0)
            dir_combo.addItem("向右运动 (+)", 1)
        elif axis == 'Z':
            dir_combo.addItem("向上运动 (-)", 0)
            dir_combo.addItem("向下运动 (+)", 1)
        grid.addWidget(dir_combo, 0, 1)
        
        # 单次脉冲数
        grid.addWidget(QLabel("单次脉冲数:"), 1, 0)
        pulses_edit = QLineEdit("10")
        pulses_edit.setFixedWidth(100)
        grid.addWidget(pulses_edit, 1, 1)
        
        # 重复次数
        grid.addWidget(QLabel("重复次数:"), 2, 0)
        repeat_edit = QLineEdit("10")
        repeat_edit.setFixedWidth(100)
        grid.addWidget(repeat_edit, 2, 1)
        
        # 时间间隔
        grid.addWidget(QLabel("时间间隔(ms):"), 3, 0)
        interval_edit = QLineEdit("500")
        interval_edit.setFixedWidth(100)
        grid.addWidget(interval_edit, 3, 1)
        
        # 发送按钮
        send_btn = QPushButton(f"发送{axis}微调")
        send_btn.setFixedHeight(50)
        send_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        send_btn.clicked.connect(lambda _, a=axis: self.send_adjust(a))
        grid.addWidget(send_btn, 0, 2, 2, 1)
        
        # 统计信息
        total_pulses_label = QLabel("总微调脉冲数: 0")
        total_time_label = QLabel("总微调耗时: 0 s (0.00 min)")
        grid.addWidget(total_pulses_label, 4, 0, 1, 2)
        grid.addWidget(total_time_label, 5, 0, 1, 2)
        
        # 自动更新统计
        def update_info():
            try:
                pulses = int(pulses_edit.text())
                repeat = int(repeat_edit.text())
                interval = int(interval_edit.text())
                total_pulses = pulses * repeat
                total_ms = repeat * interval
                total_s = total_ms / 1000
                total_min = total_s / 60
                total_pulses_label.setText(f"总微调脉冲数: {total_pulses}")
                total_time_label.setText(f"总微调耗时: {total_s:.2f} s ({total_min:.2f} min)")
            except ValueError:
                total_pulses_label.setText("总微调脉冲数: -")
                total_time_label.setText("总微调耗时: -")
        
        pulses_edit.textChanged.connect(update_info)
        repeat_edit.textChanged.connect(update_info)
        interval_edit.textChanged.connect(update_info)
        update_info()
        
        group.setLayout(grid)
        self.adjControls[axis] = (dir_combo, pulses_edit, repeat_edit, interval_edit, send_btn)
        return group

    # ========== 串口通信 ==========
    
    def refreshPorts(self):
        """刷新可用串口列表"""
        self.portCombo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.portCombo.addItem(port.device)
    
    def auto_connect(self):
        """自动连接串口"""
        candidate_ports = ["COM10", "COM11", "COM9", "COM7", "/dev/ttyUSB0", "/dev/ttyACM0"]
        for port in candidate_ports:
            try:
                self.serial = serial.Serial(port, 115200, timeout=0.1)
                self.connectBtn.setText(f"已连接 {port}")
                self.connectBtn.setStyleSheet("background-color: green; color: white;")
                self.timer.start(100)
                self.logOutput.append(f"✅ 自动连接成功: {port}")
                return
            except Exception:
                continue
        self.logOutput.append("⚠️  未找到可用的串口，请手动连接")
    
    def connect_serial(self):
        """连接串口"""
        port_name = self.portCombo.currentText()
        try:
            self.serial = serial.Serial(port_name, 115200, timeout=0.1)
            self.connectBtn.setText(f"已连接 {port_name}")
            self.connectBtn.setStyleSheet("background-color: green; color: white;")
            self.logOutput.append(f"✅ 成功连接 {port_name}")
            self.timer.start(100)
        except Exception as e:
            self.logOutput.append(f"❌ 串口连接失败: {e}")
    
    def disconnect_serial(self):
        """断开串口"""
        if self.serial and self.serial.is_open:
            self.timer.stop()
            self.serial.close()
            self.connectBtn.setText("连接")
            self.connectBtn.setStyleSheet("")
            self.logOutput.append("✅ 串口已断开")
    
    def read_serial(self):
        """读取串口数据"""
        if self.serial and self.serial.in_waiting:
            try:
                data = self.serial.readline().decode(errors='ignore').strip()
                if data:
                    self.logOutput.append(f"📥 {data}")
            except Exception as e:
                self.logOutput.append(f"❌ 读取错误: {e}")

    # ========== 运动控制 ==========
    
    def send_axis(self, axis):
        """发送轴控制命令（粗调）"""
        if self.emergency_locked:
            self.logOutput.append("🔒 当前处于急停锁定状态")
            return
        if not self.serial or not self.serial.is_open:
            self.logOutput.append("❌ 串口未连接")
            return
        
        dir_combo, steps_input, _ = self.axisControls[axis]
        try:
            direction = dir_combo.currentData()
            steps = int(steps_input.text())
            if steps <= 0 or steps > 10000:
                raise ValueError("步数超出范围")
            
            # 命令格式: X 0 100 (轴 方向 步数)
            cmd = f"{axis} {direction} {steps}\n"
            self.serial.write(cmd.encode())
            self.logOutput.append(f"📤 发送指令: {cmd.strip()}")
        except ValueError:
            self.logOutput.append(f"❌ {axis}轴步数无效 (1~10000)")
    
    def send_adjust(self, axis):
        """发送微调命令"""
        if self.emergency_locked:
            self.logOutput.append("🔒 当前处于急停锁定状态")
            return
        if not self.serial or not self.serial.is_open:
            self.logOutput.append("❌ 串口未连接")
            return
        
        dir_combo, pulses_edit, repeat_edit, interval_edit, _ = self.adjControls[axis]
        try:
            direction = dir_combo.currentData()
            pulses = int(pulses_edit.text())
            repeat = int(repeat_edit.text())
            interval = int(interval_edit.text())
            total_steps = pulses * repeat
            
            if pulses <= 0 or repeat <= 0 or interval < 10 or total_steps > 10000:
                raise ValueError("参数超出范围")
            
            # 命令格式: XADJ 0 10 10 500 (轴ADJ 方向 单次脉冲 重复次数 间隔ms)
            cmd = f"{axis}ADJ {direction} {pulses} {repeat} {interval}\n"
            self.serial.write(cmd.encode())
            
            total_time_s = (repeat * interval) / 1000
            total_time_min = total_time_s / 60
            self.logOutput.append(
                f"📤 发送微调: {cmd.strip()}\n"
                f"   总步数={total_steps}, 总耗时={total_time_s:.2f}s ({total_time_min:.2f}min)"
            )
        except ValueError:
            self.logOutput.append("❌ 微调输入无效：脉冲>0，次数>0，总步数≤10000，间隔≥10")
    
    def send_stop(self):
        """发送急停命令"""
        if not self.serial or not self.serial.is_open:
            self.logOutput.append("❌ 串口未连接")
            return
        try:
            self.serial.write(b"STOP\n")
            self.logOutput.append("🛑 急停指令已发送")
            self.lock_controls()
        except Exception as e:
            self.logOutput.append(f"❌ 急停发送失败: {e}")
    
    def lock_controls(self):
        """锁定所有控制"""
        self.emergency_locked = True
        self.unlockBtn.setEnabled(True)
        
        # 锁定粗调控制
        for axis in ['X', 'Y', 'Z']:
            dir_combo, steps_input, send_btn = self.axisControls[axis]
            dir_combo.setEnabled(False)
            steps_input.setEnabled(False)
            send_btn.setEnabled(False)
        
        # 锁定微调控制
        for axis in ['X', 'Y', 'Z']:
            dir_combo, pulses_edit, repeat_edit, interval_edit, send_btn = self.adjControls[axis]
            dir_combo.setEnabled(False)
            pulses_edit.setEnabled(False)
            repeat_edit.setEnabled(False)
            interval_edit.setEnabled(False)
            send_btn.setEnabled(False)
        
        self.stopBtn.setEnabled(False)
        self.logOutput.append("🔒 所有控制已锁定")
    
    def unlock_controls(self):
        """解锁所有控制"""
        if not self.serial or not self.serial.is_open:
            self.logOutput.append("❌ 串口未连接")
            return
        
        try:
            self.serial.write(b"RESUME\n")
        except Exception:
            pass
        
        self.emergency_locked = False
        self.unlockBtn.setEnabled(False)
        
        # 解锁粗调控制
        for axis in ['X', 'Y', 'Z']:
            dir_combo, steps_input, send_btn = self.axisControls[axis]
            dir_combo.setEnabled(True)
            steps_input.setEnabled(True)
            send_btn.setEnabled(True)
        
        # 解锁微调控制
        for axis in ['X', 'Y', 'Z']:
            dir_combo, pulses_edit, repeat_edit, interval_edit, send_btn = self.adjControls[axis]
            dir_combo.setEnabled(True)
            pulses_edit.setEnabled(True)
            repeat_edit.setEnabled(True)
            interval_edit.setEnabled(True)
            send_btn.setEnabled(True)
        
        self.stopBtn.setEnabled(True)
        self.logOutput.append("✅ 已取消急停，控制已解锁")

    # ========== ATI传感器 ==========
    
    def init_ati_sensor(self):
        """初始化ATI传感器"""
        if not ATI_AVAILABLE:
            return
        try:
            self.ati = ATISensor(ip="192.168.1.10", filter_on=False)
            time.sleep(1)
            self.ati.tare()
            self.logOutput.append("✅ ATI传感器已连接并去皮")
            
            # 启动ATI更新定时器
            self.ati_timer.start(100)  # 100ms更新一次
        except Exception as e:
            self.logOutput.append(f"⚠️  ATI传感器连接失败: {e}")
            self.ati = None
    
    def update_ati_display(self):
        """更新ATI传感器显示"""
        if not hasattr(self, 'ati') or self.ati is None:
            return
        try:
            data = self.ati.data
            # 更新力显示
            self.force_label.setText(
                f"Force (N):\n"
                f"Fx: {data[0]:>7.3f}\n"
                f"Fy: {data[1]:>7.3f}\n"
                f"Fz: {data[2]:>7.3f}"
            )
            # 更新力矩显示
            self.torque_label.setText(
                f"Torque (Nm):\n"
                f"Tx: {data[3]:>7.3f}\n"
                f"Ty: {data[4]:>7.3f}\n"
                f"Tz: {data[5]:>7.3f}"
            )
        except Exception:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 全局字体设置
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)
    
    window = StageControlXYZ()
    window.show()
    
    sys.exit(app.exec_())