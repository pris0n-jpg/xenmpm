#!/usr/bin/env python3
"""
位移平台控制GUI - 数据采集调试工具
基于pyqt.py改编，适配数据采集用的4轴位移平台

功能：
- Z轴精确控制（A命令，用于压入）
- XY轴手动定位（B命令，用于传感器定位）  
- ATI力传感器实时显示
- 串口自动连接

使用: python stage_control_gui.py
"""
import sys
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QTextEdit
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import Qt


QApplication.setFont(QFont("Microsoft YaHei", 12))

class StageControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("位移平台控制界面 - 数据采集调试")
        self.resize(800, 600)   # 初始放大窗口

        # 默认全局字体
        font = QFont("Microsoft YaHei", 12)
        self.setFont(font)

        self.setWindowTitle("位移平台控制界面 - 数据采集调试")
        self.serial = None

        # === 串口选择区 ===
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.connect_btn = QPushButton("连接串口")
        self.connect_btn.clicked.connect(self.toggle_connection)

        # === 控制输入区 ===
        self.input_mm = QLineEdit()
        self.input_mm.setPlaceholderText("输入目标位置 (mm)")
        self.sendA_btn = QPushButton("发送 A 指令 (M0)")
        self.sendA_btn.clicked.connect(lambda: self.send_command('A'))
        self.sendB_btn = QPushButton("发送 B 指令 (M1)")
        self.sendB_btn.clicked.connect(lambda: self.send_command('B'))
        self.sendT_btn = QPushButton("发送 T 指令 (M0/M1 同步)")
        self.sendT_btn.clicked.connect(lambda: self.send_command('T'))

        # === 速度显示区 ===
        self.vel0_label = QLabel("M0速度: 0.00 rad/s")
        self.vel1_label = QLabel("M1速度: 0.00 rad/s")
        self.vel_btn = QPushButton("开始测速")
        self.vel_btn.clicked.connect(self.toggle_velocity_poll)

        # === 日志显示区 ===
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # === 布局 ===
        layout = QVBoxLayout()

        port_layout = QHBoxLayout()
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.connect_btn)
        layout.addLayout(port_layout)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_mm)
        input_layout.addWidget(self.sendA_btn)
        input_layout.addWidget(self.sendB_btn)
        input_layout.addWidget(self.sendT_btn)
        layout.addLayout(input_layout)

        layout.addWidget(self.vel0_label)
        layout.addWidget(self.vel1_label)
        layout.addWidget(self.vel_btn)
        layout.addWidget(self.log)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # === 定时器读取串口数据 ===
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_serial)

        # === 定时器：测速 ===
        self.vel_timer = QTimer()
        self.vel_timer.timeout.connect(self.poll_velocity)
        self.vel_toggle = True  # 轮流发 M0 / M1

        # === 前进/后退按钮区 ===

        self.m0_pos_mm = 0
        self.m1_pos_mm = 0

        self.m0_forward_btn = QPushButton("M0 前进 (+10 mm)")
        self.m0_forward_btn.clicked.connect(self.m0_forward)

        self.m0_backward_btn = QPushButton("M0 后退 (-10 mm)")
        self.m0_backward_btn.clicked.connect(self.m0_backward)

        self.m1_forward_btn = QPushButton("M1 前进 (+10 mm)")
        self.m1_forward_btn.clicked.connect(self.m1_forward)

        self.m1_backward_btn = QPushButton("M1 后退 (-10 mm)")
        self.m1_backward_btn.clicked.connect(self.m1_backward)

        move_layout = QHBoxLayout()
        move_layout.addWidget(self.m0_forward_btn)
        move_layout.addWidget(self.m0_backward_btn)
        move_layout.addWidget(self.m1_forward_btn)
        move_layout.addWidget(self.m1_backward_btn)
        layout.addLayout(move_layout)
        

        # === 一键归零 ===
        self.reset_btn = QPushButton("一键归零")
        self.reset_btn.clicked.connect(self.reset_positions)
        
        # 添加说明
        info_label = QLabel("💡 Z轴(A命令)用于压入，XY轴(B命令)用于定位")
        info_label.setStyleSheet("color: #0077b6; font-weight: bold;")
        layout.addWidget(info_label)
        layout.addWidget(self.reset_btn)

        # === 清屏按钮 ===
        self.clear_btn = QPushButton("清屏")
        self.clear_btn.clicked.connect(self.clear_log)
        layout.addWidget(self.clear_btn)


        #自动连接
        self.auto_connect()
        
        # 初始化ATI传感器
        if ATI_AVAILABLE:
            self.init_ati_sensor()

        # === 执行模式按钮 ===
        self.exec_mode = False
        self.exec_steps = []
        self.exec_index = 0

        self.exec_btn = QPushButton("执行模式")
        self.exec_btn.setCheckable(True)
        self.exec_btn.clicked.connect(self.toggle_exec_mode)

        self.next_btn = QPushButton("下一步")
        self.next_btn.setEnabled(False)  # 只有在执行模式下可用
        self.next_btn.clicked.connect(self.run_exec_step)

        exec_layout = QHBoxLayout()
        exec_layout.addWidget(self.exec_btn)
        exec_layout.addWidget(self.next_btn)
        layout.addLayout(exec_layout)

        #界面设置
        self.setStyleSheet("""

            QLineEdit {
                border: 2px solid #4facfe;
                border-radius: 8px;
                padding: 4px;
                background: white;
            }
            QTextEdit {
                background-color: black;
                color: #00FF00;
                border: 2px solid #4facfe;
                border-radius: 8px;
                padding: 6px;
                font-family: Consolas, monospace;
                font-size: 11pt;
            }

        """)

    def run_exec_step(self):
        if not self.exec_mode:
            return

        if self.in_init_phase:
            # 归零阶段
            if self.exec_index < len(self.exec_steps_init):
                step_cmd = self.exec_steps_init[self.exec_index]
                self.exec_index += 1
            else:
                # 归零结束，切换到循环阶段
                self.in_init_phase = False
                self.exec_index = 0
                self.run_exec_step()  # 马上进入下一步
                return
        else:
            # 循环阶段
            if self.exec_index >= len(self.exec_steps_loop):
                self.exec_index = 0  # 循环回到 A80
            step_cmd = self.exec_steps_loop[self.exec_index]
            self.exec_index += 1

        # 发送命令
        self.send_command_raw(step_cmd + "\r\n")
        self.log.append(f"执行: {step_cmd}")

        # 更新位置变量
        if step_cmd.startswith("A"):
            self.m0_pos_mm = int(step_cmd[1:])
        elif step_cmd.startswith("B"):
            self.m1_pos_mm = int(step_cmd[1:])


    def toggle_exec_mode(self, checked):
        if checked:
            self.exec_mode = True
            self.log.append("进入执行模式，点击“下一步”逐步执行")

            # 步骤 1：归零（只执行一次）
            self.reset_positions()

            # 步骤 2 以后：循环动作
            self.exec_steps_loop = [
                "A80", "A0", "B55",
                "A80", "A0", "B130",
                "A80", "A0", "B205",
                "A80", "A0", "B250",
                "A80", "A0", "B0"
            ]

            # 初始化索引和状态
            self.exec_index = 0
            self.in_init_phase = False   # 已经归零过了
            self.next_btn.setEnabled(True)

        else:
            self.exec_mode = False
            self.log.append("退出执行模式")
            self.next_btn.setEnabled(False)
            self.exec_index = 0
            self.in_init_phase = True



    def clear_log(self):
        self.log.clear()

    def auto_connect(self):
    # 候选端口
        candidate_ports = ["COM10", "COM11", "COM9", "COM7"]
        for port in candidate_ports:
            try:
                self.serial = serial.Serial(port, 115200, timeout=0.1)
                self.connect_btn.setText(f"已连接 {port}")
                self.timer.start(100)  # 100ms 轮询
                self.log.append(f"自动连接成功: {port}")
                return
            except Exception:
                continue
        self.log.append("未找到可用的自动连接端口")


    def reset_positions(self):
        self.m0_pos_mm = 0
        self.m1_pos_mm = 0
        self.send_command_raw("A0\r\n")
        QTimer.singleShot(10, lambda: self.send_command_raw("B0\r\n"))
        self.log.append("已归零: M0=0 mm, M1=0 mm")
        self.vel0_label.setText("M0速度: 0.00 rad/s")
        self.vel1_label.setText("M1速度: 0.00 rad/s")

    def m0_forward(self):
        if self.m0_pos_mm < 80:     # 最大 80 mm
            self.m0_pos_mm += 10
            if self.m0_pos_mm > 80:
                self.m0_pos_mm = 80
        self.send_command_raw(f"A{self.m0_pos_mm}\r\n")
        self.log.append(f"M0 前进 -> {self.m0_pos_mm} mm")

    def m0_backward(self):
        if self.m0_pos_mm > 0:      # 最小 0 mm
            self.m0_pos_mm -= 10
            if self.m0_pos_mm < 0:
                self.m0_pos_mm = 0
        self.send_command_raw(f"A{self.m0_pos_mm}\r\n")
        self.log.append(f"M0 后退 -> {self.m0_pos_mm} mm")

    def m1_forward(self):
        if self.m1_pos_mm < 250:    # 最大 250 mm
            self.m1_pos_mm += 10
            if self.m1_pos_mm > 250:
                self.m1_pos_mm = 250
        self.send_command_raw(f"B{self.m1_pos_mm}\r\n")
        self.log.append(f"M1 前进 -> {self.m1_pos_mm} mm")

    def m1_backward(self):
        if self.m1_pos_mm > 0:      # 最小 0 mm
            self.m1_pos_mm -= 10
            if self.m1_pos_mm < 0:
                self.m1_pos_mm = 0
        self.send_command_raw(f"B{self.m1_pos_mm}\r\n")
        self.log.append(f"M1 后退 -> {self.m1_pos_mm} mm")



    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.port_combo.addItem(p.device)

    def toggle_connection(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.connect_btn.setText("连接串口")
            self.timer.stop()
            self.vel_timer.stop()
            self.vel_btn.setText("开始测速")
        else:
            try:
                port = self.port_combo.currentText()
                self.serial = serial.Serial(port, 115200, timeout=0.1)
                self.connect_btn.setText("断开串口")
                self.timer.start(100)  # 100ms 轮询
            except Exception as e:
                self.log.append(f"串口连接失败: {e}")

    def send_command(self, cmd):
        if not (self.serial and self.serial.is_open):
            self.log.append("请先连接串口！")
            return
        try:
            val = float(self.input_mm.text())
            command = f"{cmd}{val:.2f}\r\n"
            self.serial.write(command.encode())
            self.log.append(f"发送: {command.strip()}")
        except ValueError:
            self.log.append("请输入正确的数字")

    def send_command_raw(self, cmd_str):
        if self.serial and self.serial.is_open:
            self.serial.write(cmd_str.encode())
        else:
            self.log.append("串口未连接")

    def toggle_velocity_poll(self):
        if self.vel_timer.isActive():
            self.vel_timer.stop()
            self.vel_btn.setText("开始测速")
        else:
            self.vel_timer.start(200)  # 每 200ms 交替发一次
            self.vel_btn.setText("停止测速")

    def poll_velocity(self):
        if self.vel_toggle:
            self.send_command_raw("MV\r\n")  # 请求 M0
        else:
            self.send_command_raw("NV\r\n")  # 请求 M1
        self.vel_toggle = not self.vel_toggle
    
    def init_ati_sensor(self):
        """初始化ATI传感器"""
        if not ATI_AVAILABLE:
            return
        try:
            self.ati = ATISensor(ip="192.168.1.10", filter_on=False)
            time.sleep(1)
            self.ati.tare()
            self.log.append("✅ ATI传感器已连接并去皮")
            
            # 启动ATI更新定时器
            self.ati_timer = QTimer()
            self.ati_timer.timeout.connect(self.update_ati_display)
            self.ati_timer.start(100)
        except Exception as e:
            self.log.append(f"⚠️ ATI传感器连接失败: {e}")
            self.ati = None
    
    def update_ati_display(self):
        """更新ATI传感器显示"""
        if not hasattr(self, 'ati') or self.ati is None:
            return
        try:
            data = self.ati.data
            # 在日志中显示力数据
            self.log.append(f"📊 ATI: Fx={data[0]:.3f} Fy={data[1]:.3f} Fz={data[2]:.3f} N")
        except:
            pass

    def read_serial(self):
        if self.serial and self.serial.in_waiting:
            data = self.serial.readline().decode(errors="ignore").strip()
            if data:
                if data.startswith("Vel0="):
                    try:
                        v0 = float(data.split("=")[1])
                        self.vel0_label.setText(f"M0速度: {v0:.2f} rad/s")
                    except:
                        pass
                elif data.startswith("Vel1="):
                    try:
                        v1 = float(data.split("=")[1])
                        self.vel1_label.setText(f"M1速度: {v1:.2f} rad/s")
                    except:
                        pass
                else:
                    # 检测 homing 完成信号
                    if "Platform homing finished." in data:
                        self.log.append("检测到平台归零完成，执行一键归零")
                        self.reset_positions()
                        return  # 已处理，不再打印
                    # 过滤无关调试信息
                    if "backoff applied" in data or "target=" in data:
                        return
                    self.log.append(f"接收: {data}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 12))  # 全局字体
    gui = StageControlGUI()
    gui.show()
    sys.exit(app.exec_())
