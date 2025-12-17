import sys
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QSizePolicy, QHBoxLayout, QLabel, QSpacerItem,
    QPushButton, QComboBox, QTextEdit, QLineEdit, QGridLayout, QGroupBox
)
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt


class StepperControl(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Four-axis stepper motor control host computer（X/Y/Z/θ）")
        self.serial = None
        self.emergency_locked = False

        self.axisControls = {}
        self.adjControls = {}

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_serial)

    # ---------------- UI ----------------
    def init_ui(self):
        mainLayout = QHBoxLayout()

        # 左侧：串口 + 控制
        leftLayout = QVBoxLayout()

        # 串口区
        portLayout = QHBoxLayout()
        self.portCombo = QComboBox()
        self.refreshPorts()
        self.refreshBtn = QPushButton("Refresh serial port")
        self.connectBtn = QPushButton("Connect")
        self.disconnectBtn = QPushButton("Disconnect")
        self.refreshBtn.clicked.connect(self.refreshPorts)
        self.connectBtn.clicked.connect(self.connect_serial)
        self.disconnectBtn.clicked.connect(self.disconnect_serial)
        portLayout.addWidget(QLabel("Serial:"))
        portLayout.addWidget(self.portCombo)
        portLayout.addWidget(self.refreshBtn)
        portLayout.addWidget(self.connectBtn)
        portLayout.addWidget(self.disconnectBtn)
        leftLayout.addLayout(portLayout)

        # 普通运动
        moveGrid = QGridLayout()
        combo_width = 400
        axes = ['X', 'Y', 'Z', 'T']

        for i, axis in enumerate(axes):
            # 轴标签
            moveGrid.addWidget(QLabel(f"{axis if axis != 'T' else 'θ'}-axis direction:"), i, 0)

            # 一行的水平布局
            hbox = QHBoxLayout()

            # 方向选择
            dir_combo = QComboBox()
            dir_combo.setFixedWidth(combo_width)
            if axis == 'X':
                dir_combo.addItem("Backward Movement", 0)
                dir_combo.addItem("Forward Movement", 1)

            elif axis == 'Y':
                dir_combo.addItem("Move Left", 0)
                dir_combo.addItem("Move Right", 1)
            elif axis == 'Z':
                dir_combo.addItem("Upward Movement", 0)
                dir_combo.addItem("Downward Movement", 1)
            elif axis == 'T':
                dir_combo.addItem("Counterclockwise Movement", 0)
                dir_combo.addItem("Clockwise Movement", 1)

            hbox.addWidget(dir_combo)

            # 水平间隔
            hbox.addSpacing(100)

            # Steps 标签和输入框
            hbox.addWidget(QLabel("Steps:"))
            steps_input = QLineEdit("1000")
            steps_input.setFixedWidth(600)
            hbox.addWidget(steps_input)

            # 水平间隔
            hbox.addSpacing(20)

            # 发送按钮
            send_btn = QPushButton(f"Send {axis if axis != 'T' else 'θ'} Axis")
            send_btn.setFixedSize(180, 50)
            send_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
            send_btn.clicked.connect(lambda _, a=axis: self.send_axis(a))
            hbox.addWidget(send_btn)

            # 将整行布局添加到 GridLayout
            moveGrid.addLayout(hbox, i, 1)

            # 保存控件引用
            self.axisControls[axis] = (dir_combo, steps_input)

        leftLayout.addLayout(moveGrid)

        # 微调区 2x2
        adjGrid = QGridLayout()
        adjGrid.addWidget(self._build_adjust_group('X', color="#d0e8ff"), 0, 0)
        adjGrid.addWidget(self._build_adjust_group('Y', color="#d0ffd0"), 0, 1)
        adjGrid.addWidget(self._build_adjust_group('Z', color="#fff5b0"), 1, 0)
        adjGrid.addWidget(self._build_adjust_group('T', display_name='θ', color="#e0e0e0"), 1, 1)
        leftLayout.addLayout(adjGrid)

        mainLayout.addLayout(leftLayout, stretch=2)

        # 右侧：急停 + 串口反馈
        rightLayout = QVBoxLayout()

        emergencyLayout = QHBoxLayout()
        self.stopBtn = QPushButton("Emergency Stop")
        self.stopBtn.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 32px;")
        self.stopBtn.setFixedSize(260, 120)
        self.stopBtn.clicked.connect(self.send_stop)

        self.unlockBtn = QPushButton("Cancel \nEmergency Stop")
        self.unlockBtn.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 32px;")
        self.unlockBtn.setFixedSize(260, 120)
        self.unlockBtn.setEnabled(False)
        self.unlockBtn.clicked.connect(self.unlock_controls)

        emergencyLayout.addWidget(self.stopBtn)
        emergencyLayout.addWidget(self.unlockBtn)
        emergencyLayout.addStretch()
        rightLayout.addLayout(emergencyLayout)

        rightLayout.addWidget(QLabel("Serial Port Feedback:"))
        self.logOutput = QTextEdit()
        self.logOutput.setReadOnly(True)
        self.logOutput.setMinimumWidth(300)
        rightLayout.addWidget(self.logOutput)

        mainLayout.addLayout(rightLayout, stretch=1)
        self.setLayout(mainLayout)

    def _build_adjust_group(self, axis, display_name=None, color=None):
        name = display_name if display_name else axis
        group = QGroupBox(f"{name}Axis Fine Adjustment")
        if color:
            group.setStyleSheet(f"QGroupBox {{ background-color: {color}; }}")
        grid = QGridLayout()

        # 方向
        grid.addWidget(QLabel("Direction:"), 0, 0)
        dir_combo = QComboBox()
        if axis == 'X':
            dir_combo.addItem("Backward Movement", 0)
            dir_combo.addItem("Forward Movement", 1)
        elif axis == 'Y':
            dir_combo.addItem("Move Left", 0)
            dir_combo.addItem("Move Right", 1)
        elif axis == 'Z':
            dir_combo.addItem("Upward Movement", 0)
            dir_combo.addItem("Downward Movement", 1)
        elif axis == 'T':
            dir_combo.addItem("Counterclockwise movement", 0)
            dir_combo.addItem("Clockwise movement", 1)
        grid.addWidget(dir_combo, 0, 1)

        # 输入参数
        grid.addWidget(QLabel("Single pulse number:"), 1, 0)
        pulses_edit = QLineEdit("10")
        grid.addWidget(pulses_edit, 1, 1)

        grid.addWidget(QLabel("Number of repetitions:"), 2, 0)
        repeat_edit = QLineEdit("10")
        grid.addWidget(repeat_edit, 2, 1)

        grid.addWidget(QLabel("Time interval (ms):"), 3, 0)
        interval_edit = QLineEdit("500")
        grid.addWidget(interval_edit, 3, 1)

        # 发送按钮
        send_btn = QPushButton(f"Send {name} \n Fine \n Adjustment")
        send_btn.setFixedSize(180, 100)
        send_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        send_btn.clicked.connect(lambda _, a=axis: self.send_adjust(a))
        grid.addWidget(send_btn, 0, 2, 2, 1)

        # 总脉冲数/总耗时
        total_pulses_label = QLabel("Total fine-tuning pulses: 0")
        total_time_label = QLabel("Total fine-tuning time: 0 s (0.00 min)")
        grid.addWidget(total_pulses_label, 4, 0, 1, 2)
        grid.addWidget(total_time_label, 5, 0, 1, 2)

        # 自动更新
        def update_info():
            try:
                pulses = int(pulses_edit.text())
                repeat = int(repeat_edit.text())
                interval = int(interval_edit.text())
                total_pulses = pulses * repeat
                total_ms = repeat * interval
                total_s = total_ms / 1000
                total_min = total_s / 60
                total_pulses_label.setText(f"Total fine-tuning pulses: {total_pulses}")
                total_time_label.setText(f"Total fine-tuning time: {total_s:.2f} s ({total_min:.2f} min)")
            except ValueError:
                total_pulses_label.setText("Total fine-tuning pulses: -")
                total_time_label.setText("Total fine-tuning time: -")

        pulses_edit.textChanged.connect(update_info)
        repeat_edit.textChanged.connect(update_info)
        interval_edit.textChanged.connect(update_info)
        update_info()

        group.setLayout(grid)
        self.adjControls[axis] = (dir_combo, pulses_edit, repeat_edit, interval_edit, send_btn)
        return group

    # ---------------- 串口逻辑 ----------------
    def refreshPorts(self):
        self.portCombo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.portCombo.addItem(port.device)

    def connect_serial(self):
        port_name = self.portCombo.currentText()
        try:
            self.serial = serial.Serial(port_name, 9600, timeout=0.1)
            self.logOutput.append(f"✅ Successful connection {port_name}")
            self.timer.start(100)
        except Exception as e:
            self.logOutput.append(f"❌ Serial port connection failed: {e}")

    def disconnect_serial(self):
        if self.serial and self.serial.is_open:
            self.timer.stop()
            self.serial.close()
            self.logOutput.append("✅ The serial port is disconnected")

    def send_axis(self, axis):
        if self.emergency_locked:
            self.logOutput.append("🔒 Currently in emergency stop lock state")
            return
        if not self.serial or not self.serial.is_open:
            self.logOutput.append("❌ Serial Port Not Connected")
            return

        dir_combo, steps_input = self.axisControls[axis]
        try:
            direction = dir_combo.currentData()
            steps = int(steps_input.text())
            if steps <= 0 or steps > 10000:
                raise ValueError
            cmd = f"{axis} {direction} {steps}\n"
            self.serial.write(cmd.encode())
            self.logOutput.append(f"📤 Sending Instructions: {cmd.strip()}")
        except ValueError:
            self.logOutput.append(f"❌ Invalid {axis}-Axis Step Number (1~10000)")

    def send_adjust(self, axis):
        if self.emergency_locked:
            self.logOutput.append("🔒 Currently in emergency stop lock state")
            return
        if not self.serial or not self.serial.is_open:
            self.logOutput.append("❌ Serial port not connected")
            return

        dir_combo, pulses_edit, repeat_edit, interval_edit, _ = self.adjControls[axis]
        try:
            direction = dir_combo.currentData()
            pulses = int(pulses_edit.text())
            repeat = int(repeat_edit.text())
            interval = int(interval_edit.text())
            total_steps = pulses * repeat
            if pulses <= 0 or repeat <= 0 or interval < 10 or total_steps > 10000:
                raise ValueError
            cmd = f"{axis}ADJ {direction} {pulses} {repeat} {interval}\n"
            self.serial.write(cmd.encode())
            total_time_s = (repeat * interval) / 1000
            total_time_min = total_time_s / 60
            self.logOutput.append(f"📤 Send fine-tuning: {cmd.strip()} (Total steps={total_steps}, Total time={total_time_s:.2f}s / {total_time_min:.2f}min)")
        except ValueError:
            self.logOutput.append("❌ Fine-tuning input is invalid: pulse>0, times>0, total steps ≤10000, interval ≥10")

    def send_stop(self):
        if not self.serial or not self.serial.is_open:
            self.logOutput.append("❌ Serial port not connected")
            return
        try:
            self.serial.write(b"STOP\n")
            self.logOutput.append("🛑 Emergency stop command has been sent")
            self.lock_controls()
        except Exception as e:
            self.logOutput.append(f"❌ Emergency stop sending failed: {e}")

    def lock_controls(self):
        self.emergency_locked = True
        self.unlockBtn.setEnabled(True)
        for axis in ['X', 'Y', 'Z', 'T']:
            dir_combo, steps_input = self.axisControls[axis]
            dir_combo.setEnabled(False)
            steps_input.setEnabled(False)
        for axis in ['X', 'Y', 'Z', 'T']:
            dir_combo, pulses_edit, repeat_edit, interval_edit, send_btn = self.adjControls[axis]
            dir_combo.setEnabled(False)
            pulses_edit.setEnabled(False)
            repeat_edit.setEnabled(False)
            interval_edit.setEnabled(False)
            send_btn.setEnabled(False)
        self.stopBtn.setEnabled(False)

    def unlock_controls(self):
        if not self.serial or not self.serial.is_open:
            self.logOutput.append("❌ Serial port not connected")
            return
        try:
            self.serial.write(b"RESUME\n")
        except Exception:
            pass
        self.emergency_locked = False
        self.unlockBtn.setEnabled(False)
        for axis in ['X', 'Y', 'Z', 'T']:
            dir_combo, steps_input = self.axisControls[axis]
            dir_combo.setEnabled(True)
            steps_input.setEnabled(True)
        for axis in ['X', 'Y', 'Z', 'T']:
            dir_combo, pulses_edit, repeat_edit, interval_edit, send_btn = self.adjControls[axis]
            dir_combo.setEnabled(True)
            pulses_edit.setEnabled(True)
            repeat_edit.setEnabled(True)
            interval_edit.setEnabled(True)
            send_btn.setEnabled(True)
        self.stopBtn.setEnabled(True)
        self.logOutput.append("✅ Emergency stop canceled")

    def read_serial(self):
        if self.serial and self.serial.in_waiting:
            try:
                data = self.serial.readline().decode(errors='ignore').strip()
                if data:
                    self.logOutput.append(f"📥 {data}")
            except Exception as e:
                self.logOutput.append(f"❌ Read Error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 全局字体放大
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)

    window = StepperControl()
    window.show()
    sys.exit(app.exec_())
