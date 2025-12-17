import sys
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTextEdit, QLineEdit, QGridLayout, QGroupBox
)
from PyQt5.QtCore import QTimer


class StepperControl(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("四轴步进电机控制上位机（X/Y/Z/θ）")
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

        # 普通运动
        moveGrid = QGridLayout()
        axes = ['X', 'Y', 'Z', 'T']
        for i, axis in enumerate(axes):
            moveGrid.addWidget(QLabel(f"{axis if axis != 'T' else 'θ'}轴方向:"), i, 0)

            dir_combo = QComboBox()
            if axis == 'X':
                dir_combo.addItem("向后运动", 0)
                dir_combo.addItem("向前运动", 1)
            elif axis == 'Y':
                dir_combo.addItem("向左运动", 0)
                dir_combo.addItem("向右运动", 1)
            elif axis == 'Z':
                dir_combo.addItem("向上运动", 0)
                dir_combo.addItem("向下运动", 1)
            elif axis == 'T':
                dir_combo.addItem("逆时针转", 0)
                dir_combo.addItem("顺时针转", 1)

            moveGrid.addWidget(dir_combo, i, 1)

            moveGrid.addWidget(QLabel("步数:"), i, 2)
            steps_input = QLineEdit("100")
            moveGrid.addWidget(steps_input, i, 3)

            send_btn = QPushButton(f"发送{axis if axis != 'T' else 'θ'}轴")
            send_btn.setFixedHeight(50)
            send_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
            send_btn.clicked.connect(lambda _, a=axis: self.send_axis(a))
            moveGrid.addWidget(send_btn, i, 4)

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
        self.stopBtn = QPushButton("急停")
        self.stopBtn.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 32px;")
        self.stopBtn.setFixedSize(180, 90)
        self.stopBtn.clicked.connect(self.send_stop)

        self.unlockBtn = QPushButton("取消急停")
        self.unlockBtn.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 32px;")
        self.unlockBtn.setFixedSize(180, 90)
        self.unlockBtn.setEnabled(False)
        self.unlockBtn.clicked.connect(self.unlock_controls)

        emergencyLayout.addWidget(self.stopBtn)
        emergencyLayout.addWidget(self.unlockBtn)
        emergencyLayout.addStretch()
        rightLayout.addLayout(emergencyLayout)

        rightLayout.addWidget(QLabel("串口反馈:"))
        self.logOutput = QTextEdit()
        self.logOutput.setReadOnly(True)
        self.logOutput.setMinimumWidth(300)
        rightLayout.addWidget(self.logOutput)

        mainLayout.addLayout(rightLayout, stretch=1)
        self.setLayout(mainLayout)

    def _build_adjust_group(self, axis, display_name=None, color=None):
        name = display_name if display_name else axis
        group = QGroupBox(f"{name}轴微调")
        if color:
            group.setStyleSheet(f"QGroupBox {{ background-color: {color}; }}")
        grid = QGridLayout()

        # 方向
        grid.addWidget(QLabel("方向:"), 0, 0)
        dir_combo = QComboBox()
        if axis == 'X':
            dir_combo.addItem("向后运动", 0)
            dir_combo.addItem("向前运动", 1)
        elif axis == 'Y':
            dir_combo.addItem("向左运动", 0)
            dir_combo.addItem("向右运动", 1)
        elif axis == 'Z':
            dir_combo.addItem("向上运动", 0)
            dir_combo.addItem("向下运动", 1)
        elif axis == 'T':
            dir_combo.addItem("逆时针转", 0)
            dir_combo.addItem("顺时针转", 1)
        grid.addWidget(dir_combo, 0, 1)

        # 输入参数
        grid.addWidget(QLabel("单次脉冲数:"), 1, 0)
        pulses_edit = QLineEdit("10")
        grid.addWidget(pulses_edit, 1, 1)

        grid.addWidget(QLabel("重复次数:"), 2, 0)
        repeat_edit = QLineEdit("10")
        grid.addWidget(repeat_edit, 2, 1)

        grid.addWidget(QLabel("时间间隔(ms):"), 3, 0)
        interval_edit = QLineEdit("500")
        grid.addWidget(interval_edit, 3, 1)

        # 发送按钮
        send_btn = QPushButton(f"发送{name}微调")
        send_btn.setFixedHeight(50)
        send_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        send_btn.clicked.connect(lambda _, a=axis: self.send_adjust(a))
        grid.addWidget(send_btn, 0, 2, 2, 1)

        # 总脉冲数/总耗时
        total_pulses_label = QLabel("总微调脉冲数: 0")
        total_time_label = QLabel("总微调耗时: 0 s (0.00 min)")
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
            self.logOutput.append(f"✅ 成功连接 {port_name}")
            self.timer.start(100)
        except Exception as e:
            self.logOutput.append(f"❌ 串口连接失败: {e}")

    def disconnect_serial(self):
        if self.serial and self.serial.is_open:
            self.timer.stop()
            self.serial.close()
            self.logOutput.append("✅ 串口已断开")

    def send_axis(self, axis):
        if self.emergency_locked:
            self.logOutput.append("🔒 当前处于急停锁定状态")
            return
        if not self.serial or not self.serial.is_open:
            self.logOutput.append("❌ 串口未连接")
            return

        dir_combo, steps_input = self.axisControls[axis]
        try:
            direction = dir_combo.currentData()
            steps = int(steps_input.text())
            if steps <= 0 or steps > 10000:
                raise ValueError
            cmd = f"{axis} {direction} {steps}\n"
            self.serial.write(cmd.encode())
            self.logOutput.append(f"📤 发送指令: {cmd.strip()}")
        except ValueError:
            self.logOutput.append(f"❌ {axis}轴步数无效 (1~10000)")

    def send_adjust(self, axis):
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
                raise ValueError
            cmd = f"{axis}ADJ {direction} {pulses} {repeat} {interval}\n"
            self.serial.write(cmd.encode())
            total_time_s = (repeat * interval) / 1000
            total_time_min = total_time_s / 60
            self.logOutput.append(f"📤 发送微调: {cmd.strip()} (总步数={total_steps}, 总耗时={total_time_s:.2f}s / {total_time_min:.2f}min)")
        except ValueError:
            self.logOutput.append("❌ 微调输入无效：脉冲>0，次数>0，总步数≤10000，间隔≥10")

    def send_stop(self):
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
            self.logOutput.append("❌ 串口未连接")
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
        self.logOutput.append("✅ 已取消急停")

    def read_serial(self):
        if self.serial and self.serial.in_waiting:
            try:
                data = self.serial.readline().decode(errors='ignore').strip()
                if data:
                    self.logOutput.append(f"📥 {data}")
            except Exception as e:
                self.logOutput.append(f"❌ 读取错误: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 全局字体放大
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)

    window = StepperControl()
    window.show()
    sys.exit(app.exec_())
