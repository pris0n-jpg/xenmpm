from contextlib import contextmanager
from typing import List, Union, Dict, Callable, Type, Tuple, Sequence
from qtpy.QtCore import Qt, QPoint, QSize, QEvent
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtGui import QFocusEvent, QMouseEvent
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
import numpy as np

QtWidgets.QApplication.setStyle('Fusion')


Number = Union[int, float]
NumberTuple = Tuple[Number]

def create_layout(
    parent,
    horizontal : bool = True,
    widgets : Sequence = None,
    stretchs : Sequence = None,
    content_margins : tuple = (0, 0, 0, 0),
    spacing : int = 0,
):
    """创建布局"""
    widgets = widgets if widgets is not None else []
    stretchs = stretchs if stretchs is not None else []

    layout = QtWidgets.QHBoxLayout(parent) if horizontal else QtWidgets.QVBoxLayout(parent)
    layout.setContentsMargins(*content_margins)
    layout.setSpacing(spacing)
    for i, widget in enumerate(widgets):
        if i > len(stretchs) - 1:
            layout.addWidget(widget)
        else:
            layout.addWidget(widget, stretchs[i])
    return layout


class CollapseTitleBar(QtWidgets.QFrame):
    """折叠标题栏"""
    toggleCollapsed = QtCore.Signal(bool)

    def __init__(self, parent:QWidget, closeable=True):
        super().__init__(parent)
        self.is_collapsed = False

        sizeFixedPolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Fixed)
        self.setSizePolicy(sizeFixedPolicy)
        self.setStyleSheet("background-color: #0f5687; color: white;")

        self.collapse_button = QtWidgets.QPushButton("▾", self)
        self.collapse_button.setFixedSize(QSize(20, 20))
        self.collapse_button.setStyleSheet(
            "QPushButton { border-radius: 10px; }"
            "QPushButton:hover { background-color: #288ad4; }"
        )
        self.label = QtWidgets.QLabel(self)
        self.label.setMinimumSize(QSize(0, 25))

        if closeable:
            self.close_button = QtWidgets.QPushButton("×", self)
            self.close_button.setFixedSize(QSize(25, 25))
            self.close_button.setStyleSheet(
                "QPushButton { border: 0px; }"
                "QPushButton:hover { background-color: #288ad4; }"
            )
            self.close_button.clicked.connect(parent.close)
            self.hbox = create_layout(self, True,
                                    [self.collapse_button, self.label, self.close_button],
                                    [1, 1, 1], content_margins=(5, 0, 0, 0), spacing=5)

        else:
            self.hbox = create_layout(self, True,
                                    [self.collapse_button, self.label],
                                    [1, 2], content_margins=(5, 0, 0, 0), spacing=5)

        self.collapse_button.clicked.connect(self.on_collapse)

    def setColor(self, bk_color, text_color):
        self.setStyleSheet(f"background-color: {bk_color}; color: {text_color};")

    def setLabel(self, label):
        self.label.setText(label)

    def on_collapse(self, value):
        self.is_collapsed = not self.is_collapsed
        self.collapse_button.setText("▸" if self.is_collapsed else "▾")
        self.toggleCollapsed.emit(self.is_collapsed)


class ToolItem():

    def set_label(self, label:str):
        self.__label = label

    def get_label(self) -> str:
        return self.__label

    @property
    def value(self):
        raise NotImplementedError

    @value.setter
    def value(self, val):
        raise NotImplementedError


class ToolContainer():

    def add_item(self, item: Union[ToolItem, QWidget]):
        raise NotImplementedError

    def get_layout(self):
        raise NotImplementedError


class ToolSplitter(QtWidgets.QWidget, ToolContainer):
    """不带边框的group"""

    def __init__(self, horizontal=True, spacing=5):
        super().__init__()
        self.container_box = QtWidgets.QHBoxLayout(self) if horizontal else QVBoxLayout(self)
        self.container_box.setContentsMargins(0, 0, 0, 0)
        self.container_box.setSpacing(spacing)

        self.splitter = QtWidgets.QSplitter(self)
        self.splitter.setStyleSheet("""
    QSplitter::handle {
        background-color: #e0e0e0;
        border: 1px solid #c0c0c0;
        width: 3px; /* 设置分割条的宽度 */
    }
    QSplitter::handle:horizontal {
        margin-left: 12px; /* 左侧边距 */
        margin-right: 12px; /* 右侧边距 */
        margin-top: 12px;
        margin-bottom: 12px;
    }
""")
        self.splitter.setHandleWidth(spacing)
        self.splitter.setChildrenCollapsible(False)

        self.container_box.addWidget(self.splitter)
        self.setMinimumWidth(200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def add_item(self, item: Union[ToolItem, QWidget]):
        return self.splitter.addWidget(item)

    def get_layout(self):
        return self.container_box

    def set_stretchs(self, stretchs: Sequence[int]):
        for i, stretch in enumerate(stretchs):
            self.splitter.setStretchFactor(i, stretch)


class ToolGroup(QtWidgets.QWidget, ToolContainer):
    """不带边框的group"""

    def __init__(self, horizontal=True, spacing=5):
        super().__init__()
        self.container_box = QtWidgets.QHBoxLayout(self) if horizontal else QVBoxLayout(self)
        self.container_box.setContentsMargins(0, 0, 0, 0)
        self.container_box.setSpacing(spacing)
        self.setMinimumWidth(200)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

    def add_item(self, item: Union[ToolItem, QWidget]):
        return self.container_box.addWidget(item)

    def get_layout(self):
        return self.container_box


class ToolGroupBox(QtWidgets.QGroupBox, ToolContainer):

    def __init__(self, label:str="", horizontal=True, margins:Sequence[int]=(5,15,5,15), spacing=5):
        super().__init__()
        self.setTitle(label)
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #0f5687;  /* 设置边框样式和颜色 */
                border-radius: 5px;  /* 设置边框圆角 */
                background-color: #f2f2f2;  /* 设置背景颜色 */
                margin-top: 3ex;  /* 上边距 */
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left; /* 设置标题位置 */
                padding: 0 0px;  /* 设置标题的内边距 */
                color:  #0f5687;  /* 设置标题的颜色 */
            }
        """)
        self.container_box = QtWidgets.QHBoxLayout(self) if horizontal else QVBoxLayout(self)
        self.container_box.setAlignment(Qt.AlignLeft)
        self.container_box.setContentsMargins(*margins)
        self.container_box.setSpacing(spacing)
        self.setMinimumWidth(200)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setSizePolicy(sizePolicy)

    def add_item(self, item: Union[ToolItem, QWidget]):
        return self.container_box.addWidget(item)

    def get_layout(self):
        return self.container_box


class ToolCollapsibeGroup(QtWidgets.QGroupBox, ToolContainer):

    def __init__(
        self,
        label: str="",
        horizontal=True,
        margins:Sequence[int]=(5,15,5,15),
        spacing=5,
        collapsed=False,
        window: 'ToolWindow'=None
    ):
        super().__init__()
        self.setupUi(label, horizontal, margins, spacing)
        # toggleCollapsed signal
        self.title_bar.toggleCollapsed.connect(self.on_collapse)
        self._win = window  # 为了在折叠的时候调用 adjustSize, 在 windows 系统需要用 QTimer.singleShot 延时调用
        if collapsed:
            self.title_bar.collapse_button.click()

    def setupUi(self, label, horizontal, margins, spacing):
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.setMinimumWidth(200)

        # 标题栏
        self.title_bar = CollapseTitleBar(self, closeable=False)
        self.title_bar.setLabel(label)
        self.title_bar.setColor("#c0c0c0", "#0f5687")

        # 容器
        self.container = QtWidgets.QWidget()
        self.container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.container_box = QHBoxLayout(self.container) if horizontal else QVBoxLayout(self.container)
        self.container_box.setAlignment(Qt.AlignTop)
        self.container_box.setContentsMargins(*margins)
        self.container_box.setSpacing(spacing)

        # 创建主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.container)

    def add_item(self, item: Union[ToolItem, QWidget]):
        return self.container_box.addWidget(item)

    def get_layout(self):
        return self.container_box

    def on_collapse(self, is_collapsed):
        self.container.setVisible(not is_collapsed)
        QtCore.QTimer.singleShot(50, self._win.adjustSize)

class ToolWindowFrameless(QtWidgets.QWidget, ToolContainer):

    def __init__(self, parent=None, label="ToolWindowFrameless", spacing=5):
        super().__init__(parent)
        self.setupUi()
        # 设置窗口样式为无边框
        if parent:
            self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint)

        self.title_bar.setLabel(label)
        self.container_box.setSpacing(spacing)

        # 用于保存鼠标点击位置的变量
        self.drag_position = QPoint()
        self.movable = False

        # signals
        self.title_bar.toggleCollapsed.connect(self.on_toggleCollapsed)

    def setupUi(self):
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.setMinimumWidth(220)

        # 创建一个自定义标题栏
        self.title_bar = CollapseTitleBar(self)

        # 组件容器
        self.container = QtWidgets.QFrame()
        self.container.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                     QtWidgets.QSizePolicy.Minimum)
        self.container_box = QVBoxLayout(self.container)  # 创建容器的布局
        self.container_box.setAlignment(Qt.AlignTop)

        # 创建主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.main_layout.setAlignment(Qt.AlignTop)  # 设置布局对齐方式为顶部对齐
        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.container)

        # 添加一个QSizeGrip到布局的右下角
        self.size_grip = QtWidgets.QSizeGrip(self)
        self.main_layout.addWidget(self.size_grip, 0, Qt.AlignBottom | Qt.AlignRight)
        self.setMaximumHeight(25)

    def on_toggleCollapsed(self, is_collapsed):
        # 折叠/展开窗口
        if is_collapsed:
            self.container.setVisible(False)
            self.size_grip.setVisible(False)
            self.adjustSize()
        else:
            self.container.setVisible(True)
            self.size_grip.setVisible(True)

    def adjustSize(self):
        self.setFixedWidth(self.width())  # 保持折叠后宽度不变
        super().adjustSize()
        self.setMinimumWidth(220)       # 恢复宽度可变
        self.setMaximumWidth(5000)
        self.size_grip.update()
        return

    def mousePressEvent(self, event):
        self.setFocus()  # 设置窗口为焦点, 实现点击窗口任意位置, 取消子组件的焦点的效果
        # 保存鼠标点击位置
        self.movable = False
        if self.title_bar.hbox.geometry().contains(event.pos()):
            self.movable = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        # 移动窗口位置
        if event.buttons() == Qt.LeftButton and self.movable:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, a0) -> None:
        self.movable = False
        return super().mouseReleaseEvent(a0)

    def add_item(self, Item: ToolItem):
        self.container_box.addWidget(Item)

    def get_layout(self):
        return self.container_box


class ToolWindow(QtWidgets.QMainWindow, ToolContainer):

    def __init__(self, parent=None, label="ToolWindow", spacing=5):
        super().__init__(parent)
        self.setupUi()
        if parent is not None:
            self.setWindowFlags(Qt.Tool)
        self.setWindowTitle(label)
        self.container_box.setSpacing(spacing)

    def setupUi(self):
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumWidth(220)

        # 组件容器
        self.container = QtWidgets.QFrame()
        self.container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.container_box = QVBoxLayout(self.container)  # 创建容器的布局
        self.container_box.setAlignment(Qt.AlignTop)

        # 创建主布局
        # self.main_layout = QVBoxLayout(self)
        # self.main_layout.setContentsMargins(0, 0, 0, 0)
        # self.main_layout.setSpacing(0)
        # self.main_layout.setAlignment(Qt.AlignTop)  # 设置布局对齐方式为顶部对齐
        # self.main_layout.addWidget(self.container)

        self.setCentralWidget(self.container)

        # 添加一个QSizeGrip到布局的右下角
        # self.size_grip = QtWidgets.QSizeGrip(self)
        # self.main_layout.addWidget(self.size_grip, 0, Qt.AlignBottom | Qt.AlignRight)

    def add_item(self, Item: ToolItem):
        self.container_box.addWidget(Item)

    def get_layout(self):
        return self.container_box

    def mousePressEvent(self, event):
        self.setFocus()  # 设置窗口为焦点, 实现点击窗口任意位置, 取消子组件的焦点的效果
        super().mousePressEvent(event)

    def __get_menu_by_title(self, title: str):
        for menu in self.menuBar().findChildren(QtWidgets.QMenu):
            if menu.title() == title:
                return menu
        return None

    def add_menu(self, label: str, action_label: str, callback: Callable=None):
        menu = self.__get_menu_by_title(label)
        if menu is None:
            menu = self.menuBar().addMenu(label)

        action = QtWidgets.QAction(action_label, self)
        menu.addAction(action)
        if callback is not None:
            action.triggered.connect(callback)


class ButtonItem(QPushButton, ToolItem):

    def __init__(self, label="Button", value=False, checkable=False, callback: Callable=None):
        super().__init__(label)
        self.set_label(label)
        self.setMaximumWidth(600)
        if checkable:
            self.setCheckable(True)
            self.setChecked(value)
        if callback is not None:
            self.clicked.connect(callback)

    @property
    def value(self):
        return self.isChecked()

    @value.setter
    def value(self, val):
        self.setChecked(val)


class CheckBoxItem(QtWidgets.QCheckBox, ToolItem):

    def __init__(self, label, value=False, callback: Callable=None):
        super().__init__(label, None)
        self.set_label(label)
        self.setMaximumWidth(300)
        self.setChecked(value)
        if callback is not None:
            self.clicked.connect(callback)

    @property
    def value(self):
        return self.isChecked()

    @value.setter
    def value(self, val):
        self.setChecked(val)


class CheckListItem(QtWidgets.QFrame, ToolItem):
    """互斥 checkboxes"""
    sigClicked = QtCore.Signal(object)

    def __init__(self, label:str, items: Sequence[str], value:Union[int, Sequence[bool]]=None,
                 horizontal=True, exclusive=True, callback: Callable=None):
        """value=None: 全部不选中, value=int: 选中第value个, value=Sequence[bool]: 选中对应的checkbox"""
        super().__init__()
        self.set_label(label)
        # 设置边框颜色和底色
        self.setStyleSheet("QFrame{border:1px solid #aaaaaa; border-radius: 3px; background-color: #ffffff;}")
        self.exclusive = exclusive
        self._items = tuple(items)
        if horizontal:
            self.box = QtWidgets.QHBoxLayout(self)
        else:
            self.box = QtWidgets.QVBoxLayout(self)

        self.box.setContentsMargins(10, 5, 0, 5)
        self.box_group = QtWidgets.QButtonGroup(self)
        for i, item in enumerate(items):
            checkbox = QtWidgets.QRadioButton(str(item), self)
            checkbox.setChecked(False)
            self.box.addWidget(checkbox)
            self.box_group.addButton(checkbox, i)
        # 互斥
        self.box_group.setExclusive(exclusive)

        # 当状态发生变化时, 发射信号
        self.box_group.idToggled.connect(self._on_toggled)
        if callback is not None:
            self.sigClicked.connect(callback)

        self.value = value

    @property
    def value(self):
        if self.exclusive:
            return self.box_group.checkedId()
        else:
            ret = []
            for i, button in enumerate(self.box_group.buttons()):
                if button.isChecked():
                    ret.append(i)
            return ret

    @value.setter
    def value(self, val: Union[int, Sequence[bool]]):
        if val is None:
            return
        elif isinstance(val, int):
            self.box_group.button(val).setChecked(True)
        else:
            for v, button in zip(val, self.box_group.buttons()):
                button.setChecked(v)

    @property
    def items(self):
        return self._items

    def _on_toggled(self, button_id, val):
        button = self.box_group.button(button_id)
        return self.sigClicked.emit((button.text(), button.isChecked()))


class ComboItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.Signal(str)

    def __init__(self, label:str, items: Sequence[str], value: int=0, callback: Callable=None):
        super().__init__()
        self.set_label(label)
        self.setMaximumWidth(600)
        self.combo = QtWidgets.QComboBox(self)
        self.name_label = QtWidgets.QLabel(label, self)
        self.combo.addItems(items)
        self.combo.setCurrentIndex(value)
        self.box = create_layout(self, True, [self.combo, self.name_label], [5, 2], spacing=10)
        # 信号/槽
        self.combo.currentTextChanged.connect(self._on_changed)
        if callback:
            self.sigChanged.connect(callback)

    def _on_changed(self, val):
        return self.sigChanged.emit(self.value)

    @property
    def value(self) -> str:
        return self.combo.currentText()

    @value.setter
    def value(self, val):
        self.combo.setCurrentText(val)

    def updateItems(self, items):
        self.combo.clear()
        self.combo.addItems(items)

class TextEditorItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.Signal(object)

    def __init__(self, label:str, value:str, editable=True, callback: Callable=None):
        super().__init__()
        self.set_label(label)
        self.setMaximumWidth(600)
        self.name_label = QtWidgets.QLabel(label, self)
        self.text_editor = QtWidgets.QLineEdit(value, self)
        if not editable:
            self.text_editor.setFocusPolicy(Qt.NoFocus)
        self.box = create_layout(self, True, [self.text_editor, self.name_label], [5, 2], spacing=10)
        # 信号/槽
        self.text_editor.editingFinished.connect(self._on_changed)
        if callback is not None:
            self.sigChanged.connect(callback)

    @property
    def value(self):
        return self.text_editor.text()

    @value.setter
    def value(self, val):
        self.text_editor.setText(val)

    def _on_changed(self, val):
        return self.sigChanged.emit(self.value)


class SliderItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.Signal(object)
    def __init__(self, label:str, value, min_val, max_val, step, decimals=0, callback: Callable=None):
        """decimals: 小数位数"""
        super().__init__()
        self.set_label(label)
        self.step = step
        self.decimals = decimals
        value = max(min_val, min(value, max_val))
        l_steps = int((value - min_val) / step)
        r_steps = int((max_val - value) / step)
        self.min_val = value - l_steps * step
        self.steps = l_steps + r_steps

        self.name_label = QtWidgets.QLabel(label, self)

        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, self.steps+1)

        self.spinbox = QtWidgets.QDoubleSpinBox(self)
        self.spinbox.setRange(self.min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setDecimals(decimals)

        self.box = create_layout(self, True,
                                 [self.slider, self.spinbox, self.name_label],
                                 [9, 1, 4], spacing=10)

        self.spinbox.setValue(value)
        self.slider.setValue(int((value - self.min_val) / self.step))

        self.slider.valueChanged.connect(self._on_changed)
        self.spinbox.valueChanged.connect(self._on_changed)
        if callback is not None:
            self.sigChanged.connect(callback)

    @property
    def value(self):
        val = self.spinbox.value()
        return int(val) if self.decimals == 0 else val

    @value.setter
    def value(self, val):
        self.spinbox.setValue(val)

    def _on_changed(self, value):
        if isinstance(self.sender(), QtWidgets.QSlider):
            val = self.min_val + self.slider.value() * self.step
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(val)
            self.spinbox.blockSignals(False)
        else:
            val = self.spinbox.value()
            self.slider.blockSignals(True)
            self.slider.setValue(int((val - self.min_val) / self.step))
            self.slider.blockSignals(False)

        self.sigChanged.emit(self.value)


class ArrayTypeItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.Signal(object)

    def __init__(self, label:str, value:NumberTuple, type:Union[Type[int], Type[float]],
                 editable=True, format: Union[str, Tuple[str]]=None, callback: Callable=None,
                 horizontal=True, show_label=True):
        super().__init__()
        self.set_label(label)
        self._value = list(value)
        self.length = len(value)
        self.type = type
        assert self.length > 0, "Array length must be greater than 0"

        # format
        if format is None:
            format = "%d" if type == int else "%.2f"
        if isinstance(format, str):
            self.format = [format] * self.length
        elif isinstance(format, (list, tuple)):
            assert len(format) == self.length, "Format length must be equal to value length"
            self.format = format
        else:
            raise TypeError("Format must be str, list or tuple")

        self.inputs_frame = QtWidgets.QFrame(self)
        self.inputs_layout = QtWidgets.QHBoxLayout(self.inputs_frame) if horizontal \
            else QtWidgets.QVBoxLayout(self.inputs_frame)
        self.inputs_layout.setContentsMargins(0, 0, 0, 0)

        self.inputs = []
        validator = QtGui.QIntValidator() if type == int else QtGui.QDoubleValidator()
        for i in range(self.length):
            input = QtWidgets.QLineEdit(self.format[i] % self.type(value[i]), self.inputs_frame)
            input.setMinimumWidth(10)
            input.setValidator(validator)
            input.setReadOnly(not editable)
            input.editingFinished.connect(self._on_changed)
            input.setObjectName(str(i))  # 设置objectName, 用于区分信号来源
            input.setAlignment(Qt.AlignCenter)

            self.inputs.append(input)
            self.inputs_layout.addWidget(input, 1)

        if show_label:
            self.box = create_layout(self, True, [self.inputs_frame, QtWidgets.QLabel(label, self)], [5, 2], spacing=10)
        else:
            self.box = create_layout(self, True, [self.inputs_frame], [5], spacing=10)

        if callback is not None:
            self.sigChanged.connect(callback)

        # style
        self.setStyleSheet(
        """ QLineEdit {background-color: #f7f7f7; border: 1px solid #aaaaaa; border-radius: 3px; }
            QLineEdit:hover { background-color: #dfe8ed; }
        """)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        for i in range(self.length):
            if self._value[i] != val[i]:
                self._value[i] = val[i]
                self.inputs[i].setText(self.format[i] % val[i])
        self.sigChanged.emit(self._value)

    def _on_changed(self):
        id = int(self.sender().objectName())
        val = self.type(self.inputs[id].text())
        if self._value[id] != val:
            self._value[id] = val
            self.sigChanged.emit(self._value)


class DragValue(QtWidgets.QLineEdit):

    sigValueChanged = QtCore.Signal(object)

    def __init__(self, value, min_val, max_val, step, decimals=2, format: str=None,
                 parent=None):
        super().__init__(parent)
        self.decimal_format = "%." + str(decimals) + "f"
        if format is None:
            self.format = self.decimal_format
        else:
            self.format = format
        self.drag_position = QPoint()  # 记录鼠标按下的位置
        self.pressed = False

        self._value = None
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.value = value
        self._on_press_value = self._value  # 记录鼠标按下时的值

        # 设置验证器
        double_validator = QtGui.QDoubleValidator()
        double_validator.setDecimals(decimals)
        self.setValidator(double_validator)

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(20)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet(
        """ QLineEdit {background-color: #f7f7f7; border: 1px solid #aaaaaa; border-radius: 3px; }
            QLineEdit:hover { background-color: #dfe8ed; }
        """)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        _value = min(self.max_val, max(self.min_val, val))
        if self._value != _value:
            self._value = _value
            self.sigValueChanged.emit(self._value)
        self.setText(self.format % self._value)

    def set_value_no_signal(self, val):
        self._value = min(self.max_val, max(self.min_val, val))
        self.setText(self.format % self._value)

    def mouseDoubleClickEvent(self, event):
        """双击可编辑"""
        self.setFocus()
        event.accept()

    def mousePressEvent(self, event):
        if not self.hasFocus():
            self._on_press_value = self._value
            self.pressed = True
            self.drag_position = event.pos()
        event.ignore()  # 忽略事件, 使父类 ToolWindow 能够接收到事件, 并获取 Focus

    def mouseReleaseEvent(self, event):
        self.pressed = False
        event.accept()

    def mouseMoveEvent(self, event):
        """按下并拖动可改变数值"""
        if self.pressed and not self.hasFocus():
            scale = 1
            # 如果按下shift键, 则按照最小步长移动
            if event.modifiers() == Qt.ShiftModifier:
                scale = 0.02
            # value.setter
            self.value = self._on_press_value + \
                int((event.pos().x() - self.drag_position.x()) * scale) * self.step
        event.accept()

    def keyPressEvent(self, event):
        """ESC或回车退出编辑"""
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Return:
            self.clearFocus() # 清除焦点
        else:
            super().keyPressEvent(event) # 调用父类的方法

    def focusInEvent(self, a0: QFocusEvent) -> None:
        # 修改文字为只包含数字
        self.setText(self.decimal_format % self._value)
        self.setAlignment(Qt.AlignLeft)
        return super().focusInEvent(a0)

    def focusOutEvent(self, event):
        """失去焦点时恢复原来的样式"""
        self.setAlignment(Qt.AlignCenter)
        super().focusOutEvent(event)
        # value.setter
        self.value = float(self.text())

    def paintEvent(self, event):
        """绘制进度条"""
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QtGui.QColor(41, 128, 185, 80))  # 设置长方形颜色为半透明
        rect_width = (self.value - self.min_val) / max((self.max_val - self.min_val), 0.0001) * self.width()
        painter.drawRoundedRect(0, 0, rect_width, self.height(), 3, 3)  # 绘制带圆角的长方形
        painter.end()


class DragValueItem(QtWidgets.QWidget, ToolItem):

        sigChanged = QtCore.Signal(object)

        def __init__(self, label:str, value, min_val, max_val, step, decimals:int=0,
                     format: str=None,
                     callback: Callable=None):
            super().__init__()
            self.set_label(label)

            self.name_label = QtWidgets.QLabel(label, self)
            self.value_drager = DragValue(value, min_val, max_val, step, decimals, format, self)

            self.box = create_layout(self, True,
                                    [self.value_drager, self.name_label],
                                    [5, 2], spacing=10)

            self.value_drager.sigValueChanged.connect(self._on_changed)

            if callback is not None:
                self.sigChanged.connect(callback)

        def _on_changed(self, val):
            return self.sigChanged.emit(val)

        @property
        def value(self):
            return self.value_drager.value

        @value.setter
        def value(self, val):
            self.value_drager.value = val


class DragArrayItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.Signal(object)

    def __init__(self, label:str, value, min_val, max_val, step, decimals,
                    format: Union[str, Tuple[str]]=None, callback: Callable=None,
                    horizontal=True, show_label=True):
        super().__init__()
        self.length = len(value)
        self.set_label(label)

        self._value = list(value)
        min_val = self._validate_arg(min_val)
        max_val = self._validate_arg(max_val)
        step = self._validate_arg(step)
        decimals = self._validate_arg(decimals)
        format = self._validate_arg(format)

        self.inputs_frame = QtWidgets.QFrame(self)
        self.inputs_layout = QtWidgets.QHBoxLayout(self.inputs_frame) if horizontal \
            else QtWidgets.QVBoxLayout(self.inputs_frame)
        self.inputs_layout.setContentsMargins(0, 0, 0, 0)

        self.inputs: List[DragValue] = []
        for i in range(self.length):
            input = DragValue(value[i], min_val[i], max_val[i], step[i], decimals[i], format[i], self.inputs_frame)
            input.sigValueChanged.connect(self._on_changed)
            input.setObjectName(str(i))  # 设置objectName, 用于区分信号来源
            self.inputs.append(input)
            self.inputs_layout.addWidget(input, 1)

        if show_label:
            self.box = create_layout(self, True, [self.inputs_frame, QtWidgets.QLabel(label, self)], [5, 2], spacing=10)
        else:
            self.box = create_layout(self, True, [self.inputs_frame], [5], spacing=10)

        if callback is not None:
            self.sigChanged.connect(callback)

    def _validate_arg(self, arg) -> Sequence[Number]:
        if isinstance(arg, (list, tuple, np.ndarray)):
            assert len(arg) == self.length, "arg length must be equal to value length"
            return arg
        elif isinstance(arg, (int, float)) or arg is None:
            return [arg] * self.length
        else:
            raise TypeError(f"arg must be list, tuple, int or float, but got {type(arg)}")

    def _on_changed(self, val):
        id = int(self.sender().objectName())
        self._value[id] = val
        # return self.sigChanged.emit(self._value)
        return self.sigChanged.emit((id, val))

    @property
    def value(self) -> List[Number]:
        return self._value

    @value.setter
    def value(self, val):
        for i in range(self.length):
            # 如果值不相等, 会触发 DragValue.sigValueChanged 信号, 进而触发 self.sigChanged 信号
            # 进而 self._value 的更新在 self._on_changed 中完成
            self.inputs[i].value = val[i]

    def set_value_no_signal(self, val):
        for i in range(len(val)):
            self.inputs[i].set_value_no_signal(val[i])
            self._value[i] = val[i]


class ImageViewItem(QtWidgets.QLabel, ToolItem):

    def __init__(self, parent=None, img: np.ndarray=None, img_size=(400, 300), auto_scale=True, window: 'ToolWindow'=None, img_format="bgr") -> None:
        super().__init__(parent)
        self.setStyleSheet("border: 1px solid gray;")
        self._img_np: np.ndarray = None
        self._auto_scale = auto_scale
        self._img_size = img_size
        self._win = window
        self._img_format_3channel ={'rgb': QtGui.QImage.Format_RGB888, 'bgr': QtGui.QImage.Format_BGR888}[img_format]
        self.setData(img)

    @property
    def value(self):
        return self._img_np

    @value.setter
    def value(self, img: np.ndarray):
        if img is None:
            return
        self._img_np = np.array(img, dtype=np.uint8)
        self._update_image(self._img_size)

    def setData(self, img: np.ndarray):
        self.value = img

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if self._img_np is not None and self._auto_scale:
            # scale 为窗口的尺寸
            w, h = self._img_size
            w_new = self.width() - 2
            h_new = int(h * max(w_new / max(w, 1), 1e-4))
            self._img_size = (w_new, h_new)
            self._update_image(self._img_size)

            # QtCore.QTimer.singleShot(10, self._win.adjustSize)
        a0.accept()

    def _update_image(self, size: Tuple):
        q_image = self.numpy_to_qimage(self._img_np, img_format=self._img_format_3channel)

        scaled_pixmap = QtGui.QPixmap(q_image).scaled(QSize(*size), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img_size = (scaled_pixmap.width(), scaled_pixmap.height())
        self.setPixmap(scaled_pixmap)

    @staticmethod
    def numpy_to_qimage(img_np: np.ndarray, img_format):
        """
        将 NumPy 数组转换为 QImage

        Parameters:
        - img_np : np.ndarray of uint8
        - img_format : QtGui.QImage.Format
        """
        channels = 1 if img_np.ndim < 3 else img_np.shape[2]

        if channels == 1:
            img_format = QtGui.QImage.Format_Grayscale8

        # 假设图像是 RGB 模式
        return QtGui.QImage(
            img_np.data,
            img_np.shape[1],  # 宽度
            img_np.shape[0],  # 高度
            img_np.shape[1] * channels,  # 每行的字节数 (3 个通道)
            img_format
        )


class DirectorySelectItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.Signal(str)

    def __init__(self, label:str, value:str, callback: Callable=None):
        super().__init__()
        self.set_label(label)
        self.setMaximumWidth(600)
        self.name_label = QtWidgets.QLabel(label, self)
        self.text_editor = QtWidgets.QLineEdit(value, self)
        self.text_editor.setFocusPolicy(Qt.NoFocus)
        self.path_button = QtWidgets.QPushButton(self)
        # 设置按钮图标
        self.path_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        # self.path_button.setFlat(True)
        self.path_button.setFixedWidth(30)
        frame = QtWidgets.QWidget()
        self.box1 = create_layout(frame, True, [self.text_editor, self.path_button], [4, 1], spacing=5)
        self.box = create_layout(self, True, [frame, self.name_label], [5, 2], spacing=10)

        # 信号/槽
        if callback is not None:
            self.sigChanged.connect(callback)

        self.path_button.clicked.connect(self.select_path)

    def select_path(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", options=options)
        if dir and dir != self.value:
            self.text_editor.setText(dir)
            self.sigChanged.emit(dir)

    @property
    def value(self):
        return self.text_editor.text()

    @value.setter
    def value(self, val):
        self.text_editor.setText(val)
        self.sigChanged.emit(val)

class PathSelectItem(DirectorySelectItem):

    def __init__(self, label: str, value: str, callback: Callable = None):
        super().__init__(label, value, callback)
        self.path_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))

    def select_path(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Path", options=options)
        if path and path != self.value:
            self.text_editor.setText(path)
            self.sigChanged.emit(path)


class ToolBox():

    Windows: Dict[str, ToolWindow] = {}
    Items: Dict[str, ToolItem] = {}
    ContainerStack: List[ToolContainer] = []
    app = None
    Key = Qt.Key
    KeyBindings = dict()
    labelCount = dict()

    @classmethod
    def exec(cls):
        if cls.app is not None:
            cls.app.exec_()

    @classmethod
    def windowAllClosed(cls):
        for win in cls.Windows.values():
            if win.isVisible():
                return False
        return True

    @classmethod
    def keyPressEvent(cls, event):
        if event.key() in cls.KeyBindings.keys():
            cls.KeyBindings[event.key()]()

    @classmethod
    @contextmanager
    def window(cls,
        label="Toolbox",
        parent:QtWidgets=None,
        spacing=5,
        pos: Sequence[int]=(0, 0),
        size: Sequence[int]=None,
        frameless: bool=True,
        hide: bool=False
    ):
        """
        创建一个窗口

        Parameters:
        - label : str, optional, default: "Toolbox", 窗口标题
        - parent : QtWidgets, optional, default: None, 父窗口，若为 None, 则为顶级窗口, 否则为 Qt.Tool
        - spacing : int, optional, default: 5, 控件间距
        - pos : Sequence[int], optional, default: (0, 0), 窗口位置 (x, y), 相对于父窗口
        - size : Sequence[int], optional, default: None, 窗口大小 (width, height)
        - frameless : bool, optional, default: True, 是否无边框
        - hide : bool, optional, default: False, 是否隐藏窗口

        Yields:
        - ToolWindow, 窗口实例
        """
        try:
            parent = parent if isinstance(parent, QtWidgets.QWidget) else None
            cls.app = QtWidgets.QApplication.instance()
            if cls.app is None:
                QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
                cls.app = QtWidgets.QApplication([])

            if label in cls.Windows.keys():
                win = cls.Windows[label]
            else:
                if frameless:
                    win = ToolWindowFrameless(parent, label, spacing)
                else:
                    win = ToolWindow(parent, label, spacing)
                if parent is not None:
                    parent.show()
                    QtCore.QTimer.singleShot(100, lambda: (win.move(parent.pos() + QPoint(pos[0], pos[1])), win.setVisible(not hide)))
                else:
                    win.move(QPoint(pos[0], pos[1]))
                    win.setVisible(not hide)

            cls.Windows[label] = win
            cls.ContainerStack.append(win)
            if size is not None:
                win.resize(size[0], size[1])

            win.keyPressEvent = cls.keyPressEvent  # 覆盖 keyPressEvent 方法
            yield win

        finally:
            cls.ContainerStack.pop()

    @classmethod
    @contextmanager
    def group(cls, label, horizontal=True, spacing=5, show=True, collapsible=False, collapsed=False):
        """show: 是否显示group的边框"""
        try:
            if collapsible:
                for container in cls.ContainerStack[::-1]:
                    if isinstance(container, ToolWindow):
                        break
                container = ToolCollapsibeGroup(label, horizontal,
                                spacing=spacing, collapsed=collapsed,
                                window=container)
            elif show:
                container = ToolGroupBox(label, horizontal, spacing=spacing)
            else:
                container = ToolGroup(horizontal, spacing=spacing)
            cls.ContainerStack[-1].add_item(container)
            cls.ContainerStack.append(container)
            yield container
        finally:
            cls.ContainerStack.pop()

    @classmethod
    @contextmanager
    def splitter(cls, horizontal=True, spacing=5, stretchs:Sequence[int]=None):
        try:
            container = ToolSplitter(horizontal, spacing=spacing)
            cls.ContainerStack[-1].add_item(container)
            cls.ContainerStack.append(container)
            yield container
        finally:
            container.set_stretchs(stretchs)
            cls.ContainerStack.pop()

    @classmethod
    def clean(cls):
        for box in cls.Windows.values():
            box.close()

    @classmethod
    def _add_item(cls, label:str, item: ToolItem):
        # check if label exists
        if cls.labelCount.get(label, 0) > 0:
            cls.labelCount[label] += 1
            label += f"_{cls.labelCount[label]-1}"
        else:
            cls.labelCount[label] = 1
        cls.ContainerStack[-1].add_item(item)
        cls.Items[label] = item

    @classmethod
    def get_value(cls, label):
        """获取某个控件的值"""
        return cls.Items[label].value

    @classmethod
    def set_value(cls, label, value):
        cls.Items[label].value = value

    @classmethod
    def get_widget(cls, label) -> ToolItem:
        return cls.Items[label]

    @classmethod
    def get_window(cls, label):
        return cls.Windows[label]

    # add items
    @classmethod
    def add_button(cls, label:str, value=False, checkable=False, callback: Callable=None) -> ButtonItem:
        button = ButtonItem(label, value, checkable, callback)
        cls._add_item(label, button)
        return button

    @classmethod
    def add_checkbox(cls, label:str, value=False, callback: Callable=None) -> CheckBoxItem:
        checkbox = CheckBoxItem(label, value, callback)
        cls._add_item(label, checkbox)
        return checkbox

    @classmethod
    def add_checklist(cls, label:str, items=Sequence[str], value=None, horizontal=True,
                      exclusive=True, callback: Callable=None) -> CheckListItem:
        checklist = CheckListItem(label, items, value, horizontal, exclusive, callback)
        cls._add_item(label, checklist)
        return checklist

    @classmethod
    def add_separator(cls, horizontal=True):
        """horizontal: True for horizontal line, False for vertical line"""
        line = QtWidgets.QFrame()
        if horizontal:
            line.setFrameShape(QtWidgets.QFrame.HLine)
        else:
            line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        cls.ContainerStack[-1].add_item(line)

    @classmethod
    def add_spacer(cls, size, horizontal=False):
        size = (size, 1) if horizontal else (1, size)
        spacer = QtWidgets.QSpacerItem(size[0], size[1], QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        cls.ContainerStack[-1].get_layout().addItem(spacer)

    @classmethod
    def add_stretch(cls, size=1):
        cls.ContainerStack[-1].get_layout().addStretch(size)

    @classmethod
    def add_text(cls, label, text=None):
        text_view = QtWidgets.QLabel(label if text is None else text)
        cls._add_item(label, text_view)
        return text_view

    @classmethod
    def add_combo(cls, label:str, items=Sequence[str], value=0, callback: Callable=None) -> ComboItem:
        combo = ComboItem(label, items, value, callback)
        cls._add_item(label, combo)
        return combo

    @classmethod
    def add_text_editor(cls, label:str, value:str="", editable=True, callback: Callable=None) -> TextEditorItem:
        text = TextEditorItem(label, value, editable, callback)
        cls._add_item(label, text)
        return text

    @classmethod
    def add_slider(cls, label:str, value, min_val, max_val, step, decimals:int=0, callback: Callable=None) -> SliderItem:
        slider = SliderItem(label, value, min_val, max_val, step, decimals, callback)
        cls._add_item(label, slider)
        return slider

    @classmethod
    def add_array_int(cls, label:str, value:Sequence[int], editable=True, format=None, callback: Callable=None,
                      horizontal: bool=True, show_label: bool=True) -> ArrayTypeItem:
        array_int = ArrayTypeItem(label, value, int, editable, format, callback, horizontal, show_label)
        cls._add_item(label, array_int)
        return array_int

    @classmethod
    def add_array_float(cls, label:str, value:Sequence[float], editable=True, format=None, callback: Callable=None,
                        horizontal: bool=True, show_label: bool=True) -> ArrayTypeItem:
        array_float = ArrayTypeItem(label, value, float, editable, format, callback, horizontal, show_label)
        cls._add_item(label, array_float)
        return array_float

    @classmethod
    def add_drag_value(cls, label:str, value, min_val, max_val, step, decimals=2, format:str=None,
                     callback: Callable=None) -> DragValueItem:
        drag_int = DragValueItem(label, value, min_val, max_val, step, decimals, format, callback)
        cls._add_item(label, drag_int)
        return drag_int

    @classmethod
    def add_drag_array(cls,
        label:str,
        value: NumberTuple,
        min_val: Union[Number, NumberTuple],
        max_val: Union[Number, NumberTuple],
        step: Union[Number, NumberTuple],
        decimals: Union[int, Sequence[int]]=0,
        format: Union[str, Sequence[str]]=None,
        callback: Callable=None,
        horizontal: bool=True,
        show_label: bool=True
    ) -> DragArrayItem:
        drag_array = DragArrayItem(label, value, min_val, max_val, step, decimals, format, callback, horizontal, show_label)
        cls._add_item(label, drag_array)
        return drag_array

    @classmethod
    def add_timer(cls, label:str, interval_ms: int, callback: Callable, parent: QtWidgets.QWidget=None):
        """
        添加一个定时器

        Parameters:
        - label : str, 定时器标签
        - interval_ms : int, 定时器间隔时间, 单位毫秒
        - callback : Callable, 定时器回调函数
        """
        if parent is None:
            parent = cls.ContainerStack[-1]
        timer = QtCore.QTimer(parent)
        cls.Items[label] = timer
        timer.timeout.connect(callback)
        timer.start(interval_ms)

    @classmethod
    def add_image_view(cls, label:str, img: np.ndarray, img_size=(400, 300), img_format="rgb") -> ImageViewItem:
        assert img_format in ['rgb', 'bgr'], "img_format must be 'rgb' or 'bgr'"
        for container in cls.ContainerStack[::-1]:
            if isinstance(container, ToolWindow):
                break
        image_viewer = ImageViewItem(None, img=img, img_size=img_size, window=container, img_format=img_format)
        cls._add_item(label, image_viewer)
        return image_viewer

    @classmethod
    def add_directory(cls, label:str, value:str="", callback: Callable=None) -> DirectorySelectItem:
        directory = DirectorySelectItem(label, value, callback)
        cls._add_item(label, directory)
        return directory

    @classmethod
    def add_filepath(cls, label:str, value:str="", callback: Callable=None) -> PathSelectItem:
        directory = PathSelectItem(label, value, callback)
        cls._add_item(label, directory)
        return directory

    @classmethod
    def add_widget(cls, label:str, widget: QtWidgets.QWidget):
        cls._add_item(label, widget)

    @classmethod
    def add_key_binding(cls, key, callback: Callable):
        cls.KeyBindings[key] = callback

    @classmethod
    def print_help(cls):
        print("ToolBox Key Bindings:")
        for key, callback in cls.KeyBindings.items():
            print(f"    - {Qt.Key(key).name.decode('utf-8')}: {callback.__name__}")
