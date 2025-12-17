import glfw
import numpy as np
from OpenGL import GL as gl
from .. import GLGraphicsItem, Matrix4x4
from ..functions import ReadWriteLock
from ..items import PointLight, RenderGroup, Camera


class Scene:

    _windows = list()

    def __init__(
        self,
        win_width = 800,
        win_height = 600,
        visible = True,
        parent: 'Scene' = None,
        title = ""
    ):
        """
        一个场景类，用于管理渲染对象和相机

        Parameters:
        - win_width : int, optional, default: 800, 窗口宽度
        - win_height : int, optional, default: 600, 窗口高度
        - visible : bool, optional, default: True, 是否显示窗口
        - parent : Scene, optional, default: None, 父场景,
            若为 None 则创建新窗口以及 OpenGL 上下文, 否则使用父场景的资源
        - title : str, optional, default: "", 窗口标题
        """

        if not glfw.init():
            raise Exception("GLFW can't be initialized")

        self._visible = visible
        if visible == False:
            win_height, win_width = 1, 1

        # 设置GLFW窗口提示，不创建实际窗口
        glfw.window_hint(glfw.VISIBLE, visible)
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)

        # 创建一个 window
        if parent is not None:
            self._window = parent._window
        else:
            self._window = glfw.create_window(win_width, win_height, title, None, None)
            Scene._windows.append(self._window)
            if not self._window:
                glfw.terminate()
                raise Exception("GLFW Pbuffer can't be created")

            glfw.set_window_aspect_ratio(self._window, win_width, win_height)
            glfw.set_mouse_button_callback(self._window, self.mousePressEvent)
            glfw.set_cursor_pos_callback(self._window, self.mouseMoveEvent)
            glfw.set_scroll_callback(self._window, self.wheelEvent)
            glfw.set_key_callback(self._window, self.keyPressEvent)
            glfw.set_window_close_callback(self._window, self.closeEvent)
            glfw.set_window_size_callback(self._window, self.resizeEvent)

        # data
        self.item_group = RenderGroup()  # 渲染物体组
        self.light_group = RenderGroup()  # 光源组
        self._render_cameras = list()
        self._camera = Camera(eye=(0,0,5), center=(0,0,0), up=(0,1,0), fov=45)
        self.last_pos = None
        self.press_pos = None
        self.cam_pressed_matrix = None
        self.rw_lock = ReadWriteLock()
        self.setBackgroundColor((0.7, 0.7, 0.7, 1))

        # 设置按键绑定
        self.key_bindings = dict()

    @property
    def visible(self):
        return self._visible

    def setVisible(self, val):
        self._visible = val
        if val:
            glfw.show_window(self._window)
        else:
            glfw.hide_window(self._window)

    def setBackgroundColor(self, color):
        self.bg_color = np.array(color, dtype=np.float32)

    def cameraLookAt(self, eye: list, center: list, up: list):
        self._camera.set_view_matrix(Matrix4x4.lookAt(eye, center, up))

    def windowShouldClose(self):
        return glfw.window_should_close(self._window)

    def update(self):
        glfw.poll_events()

        with self:
            gl.glClearColor(*self.bg_color)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glClear( gl.GL_DEPTH_BUFFER_BIT | gl.GL_COLOR_BUFFER_BIT )

            # 更新阴影贴图
            for light in self.light_group:
                light.renderShadow()

            # draw items
            self.light_group.render(self._camera)
            self.item_group.render(self._camera)

            glfw.swap_buffers(self._window)

    def terminate(self):
        glfw.terminate()

    def addItem(self, *items):
        """
        添加渲染对象
        """
        for item in items:
            self.item_group.addItem(item)
            for item in item.treeItems():
                item.setView(self)
                self.light_group |= item.lights

    def addCamera(self, *camera):
        """
        设置相机
        """
        for cam in camera:
            if cam in self._render_cameras:
                continue

            self._render_cameras.append(cam)
            cam.setView(self)

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if isinstance(value, Camera) and type(value) != Camera:
            self.addCamera(value)

        elif isinstance(value, GLGraphicsItem) and not isinstance(value, PointLight):
            self.addItem(value)

    def __enter__(self):
        self.makeCurrent()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An exception occurred: {exc_type}: {exc_value}")
        self.doneCurrent()

    def makeCurrent(self):
        glfw.make_context_current(self._window)

    def doneCurrent(self):
        glfw.make_context_current(None)

    def renderCamera(self, idx=0):
        """
        使用 camera 渲染场景
        """
        return self._render_cameras[idx].render()

    # Ballbacks
    def wheelEvent(self, window, xoffset, yoffset):
        if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            yoffset = yoffset * 0.1

        if glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS:
            self._camera._fov *= 0.999**yoffset
        else:
            self._camera.pan(0, 0, yoffset, scale=0.1)

    def mousePressEvent(self, window, button, action, mods):
        if button in [glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_MIDDLE] and action == glfw.PRESS:
            lpos = glfw.get_cursor_pos(window)
            self.press_pos = lpos
            self.last_pos = lpos
            self.cam_pressed_matrix = self._camera.get_view_matrix()

    def mouseMoveEvent(self, window, xpos, ypos):
        mouse_left_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)
        mouse_mid_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE)
        if not (mouse_mid_down or mouse_left_down):
            return

        ctrl_down = glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
        shift_down = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        alt_down = glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS

        lpos = (xpos, ypos)
        cam_matrix = self._camera.get_view_matrix()
        diff = (lpos[0] - self.last_pos[0], lpos[1] -  self.last_pos[1])
        self.last_pos = lpos

        if shift_down and not alt_down:
            cam_matrix = self.cam_pressed_matrix
            diff = (lpos[0] - self.press_pos[0], lpos[1] - self.press_pos[1])
            if abs(diff[0]) > abs(diff[1]):
                diff = (diff[0], 0)
            else:
                diff = (0, diff[1])

        if ctrl_down:
            diff = (diff[0] * 0.1, diff[1] * 0.1)

        if alt_down:
            roll = diff[0] / 5

        if mouse_left_down:
            if alt_down:
                self._camera.orbit(0, 0, roll, base=cam_matrix)
            else:
                self._camera.orbit(diff[1], diff[0], 0, base=cam_matrix)
        elif mouse_mid_down:
            self._camera.pan(diff[0], -diff[1], 0, base=cam_matrix)

    def keyPressEvent(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            # 通知所有窗口关闭
            for win in Scene._windows:
                glfw.set_window_should_close(win, True)
        if key in self.key_bindings and action == glfw.PRESS:
            self.key_bindings[key]()

    def closeEvent(self, *args):
        glfw.set_window_should_close(self._window, True)

    def resizeEvent(self, window, width, height):
        with self:
            gl.glViewport(0, 0, width, height)

    def addKeyBinding(self, key, callback):
        self.key_bindings[key] = callback


class Key:
    """
    键盘按键
    """
    LEFT_SHIFT = glfw.KEY_LEFT_SHIFT
    LEFT_CONTROL = glfw.KEY_LEFT_CONTROL
    LEFT_ALT = glfw.KEY_LEFT_ALT
    RIGHT_SHIFT = glfw.KEY_RIGHT_SHIFT
    RIGHT_CONTROL = glfw.KEY_RIGHT_CONTROL
    RIGHT_ALT = glfw.KEY_RIGHT_ALT
    SPACE = glfw.KEY_SPACE
    ESCAPE = glfw.KEY_ESCAPE
    ENTER = glfw.KEY_ENTER
    TAB = glfw.KEY_TAB
    BACKSPACE = glfw.KEY_BACKSPACE
    DELETE = glfw.KEY_DELETE
    UP = glfw.KEY_UP
    DOWN = glfw.KEY_DOWN
    LEFT = glfw.KEY_LEFT
    RIGHT = glfw.KEY_RIGHT
    A = glfw.KEY_A
    B = glfw.KEY_B
    C = glfw.KEY_C
    D = glfw.KEY_D
    E = glfw.KEY_E
    F = glfw.KEY_F
    G = glfw.KEY_G
    H = glfw.KEY_H
    I = glfw.KEY_I
    J = glfw.KEY_J
    K = glfw.KEY_K
    L = glfw.KEY_L
    M = glfw.KEY_M
    N = glfw.KEY_N
    O = glfw.KEY_O
    P = glfw.KEY_P
    Q = glfw.KEY_Q
    R = glfw.KEY_R
    S = glfw.KEY_S
    T = glfw.KEY_T
    U = glfw.KEY_U
    V = glfw.KEY_V
    W = glfw.KEY_W
    X = glfw.KEY_X
    Y = glfw.KEY_Y
    Z = glfw.KEY_Z
    F1 = glfw.KEY_F1
    F2 = glfw.KEY_F2
    F3 = glfw.KEY_F3
    F4 = glfw.KEY_F4
    F5 = glfw.KEY_F5
    F6 = glfw.KEY_F6
    F7 = glfw.KEY_F7
    F8 = glfw.KEY_F8
    F9 = glfw.KEY_F9
    NUM_0 = glfw.KEY_0
    NUM_1 = glfw.KEY_1
    NUM_2 = glfw.KEY_2
    NUM_3 = glfw.KEY_3
    NUM_4 = glfw.KEY_4
    NUM_5 = glfw.KEY_5
    NUM_6 = glfw.KEY_6
    NUM_7 = glfw.KEY_7
    NUM_8 = glfw.KEY_8
    NUM_9 = glfw.KEY_9