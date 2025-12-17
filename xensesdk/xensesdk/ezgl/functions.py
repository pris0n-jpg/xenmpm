from qtpy import QtGui, QtCore, QtWidgets, API
import numpy as np
import cv2
import sys
import warnings
import traceback
import heapq
from threading import Lock
from pathlib import Path
from datetime import datetime
from functools import update_wrapper, singledispatchmethod
from typing import Union
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

__all__ = [
    'clip_scalar', 'mkColor', 'glColor', 'intColor', 'colorstr', 'clip_array',
    'Filter', 'increment_path', 'now', 'dispatchmethod', 'singleton', 'multiton', 'get_path',
    'ReadWriteLock', 'NumberPool', 'LoopRange', 'SortedSet', 'printExc', 'call_once', "CircularBuffer"
]

Colors = {
    'b': QtGui.QColor(0,0,255,255),
    'g': QtGui.QColor(0,255,0,255),
    'r': QtGui.QColor(255,0,0,255),
    'c': QtGui.QColor(0,255,255,255),
    'm': QtGui.QColor(255,0,255,255),
    'y': QtGui.QColor(255,255,0,255),
    'k': QtGui.QColor(0,0,0,255),
    'w': QtGui.QColor(255,255,255,255),
    'd': QtGui.QColor(150,150,150,255),
    'l': QtGui.QColor(200,200,200,255),
    's': QtGui.QColor(100,100,150,255),
}

def getQApplication():
    if API == "pyside6":
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    else:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    return QtWidgets.QApplication()

def clip_scalar(val, vmin, vmax):
    """ convenience function to avoid using np.clip for scalar values """
    return vmin if val < vmin else vmax if val > vmax else val

def mkColor(*args):
    """
    Convenience function for constructing QColor from a variety of argument
    types. Accepted arguments are:

    ================ ================================================
     'c'             one of: r, g, b, c, m, y, k, w
     R, G, B, [A]    integers 0-255
     (R, G, B, [A])  tuple of integers 0-255
     float           greyscale, 0.0-1.0
     int             see :func:`intColor() <pyqtgraph.intColor>`
     (int, hues)     see :func:`intColor() <pyqtgraph.intColor>`
     "#RGB"
     "#RGBA"
     "#RRGGBB"
     "#RRGGBBAA"
     QColor          QColor instance; makes a copy.
    ================ ================================================
    """
    err = 'Not sure how to make a color from "%s"' % str(args)
    if len(args) == 1:
        if isinstance(args[0], str):
            c = args[0]
            if len(c) == 1:
                try:
                    return Colors[c]
                except KeyError:
                    raise ValueError('No color named "%s"' % c)
            have_alpha = len(c) in [5, 9] and c[0] == '#'  # "#RGBA" and "#RRGGBBAA"
            if not have_alpha:
                # try parsing SVG named colors, including "#RGB" and "#RRGGBB".
                # note that QColor.setNamedColor() treats a 9-char hex string as "#AARRGGBB".
                qcol = QtGui.QColor()
                qcol.setNamedColor(c)
                if qcol.isValid():
                    return qcol
                # on failure, fallback to pyqtgraph parsing
                # this includes the deprecated case of non-#-prefixed hex strings
            if c[0] == '#':
                c = c[1:]
            else:
                raise ValueError(f"Unable to convert {c} to QColor")
            if len(c) == 3:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = 255
            elif len(c) == 4:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = int(c[3]*2, 16)
            elif len(c) == 6:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = 255
            elif len(c) == 8:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = int(c[6:8], 16)
            else:
                raise ValueError(f"Unknown how to convert string {c} to color")
        elif isinstance(args[0], QtGui.QColor):
            return QtGui.QColor(args[0])
        elif np.issubdtype(type(args[0]), np.floating):
            r = g = b = int(args[0] * 255)
            a = 255
        elif hasattr(args[0], '__len__'):
            if len(args[0]) == 3:
                r, g, b = args[0]
                a = 255
            elif len(args[0]) == 4:
                r, g, b, a = args[0]
            elif len(args[0]) == 2:
                return intColor(*args[0])
            else:
                raise TypeError(err)
        elif np.issubdtype(type(args[0]), np.integer):
            return intColor(args[0])
        else:
            raise TypeError(err)
    elif len(args) == 3:
        r, g, b = args
        a = 255
    elif len(args) == 4:
        r, g, b, a = args
    else:
        raise TypeError(err)
    args = [int(a) if np.isfinite(a) else 0 for a in (r, g, b, a)]
    return QtGui.QColor(*args)


def glColor(*args, **kargs):
    """
    Convert a color to OpenGL color format (r,g,b,a) floats 0.0-1.0
    Accepts same arguments as :func:`mkColor <pyqtgraph.mkColor>`.
    """
    c = mkColor(*args, **kargs)
    return c.getRgbF()


def intColor(index, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255):
    """
    Creates a QColor from a single index. Useful for stepping through a predefined list of colors.

    The argument *index* determines which color from the set will be returned. All other arguments determine what the set of predefined colors will be

    Colors are chosen by cycling across hues while varying the value (brightness).
    By default, this selects from a list of 9 hues."""
    hues = int(hues)
    values = int(values)
    ind = int(index) % (hues * values)
    indh = ind % hues
    indv = ind // hues
    if values > 1:
        v = minValue + indv * ((maxValue-minValue) // (values-1))
    else:
        v = maxValue
    h = minHue + (indh * (maxHue-minHue)) // hues

    return QtGui.QColor.fromHsv(h, sat, v, alpha)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


# umath.clip was slower than umath.maximum(umath.minimum).
# See https://github.com/numpy/numpy/pull/20134 for details.
_win32_clip_workaround_needed = (
    sys.platform == 'win32' and
    tuple(map(int, np.__version__.split(".")[:2])) < (1, 22)
)

def clip_array(arr, vmin, vmax, out=None):
    # replacement for np.clip due to regression in
    # performance since numpy 1.17
    # https://github.com/numpy/numpy/issues/14281

    if vmin is None and vmax is None:
        # let np.clip handle the error
        return np.clip(arr, vmin, vmax, out=out)

    if vmin is None:
        return np.core.umath.minimum(arr, vmax, out=out)
    elif vmax is None:
        return np.core.umath.maximum(arr, vmin, out=out)
    elif _win32_clip_workaround_needed:
        if out is None:
            out = np.empty(arr.shape, dtype=np.find_common_type([arr.dtype], [type(vmax)]))
        out = np.core.umath.minimum(arr, vmax, out=out)
        return np.core.umath.maximum(out, vmin, out=out)

    else:
        return np.core.umath.clip(arr, vmin, vmax, out=out)


class Filter:
    """数据滤波"""
    def __init__(self, data=None, alpha=0.2):
        self._data = data
        self._alpha = alpha

    def update(self, new_data):
        if self._data is None:
            self._data = new_data
        self._data = (1 - self._alpha) * self._data + self._alpha * new_data

    @property
    def data(self):
        if self._data is None:
            return 0
        return self._data


def increment_path(path):
    """若输入文件路径已存在, 为了避免覆盖, 自动在后面累加数字返回一个可用的路径
    例如输入 './img.jpg' 已存在, 则返回 './img_0000.jpg'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix
    stem = path.stem
    for n in range(0, 9999):
        if not path.with_name(f"{stem}_{n:04d}{suffix}").exists():  #
            break
    return str(path.with_name(f"{stem}_{n:04d}{suffix}"))


def now(fmt='%y_%m_%d_%H_%M_%S'):
    return datetime.now().strftime(fmt)


class dispatchmethod(singledispatchmethod):
    """Dispatch a method to different implementations
    depending upon the type of its first argument.
    If there is no argument, use 'object' instead.
    """
    def __get__(self, obj, cls=None):
        def _method(*args, **kwargs):
            if len(args) > 0:
                class__ = args[0].__class__
            elif len(kwargs) > 0:
                class__ = next(kwargs.values().__iter__()).__class__
            else:
                class__ = object

            method = self.dispatcher.dispatch(class__)
            return method.__get__(obj, cls)(*args, **kwargs)

        _method.__isabstractmethod__ = self.__isabstractmethod__
        _method.register = self.register
        update_wrapper(_method, self.func)
        return _method


def singleton(cls):
    """
    单例装饰器

    Usage:
        @singleton
        class MyClass:
            pass
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def multiton(max_instances):
    """
    多例装饰器

    Parameters:
    - max_instances : int, 最大实例数

    Usage:
        @multiton(3)
        class MyClass:
            pass
    """
    def decorator(cls):
        instances = []

        def get_instance(*args, **kwargs):
            if len(instances) < max_instances:
                instance = cls(*args, **kwargs)
                instances.append(instance)
                return instance
            else:
                # 如果已经达到最大实例数，返回第一个实例
                return instances[0]

        return get_instance
    return decorator


def call_once(func):
    """装饰器, 确保实例成员函数只能被调用一次"""
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_called_once'):
            self._called_once = list()
        if func not in self._called_once:
            result = func(self, *args, **kwargs)
            self._called_once.append(func)
            return result
        else:
            return None

    return wrapper


class NumberPool:
    """
    资源池, 用于管理一组数字 id 资源
    """
    def __init__(self, cnt):
        self.pool = list(range(cnt+1))
        heapq.heapify(self.pool)  # 将列表转换为堆
        self._pool_copy = self.pool.copy()
        self._pool_lock = Lock()

    def gen_num(self) -> int:
        """
        生成一个可用的数字

        Returns:
        - int, 可用的数字
        """
        with self._pool_lock:
            if not self.pool:
                raise Exception("No more numbers available")
            return heapq.heappop(self.pool)  # 弹出并返回堆顶元素

    def free_num(self, num):
        """
        释放一个数字

        Parameters:
        - num : int, 要归还的数字
        """
        with self._pool_lock:
            heapq.heappush(self.pool, num)  # 将元素添加到堆中

    def free_all(self):
        """
        释放所有数字
        """
        self.pool = self._pool_copy.copy()


class LoopRange:
    """
    循环范围迭代器
    例如 LoopRange(0, 5, 2) 会生成 0, 2, 4, 0, 2, 4, ...
    """
    def __init__(self, start:int, end:int, step:int=1):
        self.current = start
        self.start = start
        self.end = end
        self.step = max(step, 1)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.current
        self.current += self.step
        if self.current >= self.end:
            self.current = self.start
        return value


def get_path() -> Path:
    return BASE_DIR


class ReadWriteLock:
    def __init__(self):
        self._read_lock = Lock()
        self._write_lock = Lock()
        self._num_readers = 0

    def acquire_read(self):
        with self._read_lock:
            self._num_readers += 1
            if self._num_readers == 1:
                self._write_lock.acquire()

    def release_read(self):
        with self._read_lock:
            self._num_readers -= 1
            if self._num_readers == 0:
                self._write_lock.release()

    def acquire_write(self):
        self._write_lock.acquire()

    def release_write(self):
        self._write_lock.release()


class SortedSet:
    """
    有序集合
    """
    def __init__(self, iterable=None, key_func=None):
        """
        Initialization

        Parameters:
        - iterable : iterable, optional, default: None, 可迭代对象, 如 list
        - key_func : Callable, optional, default: None, 比较函数
        """
        self.key_func = key_func or (lambda x: x)
        self.items: list = sorted(set(iterable or []), key=self.key_func)

    def __or__(self, other: Union[list, 'SortedSet']):
        if isinstance(other, SortedSet):
            other = other.items

        # 使用 self.__class__ 保证继承类返回的是当前类的实例
        return self.__class__(self.items + other, self.key_func)

    def __and__(self, other: Union[list, 'SortedSet']):
        if isinstance(other, SortedSet):
            other = other.items
        return self.__class__(set(self.items) & set(other), self.key_func)

    def __ior__(self, other: Union[list, 'SortedSet']):
        if isinstance(other, SortedSet):
            other = other.items
        self.items += other
        self.items = sorted(set(self.items), key=self.key_func)
        return self

    def __iand__(self, other: Union[list, 'SortedSet']):
        if isinstance(other, SortedSet):
            other = other.items
        self.items = sorted(set(self.items) & set(other), key=self.key_func)
        return self

    def add(self, item):
        if item not in self.items:
            self.items.append(item)
            self.items.sort(key=self.key_func)

    def remove(self, item):
        if item in self.items:
            self.items.remove(item)

    def clear(self):
        """
        Remove all items from the set
        """
        self.items.clear()

    def update(self, *items):
        """
        Update with a new set of items
        """
        self.items = sorted(set(items), key=self.key_func)

    def __iter__(self):
        return iter(self.items)

    def __contains__(self, item):
        return item in self.items

    def __repr__(self):
        return f"SortedSet({self.items})"

    def copy(self):
        return self.__class__(self.items, self.key_func)


def formatException(exctype, value, tb, skip=0):
    """Return a list of formatted exception strings.

    Similar to traceback.format_exception, but displays the entire stack trace
    rather than just the portion downstream of the point where the exception is
    caught. In particular, unhandled exceptions that occur during Qt signal
    handling do not usually show the portion of the stack that emitted the
    signal.
    """
    lines = traceback.format_exception(exctype, value, tb)
    lines = [lines[0]] + traceback.format_stack()[:-(skip+1)] + ['  --- exception caught here ---\n'] + lines[1:]
    return lines


def getExc(indent=4, prefix="", skip=1):
    lines = formatException(*sys.exc_info(), skip=skip)
    lines2 = []
    for l in lines:
        lines2.extend(l.strip('\n').split('\n'))
    lines3 = [" "*indent + prefix + l for l in lines2]
    return '\n'.join(lines3)


def printExc(msg="", indent=2, prefix=""):
    """Print an error message followed by an indented exception backtrace
    (This function is intended to be called within except: blocks)"""
    exc = getExc(indent=indent, prefix=prefix, skip=2)
    print(" "*indent + prefix + '='*30 + '>>')
    warnings.warn("\n".join([msg, exc]), RuntimeWarning, stacklevel=2)
    print(" "*indent + prefix + '='*30 + '<<')


def draw_array_on_image(array: np.ndarray, cell_width=50, font_scale=0.5, thickness=1):
    """
    将2D数组绘制在图片上, 用于调试

    Parameters:
    - array : np.ndarray, 2D数组, 内容为要绘制的数字或字符串
    - cell_width : int, optional, default: 50, 每个单元格的宽度
    - font_scale : float, optional, default: 0.5, 字体大小
    - thickness : int, optional, default: 1, 字体粗细

    Returns:
    - np.ndarray, 带有绘制数组的单通道图像
    """
    (_, text_height), baseline = cv2.getTextSize(" ", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cell_height = text_height + 10
    # 获取数组的维度
    rows, cols = array.shape
    # 创建白色背景的图像
    image = np.zeros((cell_height * rows, cell_width * cols), dtype=np.uint8)

    # 遍历数组，并在图片上绘制每个元素
    for i in range(rows):
        for j in range(cols):
            # 计算每个单元格的位置
            start_x = j * cell_width
            start_y = i * cell_height

            # 255 填充矩形
            cv2.rectangle(image, (start_x, start_y), (start_x + cell_width, start_y + cell_height), 255, thickness=-1)
            # 绘制一个黑色边框
            cv2.rectangle(image, (start_x, start_y), (start_x + cell_width, start_y + cell_height), 0, 1)

            # 获取文本内容
            text = str(array[i, j])

            # 计算文本的大小和偏移，以使其居中
            text_x = start_x + 5
            text_y = start_y + (cell_height + text_height) // 2

            # 在单元格中心绘制文本
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return image

class CircularBuffer:
    def __init__(self, size):
        self.size = size # 缓冲区大小
        self.buffer =[None] * size
        self.ptr = -1   #指针

    def put(self, item):
        """将元素加入到尾指针处"""
        next = (self.ptr + 1) % self.size
        self.buffer[next]= item
        self.ptr = next

    def get(self):
        """返回尾指针所指元素"""
        if self.ptr != -1:
            return self.buffer[self.ptr]
        else:
            return None
        
    def clear(self):
        """清空缓冲区"""
        self.ptr = -1
        self.buffer =[None] * self.size


class GridInterpolator:

    def __init__(self, src_grid_x: Union[list, np.ndarray], src_grid_y: Union[list, np.ndarray], dst_pos):
        """
        网格插值器: 根据源数据网格 src_grid 和目标插值点 dst_pos 初始化双线性插值权重, 以便将源数据网格点位上的值插值到目标点位上

        Parameters:
        - src_grid_x : array-like of shape(ncol, ), 源数据网格 x 坐标序列
        - src_grid_y : array-like of shape(nrow, ), 源数据网格 y 坐标序列
        - dst_pos : array-like of shape(..., 2), 目标插值点坐标
        """
        assert dst_pos.shape[-1] == 2, "The last dimension of dst_pos must be 2"
        src_grid_x = src_grid_x.astype(np.float32)
        src_grid_y = src_grid_y.astype(np.float32)
        dst_pos = dst_pos.astype(np.float32)
        
        self.n_row_src, self.n_col_src = len(src_grid_y), len(src_grid_x)
        self.dst_shape = dst_pos.shape

        # 计算 dst_pos 在 src_grid 中的位置
        idx_x = np.searchsorted(src_grid_x, dst_pos[..., 0], side='right') - 1  # dst_shape
        idx_x = np.clip(idx_x, 0, self.n_col_src - 2)
        idx_y = np.searchsorted(src_grid_y, dst_pos[..., 1], side='right') - 1
        idx_y = np.clip(idx_y, 0, self.n_row_src - 2)

        # 计算 dst_pos 关联的 src_grid 中的四个点的索引
        idx_00 = idx_y * self.n_col_src + idx_x
        idx_01 = idx_y * self.n_col_src + idx_x + 1
        idx_10 = (idx_y + 1) * self.n_col_src + idx_x
        idx_11 = (idx_y + 1) * self.n_col_src + idx_x + 1
        self.dst_relate_idx = np.stack([idx_00, idx_01, idx_10, idx_11], axis=-1, dtype=np.int32)  # shape=(dst_shape, 4)
        
        # 计算 bilinear interpolation weights
        x1, x2 = src_grid_x[idx_x], src_grid_x[idx_x + 1]  # shape=(dst_shape,)
        y1, y2 = src_grid_y[idx_y], src_grid_y[idx_y + 1]
        u1 = dst_pos[..., 0] - x1
        u2 = x2 - dst_pos[..., 0]
        v1 = dst_pos[..., 1] - y1
        v2 = y2 - dst_pos[..., 1]
        denominator = (x2 - x1) * (y2 - y1)
        self.dst_weights = np.stack([u2*v2, u1*v2, u2*v1, u1*v1], axis=-1) / denominator[..., None]  # shape=(dst_shape, 4)

    def interp(self, src_data):
        """
        双线性插值

        Parameters:
        - src_data : array-like of shape (n_row_src, n_col_src, m), 源数据, m 为数据维度
        """
        assert src_data.shape[:2] == (self.n_row_src, self.n_col_src), f"The shape of src_data {src_data.shape} must be equal to src_grid {(self.n_row_src, self.n_col_src)}"
        m = src_data.shape[-1]
        src_data = src_data.reshape(self.n_row_src * self.n_col_src, m)
        
        return np.einsum('...ij,...ij->...j', src_data[self.dst_relate_idx], self.dst_weights[..., None])  # shape=(dst_shape, m)