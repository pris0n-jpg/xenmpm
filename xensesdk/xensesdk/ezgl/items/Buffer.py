"""
Description: 封装 abo, vbo, ebo 便于使用

Author: Jin Liu
Date: 2024/03/30
"""

import OpenGL.GL as gl
from typing import List, Union
import numpy as np
from ctypes import c_void_p


GL_Type = {
    np.dtype("f4"): gl.GL_FLOAT,
    np.dtype("f2"): gl.GL_HALF_FLOAT,
    np.dtype("i4"): gl.GL_INT,
    np.dtype("u4"): gl.GL_UNSIGNED_INT,
    np.dtype("u1"): gl.GL_UNSIGNED_BYTE,
}

IntOrList = Union[int, List[int]]
ATTR_UNDIFINED, ATTR_DEFAULT, ATTR_POINTER = range(3)  # 属性指针状态: 未定义, 默认属性, 指针属性


class GLDataBlock:

    def __init__(self, dtype: np.dtype=None, layout: IntOrList=1, data: np.ndarray=None, name: str=None):
        """
        初始化一个数据块
        |#########*********---------|
                 ^        ^         ^
            commited    used      total nbytes

        Parameters:
        - dtype : np.dtype, 数据类型, 若为 None, 则根据 data 推断
        - layout : IntOrList, 数据块布局, 如 vec3 为 3, mat4 为 [4,4,4,4], (vec3, vec2) 为 [3,2], ebo索引数据可取任意值
        - data : np.ndarray, optional, default: None, 初始化数据
        - name : str, optional, default: None, 若不为 None 则打印日志
        """
        if dtype is None and data is None:
            raise ValueError("dtype and data can not be None at the same time")

        self._dtype = np.dtype(dtype) if dtype is not None else data.dtype
        self._itemsize = self._dtype.itemsize
        self._layout: list = layout if isinstance(layout, list) else [layout]
        self._data = np.empty(0, self._dtype)  # buffer 数据本地备份,
        self._used_nbytes: int = 0             # 已使用数据块指针
        self._commited_nbytes: int = 0         # 已提交数据块指针
        self._stride: int = sum(self._layout)  # 数据块步长, 单位 个
        self._attr_state = ATTR_UNDIFINED      # 属性指针状态
        self._attr_id: list = None          # shader 中的属性编号
        self._divisor: int = None           # shader 中的属性 divisor 参数

        self.buffer_id: int = None
        self.require_expand = False             # 是否需要扩容
        self.gl_type = GL_Type[self._dtype]     # 数据类型
        self.offset: int = 0                    # 数据在缓冲区的偏移量, 单位 byte

        self._name = name
        self.set_data(data)

    @property
    def pending(self) -> bool:
        """
        是否有数据待上传
        """
        return self._used_nbytes > self._commited_nbytes

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: np.dtype):
        self._dtype = np.dtype(dtype)

    @property
    def used_nbytes(self):
        """已使用的字节数量"""
        return self._used_nbytes

    @property
    def used_size(self):
        """已使用的元素数量"""
        return self._used_nbytes // self._itemsize

    @property
    def nbytes(self):
        """缓冲区总字节数"""
        return self._data.nbytes

    @property
    def count(self):
        """属性数量"""
        return self._used_nbytes // (self._stride * self._itemsize)

    @property
    def commited_count(self):
        return self._commited_nbytes // (self._stride * self._itemsize)

    @property
    def data(self):
        """已使用的数据"""
        return self._data[0:self.count]

    def set_data(self, data: np.ndarray):
        """
        更新并覆盖原数据, 更新 _used_nbytes, _commited_nbytes, _data, pending, require_expand
        """
        if data is None:  # None 则返回, 若要清空数据, 传入 np.empty(0, dtype)
            return

        data = np.array(data, dtype=self._dtype).reshape(-1, self._stride)

        # 需要扩容
        if data.size > self._data.size:
            self.require_expand = True

            if self._data.nbytes == 0:  # 第一次设置数据
                self._data = data
            else:  # 扩容为 1.5 倍， 避免频繁扩容
                self._data = np.empty((int(data.shape[0] * 1.5), self._stride), dtype=self._dtype)
                self._data[:data.shape[0]] = data
        elif data.size > 0:
            self._data[:data.shape[0]] = data

        self._used_nbytes = data.nbytes
        self._commited_nbytes = 0

        if self._name:
            print(f"[{self._name}] set data, commited: {self.commited_count}, used: {self.count}, unit: count")

    def add_data(self, data: np.ndarray):
        """
        增量更新数据, 而不覆盖原数据, 每次只需要传输新数据, 以提高性能

        Parameters:
        - data : np.ndarray, 新数据
        """
        if data is None:
            return

        data = np.array(data, dtype=self._dtype).reshape(-1, self._stride)

        if data.nbytes <= self._data.nbytes - self._used_nbytes:  # 有足够空间
            self._data[self.count : self.count + data.shape[0]] = data

        else: # 需要扩容
            self.require_expand = True
            if self._data.nbytes == 0:  # 第一次设置数据
                self._data = data
            else:  # 扩容为 1.5 倍， 避免频繁扩容
                expand_count = int((self.count + data.shape[0]) * 1.5) - self.count
                self._data = np.vstack([self._data[:self.count], data,
                                        np.empty((expand_count, self._stride), dtype=self._dtype)])
        self._used_nbytes += data.nbytes

        if self._name:
            print(f"[{self._name}] add data, commited: {self.commited_count}, used: {self.count}, unit: count")

    def check_attr_state(self):
        """
        检查属性指针状态是否和数据个数匹配, 若 count==1 , 则设置成默认属性, 否则设置成属性数组, 必须确保已经绑定了 self._vao
        """
        if (self._attr_state == ATTR_UNDIFINED):
            return

        if self.count == 1 or self._attr_state == ATTR_DEFAULT and self.count != 1:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer_id)
            self.setAttrPointer(self._attr_id, self._divisor)
        else: # self._attr_state == ATTR_POINTER and self.count != 1:
            for attr_id in self._attr_id:
                gl.glEnableVertexAttribArray(attr_id)

    def setAttrPointer(self, attr_id: IntOrList, div: int=0):
        """
        设置属性指针, 需要先绑定 buffer_id, 若只包含一个属性元素, 则直接设置成默认属性, 否则设置成属性数组

        Parameters:
        - attr_id : IntOrList, shader 中的属性编号
        - div : int, optional, default: 0, divisor 参数
        """
        if isinstance(attr_id, int):
            attr_id = [attr_id]

        self._attr_id = attr_id
        self._divisor = div

        stride = self._stride * self._itemsize
        offsets = [0] + np.cumsum(self._layout).tolist()[:-1]

        for i in range(len(self._layout)):
            if self.count == 1:
                func = getattr(gl, f"glVertexAttrib{int(self._layout[i])}fv")
                func(
                    attr_id[i],
                    np.array(self._data.ravel()[offsets[i] : offsets[i] + self._layout[i]], dtype=np.float32)
                )
                gl.glDisableVertexAttribArray(attr_id[i])
                self._attr_state = ATTR_DEFAULT # 默认属性不会被记录在 VAO 中, 需要每次绘画之前设置

            else: # 设置成属性数组
                gl.glVertexAttribPointer(
                    index = attr_id[i],
                    size = self._layout[i],
                    type = self.gl_type,
                    normalized = gl.GL_FALSE,
                    stride = stride,
                    pointer = c_void_p(self.offset + offsets[i] * self._itemsize)
                )
                gl.glVertexAttribDivisor(attr_id[i], div)
                gl.glEnableVertexAttribArray(attr_id[i])
                self._attr_state = ATTR_POINTER

        if self._name:
            print(f"[{self._name}] set attr, attr_id: {attr_id}, divisor: {div}, count: {self.count}, state: "
                  +["Undefine", "Default", "Pointer"][self._attr_state])

class DataBufferObject:

    def __init__(
        self,
        data_blocks: Union[List[GLDataBlock], GLDataBlock],
        expandable = False,
        usage = gl.GL_STATIC_DRAW,
        target = gl.GL_ARRAY_BUFFER, # gl.GL_ARRAY_BUFFER, gl.GL_ELEMENT_ARRAY_BUFFER
        name = None
    ):
        """
        初始化一个缓冲区管理对象

        Parameters:
        - data_blocks : Union[List[GLDataBlock], GLDataBlock], 管理的数据块
        - expandable : bool, optional, default: False, 数据块是否可扩大, 若为 False, 则所有数据块在一个缓冲区中顺序存储, 每个数据块的长度在初始化时确定, 不可扩展; 若为 True, 则每个数据块对应一个缓冲区, 可以扩展容量
        - usage : 枚举值, optional, default: gl.GL_STATIC_DRAW, gl.GL_DYNAMIC_DRAW, 缓冲区使用方式
        - target : 枚举值, optional, default: gl.GL_ARRAY_BUFFER, 缓冲区类型
        - name : str, optional, default: None, 缓冲区名称, 若设置, 则输出相关日志
        """
        self._blocks = data_blocks if isinstance(data_blocks, list) else [data_blocks]
        self._expandable = expandable or len(self._blocks) == 1
        self._usage = usage
        self._target = target
        self._ids: list = None
        self._name = name
        self._vao: int = VAO.CURRENT_ID()  # 记录 vbo 所在的 vao

        if not self._expandable:
            offset = 0
            self._ids = [gl.glGenBuffers(1)]

            for block in self._blocks:
                block.buffer_id = self._ids[0]
                block.offset = offset
                block.require_expand = False
                offset += block.nbytes
            gl.glBindBuffer(self._target, self._ids[0])
            gl.glBufferData(self._target, self.nbytes, None, self._usage)  # 只分配一次内存

        else :  # 每个数据块对应一个缓冲区
            _ids = gl.glGenBuffers(len(self._blocks))
            self._ids = _ids if len(self._blocks)>1 else [_ids]
            for i, block in enumerate(self._blocks):
                block.buffer_id = self._ids[i]
                gl.glBindBuffer(self._target, self._ids[i])
                gl.glBufferData(self._target, 2, None, self._usage)

        if name:
            print(f"[{self._name}] created, vao: {self._vao}, buffers: {self._ids}")
        self.commit()

    @property
    def nbytes(self):
        return sum([b.nbytes for b in self._blocks])

    def commit(self):
        """
        将 pending 的数据块上传到缓冲区, 若数据块需要扩容, 则扩容
        如果某个属性为默认属性不记录在 VAO 中, 则每次绘画之前需要设置属性指针
        """
        VAO.bind_(self._vao)
        for block in self._blocks:

            # 检查是否需要更新数据
            if block.pending:
                # log info
                if self._name:
                    print(f"[{self._name}] commit buffer: {block.buffer_id}, offset: {block.offset}, commited: {block.commited_count}, count: {block.count}, vao: {self._vao}, curr vao: {VAO.CURRENT_ID()}")

                if not self._expandable and block.require_expand:
                    raise ValueError("Data block can not be expanded")

                gl.glBindBuffer(self._target, block.buffer_id)
                if self._expandable and block.require_expand:
                    gl.glBufferData(self._target, block.nbytes, block._data, self._usage)
                    block.require_expand = False
                else:
                    gl.glBufferSubData(
                        self._target,
                        block.offset + block._commited_nbytes,
                        block._used_nbytes - block._commited_nbytes,
                        block.data[block.commited_count : block.count].copy()
                    )
                block._commited_nbytes = block._used_nbytes  # 更新指针

            block.check_attr_state()
        VAO.unbind()

    def setAttrPointer(self, idx: IntOrList, attr_id: IntOrList=None, divisor: IntOrList=0):
        """
        设置属性指针

        Parameters:
        - idx : IntOrList, 数据块编号
        - attr_id : IntOrList, optional, default: None, 属性编号
        - divisor : IntOrList, optional, default: 0
        """
        idx = idx if isinstance(idx, list) else [idx]
        attr_id = attr_id if isinstance(attr_id, list) else [attr_id]
        divisor = divisor if isinstance(divisor, list) else [divisor] * len(idx)

        for i, a_id, div in zip(idx, attr_id, divisor):
            gl.glBindBuffer(self._target, self._blocks[i].buffer_id)
            self._blocks[i].setAttrPointer(a_id, div)

    def getData(self, idx: int) -> np.ndarray:
        """
        从缓冲区中获取第 idx 块数据

        Parameters:
        - idx : int, 数据块编号

        Returns:
        - np.ndarray
        """
        block = self._blocks[idx]
        gl.glBindBuffer(self._target, block.buffer_id)

        data = np.empty(block.used_size, dtype=block.dtype)
        gl.glGetBufferSubData(
            self._target,
            block.offset,
            block.used_nbytes,
            data.ctypes.data_as(c_void_p)
        )
        return data.reshape(block.data.shape)

    def size(self, idx=0) -> int:
        """
        元素个数, idx 为数据块编号
        """
        return self._blocks[idx].used_size

    def count(self, idx=0) -> int:
        """
        已经提交的属性个数, idx 为数据块编号, 如 vec3 类型为 size // 3
        """
        return self._blocks[idx].commited_count

    def __getitem__(self, idx: int) -> GLDataBlock:
        """
        返回第 idx 块数据的信息
        """
        return self._blocks[idx]

    # def __del__(self):
    #     gl.glDeleteBuffers(1, np.array(self._ids))


class VBO(DataBufferObject):

    def __init__(
        self,
        data_blocks: Union[List[GLDataBlock], GLDataBlock],
        expandable = False,
        usage = None,
        name = None
    ):
        """
        初始化一个 Vertex Buffer Object 管理对象

        Parameters:
        - data_blocks : Union[List[GLDataBlock], GLDataBlock], 管理的数据块
        - expandable : bool, optional, default: False, 数据块是否可扩大, 若为 False, 则所有数据块在一个缓冲区中顺序存储, 每个数据块的长度在初始化时确定, 不可扩展; 若为 True, 则每个数据块对应一个缓冲区, 可以扩展容量, 适合动态更新数据
        - usage : 枚举值, optional, default: None, 缓冲区使用方式, 若为 None, 则根据 expandable 自动选择 gl.GL_STATIC_DRAW 或 gl.GL_DYNAMIC_DRAW
        - name : str, optional, default: None, 缓冲区名称, 若设置, 则输出相关日志
        """
        if usage is None:
            usage = gl.GL_STATIC_DRAW if not expandable else gl.GL_DYNAMIC_DRAW

        super().__init__(data_blocks, expandable, usage, gl.GL_ARRAY_BUFFER, name)


class EBO(DataBufferObject):

    def __init__(
        self,
        data_block: GLDataBlock,
        usage = gl.GL_STATIC_DRAW,
        name = None
    ):
        """
        初始化一个 Element Buffer Object 管理对象
        **注意: 调用 EBO.commit() 时, 确保相应的 VAO 已经绑定, 否则可能会影响其他的 VAO**

        Parameters:
        - data_block : GLDataBlock, 索引数据块
        - usage : 枚举值, optional, default: gl.GL_STATIC_DRAW, 缓冲区使用方式
        - name : str, optional, default: None, 缓冲区名称, 若设置, 则输出相关日志
        """
        if data_block.dtype != np.uint32:
            print("[Warning] EBO data type should be np.uint32")
            data_block.dtype = np.uint32
        data_block._layout = [1]

        super().__init__(data_block, True, usage, gl.GL_ELEMENT_ARRAY_BUFFER, name)

    @property
    def isbind(self) -> bool:
        return self._ids[0] == gl.glGetIntegerv(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING)

    def commit(self):
        """
        将 pending 的数据块上传到缓冲区, 确保相应的 VAO 已经绑定
        """
        super().commit()


class VAO:

    _CURRENT_ID_STACK = [0]  # 用栈记录当前绑定的 VAO, 因为有时会有嵌套 with vao 的情况

    @classmethod
    def CURRENT_ID(cls):
        return cls._CURRENT_ID_STACK[-1]

    def __init__(self):
        self._id = gl.glGenVertexArrays(1)

    @property
    def id(self):
        return self._id

    @property
    def isbind(self):
        return self._id == gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)

    def bind(self):
        """
        绑定 self._id, 并将当前 id 压入栈, 必须紧跟一个 unbind 或者 __exit__ 方法
        """
        if self._id != VAO._CURRENT_ID_STACK[-1]:
            gl.glBindVertexArray(self._id)
        VAO._CURRENT_ID_STACK.append(self._id)

    @classmethod
    def bind_(cls, vao_id):
        """
        绑定 vao_id, 并将当前 id 压入栈, 必须紧跟一个 unbind 或者 __exit__ 方法
        """
        if vao_id != cls._CURRENT_ID_STACK[-1]:
            gl.glBindVertexArray(vao_id)
        cls._CURRENT_ID_STACK.append(vao_id)


    @classmethod
    def unbind(cls):
        if len(cls._CURRENT_ID_STACK) > 1:
            id = cls._CURRENT_ID_STACK.pop()
        else :
            id = 0 # 保留一个默认的 VAO
        if id != cls._CURRENT_ID_STACK[-1]:
            gl.glBindVertexArray(cls._CURRENT_ID_STACK[-1])

    def __enter__(self):
        self.bind()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unbind()