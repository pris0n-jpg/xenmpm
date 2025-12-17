import OpenGL.GL as gl
from PIL import Image
import numpy as np
from typing import Union, Optional
from threading import Lock
from ..functions import NumberPool, LoopRange

__all__ = ['Texture2D']

# 纹理单元池
# TextureUnitPool = NumberPool(32)  # 基于堆的单元池
class TextureUnitPool:
    looprange = LoopRange(0, 32)

    @classmethod
    def gen_num(cls):
        return next(cls.looprange)


class Texture2D:

    Format = {
        1 : gl.GL_RED,
        2 : gl.GL_RG,
        3 : gl.GL_RGB,
        4 : gl.GL_RGBA,
    }

    InternalFormat = {
        (1, 'uint8') : gl.GL_R8,  # 归一化
        (2, 'uint8') : gl.GL_RG8,
        (3, 'uint8') : gl.GL_RGB8,
        (4, 'uint8') : gl.GL_RGBA8,
        (1, 'float32') : gl.GL_R32F,
        (2, 'float32') : gl.GL_RG32F,
        (3, 'float32') : gl.GL_RGB32F,
        (4, 'float32') : gl.GL_RGBA32F,
        (1, 'float16') : gl.GL_R16F,
        (2, 'float16') : gl.GL_RG16F,
        (3, 'float16') : gl.GL_RGB16F,
        (4, 'float16') : gl.GL_RGBA16F,
    }

    DataType = {
        'uint8' : gl.GL_UNSIGNED_BYTE,
        'float16' : gl.GL_HALF_FLOAT,
        'float32' : gl.GL_FLOAT,
    }

    def __init__(
        self,
        source: Union[str, np.ndarray] = None,
        tex_type: Optional[str] = "tex_diffuse",
        mag_filter = gl.GL_LINEAR,
        min_filter = gl.GL_LINEAR_MIPMAP_LINEAR,
        wrap_s = gl.GL_REPEAT,
        wrap_t = gl.GL_REPEAT,
        flip_y = False,
        flip_x = False,
        width = None,
        height = None,
        internal_format = None,
        format = None,
        data_type = None
    ):
        self._id = None

        # if the texture image is updated, the flag is set to True,
        # meaning that the texture needs to be updated to the GPU.
        self._pending = False
        self._img = None  # the texture image

        self.flip_y = flip_y
        self.flip_x = flip_x
        self.mag_filter = mag_filter
        self.min_filter = min_filter
        self.wrap_s = wrap_s
        self.wrap_t = wrap_t
        self.type = tex_type
        self.generate_mipmaps = min_filter == gl.GL_LINEAR_MIPMAP_LINEAR

        self.width = width
        self.height = height
        self.internal_format = internal_format  # 纹理数据在 GPU 端的格式, 如 gl.GL_RGB8, gl.GL_RGBA32F
        self.format = format  # 你提供的纹理数据的格式, 如 gl.GL_RGB, gl.GL_RGBA
        self.data_type = data_type  # 你提供的纹理数据的数据类型, 如 gl.GL_UNSIGNED_BYTE, gl.GL_FLOAT

        self.setTexture(source)

    @property
    def img(self):
        return self._img

    @property
    def id(self):
        return self._id

    def setTexture(self, img: Union[str, np.ndarray]):
        """
        更新纹理图像数据，如果 img 是字符串，则表示文件路径，否则表示图像数据
        可以在未绑定纹理的情况下更新纹理数据, 更新完成后数据存储在 cpu 端, pending 标志位设置为 True, 表示需要更新到 gpu 端

        Parameters:
        - img : Union[str, np.ndarray], 图像数据或文件路径
        """
        if img is None:
            return

        if not isinstance(img, np.ndarray):
            self._path = str(img)
            img = np.array(Image.open(self._path))

        self._img = flip_image(img, self.flip_x, self.flip_y)

        # 更新 internal_format, format, data_type, width, height
        channels = 1 if self._img.ndim==2 else self._img.shape[2]
        dtype = self._img.dtype.name
        self.internal_format = self.InternalFormat[(channels, dtype)]
        self.format = self.Format[channels]
        self.data_type = self.DataType[dtype]
        self.width = self._img.shape[1]
        self.height = self._img.shape[0]
        # 设置 pending 标志位
        self._pending = True

    def _initialize(self):
        """
        初始化纹理对象, 设置纹理参数
        """
        self._id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._id)
        # 设置纹理参数
        # -- texture wrapping
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, self.wrap_s)
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, self.wrap_t)
        # -- texture filtering
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, self.min_filter)
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, self.mag_filter)

    def uploadTexture(self):
        """
        上传纹理数据到 GPU, 需先绑定纹理, 如果 _img 为 None, 则只为纹理分配内存, 不上传数据
        """
        # -- set alignment, 检查每行的字节数是否是 4 的倍数
        if self._img is not None:
            channels = 1 if self._img.ndim==2 else self._img.shape[2]
            nbytes_row = self._img.shape[1] * self._img.dtype.itemsize * channels
            if nbytes_row % 4 != 0:
                gl.glPixelStorei( gl.GL_UNPACK_ALIGNMENT, 1)
            else:
                gl.glPixelStorei( gl.GL_UNPACK_ALIGNMENT, 4)

        # 上传纹理数据
        gl.glTexImage2D(            # 创建一个纹理, 并分配内存
            gl.GL_TEXTURE_2D,       # target: 纹理目标
            0,                      # level: 表示多级渐远纹理的级别, 0 表示当前定义的是原级别的纹理, 1表示正在定义原始一半大小的纹理, 以此类推
            self.internal_format,   # internal format: 指定opengl以哪种格式储存纹理, 例如 GL_RGB, GL_RGBA, GL_DEPTH_COMPONENT
            self.width,             # width
            self.height,            # height
            0,                      # border: 边框的宽度
            self.format,            # format: 指定你提供的像素数据格式, 主要是通道数以及每个通道的含义
            self.data_type,         # type: 像素数据的数据类型
            self._img               # data: 图像数据
        )

        self._pending = False

    def resize(self, width: int, height: int):
        """
        重设纹理大小, 重新分配内存
        """
        self.bind()
        self.width, self.height = width, height
        self._img = None
        self.uploadTexture()

    def bind(self):
        """
        绑定纹理对象, 若纹理对象未初始化, 则初始化纹理对象
        """
        if self._id is None:
            self._initialize()
        else:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._id)

    def bindTexUnit(self) -> int:
        """
        获取一个纹理单元并绑定纹理, 如果纹理需要更新, 则更新纹理, 最后返回纹理单元
        注意: 如果使用基于堆的单元池，每次在使用一个 shader 之前会重置纹理单元池, 确保纹理单元不会被占用, 因此纹理绑定应该在 shader 使用之后.
            如果使用 LoopRange 则没有这个限制, 但仍推荐在 shader 使用之后绑定纹理, 这可以确保已经正确设置了 opengl context
        Returns:
        - int, 纹理单元
        """
        # if self._img is None:
        #     raise ValueError('Texture not initialized.')

        # 获取并激活纹理单元
        last_unit = gl.glGetIntegerv(gl.GL_ACTIVE_TEXTURE)
        unit = TextureUnitPool.gen_num()
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)

        # 绑定纹理
        self.bind()

        # 更新纹理数据
        if self._pending:
            self.uploadTexture()

            # 若 min_filter 为 gl.GL_LINEAR_MIPMAP_LINEAR, 则生成多级渐远纹理
            if self.generate_mipmaps:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        gl.glActiveTexture(last_unit)

        return unit

    def getTexture(self) -> np.ndarray:
        """
        获取纹理数据

        Returns:
        - np.ndarray, 纹理图片
        """
        self.bind()
        img = gl.glGetTexImage(
            gl.GL_TEXTURE_2D,       # tartget
            0,                      # level
            self.format,            # format
            self.data_type          # type
        )
        return np.flip(img.reshape(self.height, self.width, -1), 0)

    def delete(self):
        if self._id is not None:
            gl.glDeleteTextures([self._id])
            self._id == None


def flip_image(img, flip_x=False, flip_y=False):
    if flip_x and flip_y:
        img = np.flip(img, (0, 1))
    elif flip_x:
        img = np.flip(img, 1)
    elif flip_y:
        img = np.flip(img, 0)
    return img


class Texture1D:

    def __init__(
        self,
        source: np.ndarray = None,
        mag_filter=gl.GL_LINEAR,
        min_filter=gl.GL_LINEAR,
        wrap_s=gl.GL_REPEAT,
        flip=False,
        width=None,
        internal_format=None,
        format=None,
        data_type=None
    ):
        self._id = None
        self._pending = False
        self._img = None

        self.flip = flip
        self.mag_filter = mag_filter
        self.min_filter = min_filter
        self.wrap_s = wrap_s
        self.generate_mipmaps = min_filter == gl.GL_LINEAR_MIPMAP_LINEAR

        self.width = width
        self.internal_format = internal_format
        self.format = format
        self.data_type = data_type

        self.setTexture(source)

    @property
    def img(self):
        return self._img

    @property
    def id(self):
        return self._id

    def setTexture(self, img: np.ndarray):
        if img is None:
            return

        self._img = flip_image(img, False, self.flip)

        channels = 1 if self._img.ndim == 1 else self._img.shape[1]
        dtype = self._img.dtype.name
        self.internal_format = Texture2D.InternalFormat[(channels, dtype)]
        self.format = Texture2D.Format[channels]
        self.data_type = Texture2D.DataType[dtype]
        self.width = self._img.shape[0]
        self._pending = True

    def _initialize(self):
        self._id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_1D, self._id)
        gl.glTexParameter(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, self.wrap_s)
        gl.glTexParameter(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, self.min_filter)
        gl.glTexParameter(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, self.mag_filter)

    def uploadTexture(self):
        if self._img is not None:
            channels = 1 if self._img.ndim == 1 else self._img.shape[1]
            nbytes_row = self._img.shape[0] * self._img.dtype.itemsize * channels
            if nbytes_row % 4 != 0:
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            else:
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)

        gl.glTexImage1D(
            gl.GL_TEXTURE_1D,
            0,
            self.internal_format,
            self.width,
            0,
            self.format,
            self.data_type,
            self._img
        )

        self._pending = False

    def resize(self, width: int):
        self.bind()
        self.width = width
        self._img = None
        self.uploadTexture()

    def bind(self):
        if self._id is None:
            self._initialize()
        else:
            gl.glBindTexture(gl.GL_TEXTURE_1D, self._id)

    def bindTexUnit(self) -> int:
        unit = TextureUnitPool.gen_num()
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        self.bind()

        if self._pending:
            self.uploadTexture()

            if self.generate_mipmaps:
                gl.glGenerateMipmap(gl.GL_TEXTURE_1D)

        return unit

    def getTexture(self) -> np.ndarray:
        self.bind()
        img = gl.glGetTexImage(gl.GL_TEXTURE_1D, 0, self.format, self.data_type)
        return img.reshape(self.width, -1)

    def delete(self):
        if self._id is not None:
            gl.glDeleteTextures([self._id])
            self._id = None