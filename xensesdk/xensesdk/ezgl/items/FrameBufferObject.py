import OpenGL.GL as gl
from .texture import Texture2D


class FBO:

    class Type:
        RGB = 0
        DEPTH = 1
        RGBD =2
        STENCIL = 3
        DEPTH_STENCIL = 4

    def __init__(
        self,
        width=1600,
        height=1600,
        type: "FBO.Type" = 1
    ) -> None:
        self._id = gl.glGenFramebuffers(1)
        self._width, self._height = width, height
        self._last_viewport = None
        self._last_buffer_id = 1
        self._type = type

        self._rgb_texture = None
        self._depth_texture = None

        self.bind()

        if type in [self.Type.RGB, self.Type.RGBD]:
            self._rgb_texture = self.create_rgb_texture(width, height)
            gl.glFramebufferTexture2D(   # 将纹理附加到帧缓冲对象
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._rgb_texture.id, 0
            )

        if type == self.Type.RGB:
            # 创建一个深度渲染缓冲附件, 用于深度测试
            depth_rbo = gl.glGenRenderbuffers(1)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rbo)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, width, height)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rbo)

            # if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            #     print('Error: Framebuffer is not complete!')

        if type in [self.Type.DEPTH, self.Type.RGBD]:
            self._depth_texture = self.create_depth_texture(width, height)
            gl.glFramebufferTexture2D(   # 将纹理附加到帧缓冲对象
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._depth_texture.id, 0
            )

        if type == self.Type.DEPTH:
            gl.glDrawBuffer(gl.GL_NONE)  # 禁用颜色附件
            gl.glReadBuffer(gl.GL_NONE)  # 禁用读取颜色附件

        self.unbind()

    @property
    def depth_texture(self) -> Texture2D:
        """
        深度纹理

        Returns:
        - Texture2D, 值的范围在[0, 1]之间, 0表示近平面, 1表示远平面
        """
        return self._depth_texture

    @property
    def rgb_texture(self) -> Texture2D:
        """
        RGBA纹理

        Returns:
        - Texture2D,
        """
        return self._rgb_texture

    def bind(self):
        self._last_buffer_id = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)   # 记录当前绑定的帧缓冲对象
        self._last_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._id)
        gl.glViewport(0, 0, self._width, self._height)

    def unbind(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._last_buffer_id)
        gl.glViewport(*self._last_viewport)

    def __enter__(self):
        self.bind()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An exception occurred: {exc_type}: {exc_value}")
        self.unbind()

    def resize(self, width, height):
        """
        resize the framebuffer

        Parameters:
        - width : int, the width of the framebuffer
        - height : int, the height of the framebuffer
        """
        self._width, self._height = width, height
        if self._depth_texture is not None:
            self._depth_texture.resize(width=width, height=height)
        if self._rgb_texture is not None:
            self._rgb_texture.resize(width=width, height=height)

    def delete(self):
        if self._depth_texture is not None:
            self._depth_texture.delete()
        if self._rgb_texture is not None:
            self._rgb_texture.delete()
        gl.glDeleteFramebuffers(1, [self._id])

    @staticmethod
    def create_rgb_texture(width, height) -> Texture2D:
        """
        create a rgb texture
        """
        texture = Texture2D(
            None, None, gl.GL_NEAREST, gl.GL_NEAREST, gl.GL_REPEAT, gl.GL_REPEAT,
            width = width,
            height = height,
            internal_format = gl.GL_RGB16F,
            format = gl.GL_RGB,
            data_type = gl.GL_HALF_FLOAT
        )
        texture.bind()
        texture.uploadTexture()  # 分配内存
        return texture

    @staticmethod
    def create_depth_texture(width, height) -> Texture2D:
        """
        create a depth texture

        Parameters:
        - width : int, the width of the texture
        - height : int, the height of the texture
        """
        texture = Texture2D(
            None, None, gl.GL_NEAREST, gl.GL_NEAREST, gl.GL_CLAMP_TO_BORDER, gl.GL_CLAMP_TO_BORDER,
            width = width,
            height = height,
            internal_format = gl.GL_DEPTH_COMPONENT,
            format = gl.GL_DEPTH_COMPONENT,
            data_type = gl.GL_FLOAT
        )
        texture.bind()
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
        texture.uploadTexture()  # 分配内存
        return texture