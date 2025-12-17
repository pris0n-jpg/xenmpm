from OpenGL import GL as gl
from .transform3d import Matrix4x4, Quaternion
from .functions import NumberPool, printExc
import numpy as np
from typing import Union

GLOptions = {
    'opaque': {
        gl.GL_DEPTH_TEST: True,
        gl.GL_BLEND: False,
        # gl.GL_ALPHA_TEST: False,
        gl.GL_CULL_FACE: False,
        'glDepthMask': (gl.GL_TRUE,),
        'glPolygonMode': (gl.GL_FRONT_AND_BACK, gl.GL_FILL),
    },
    'translucent': {
        gl.GL_DEPTH_TEST: True,
        gl.GL_BLEND: True,
        # gl.GL_ALPHA_TEST: False,
        gl.GL_CULL_FACE: False,
        'glDepthMask': (gl.GL_TRUE,),
        'glBlendFunc': (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA),
        'glPolygonMode': (gl.GL_FRONT_AND_BACK, gl.GL_FILL),
    },
    'translucent_cull': {
        gl.GL_DEPTH_TEST: True,
        gl.GL_BLEND: True,
        # gl.GL_ALPHA_TEST: False,
        gl.GL_CULL_FACE: True,
        'glCullFace': (gl.GL_BACK,),
        'glDepthMask': (gl.GL_TRUE,),
        'glBlendFunc': (gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA),
        'glPolygonMode': (gl.GL_FRONT_AND_BACK, gl.GL_FILL),
    },
    'additive': {
        gl.GL_DEPTH_TEST: False,
        gl.GL_BLEND: True,
        # gl.GL_ALPHA_TEST: False,
        gl.GL_CULL_FACE: False,
        'glDepthMask': (gl.GL_TRUE,),
        'glBlendFunc': (gl.GL_SRC_ALPHA, gl.GL_ONE),
        'glPolygonMode': (gl.GL_FRONT_AND_BACK, gl.GL_FILL),
    },
    'ontop': {
        gl.GL_DEPTH_TEST: False,
        gl.GL_BLEND: False,
        # gl.GL_ALPHA_TEST: False,
        gl.GL_CULL_FACE: False,
        'glDepthMask': (gl.GL_FALSE,),
        'glPolygonMode': (gl.GL_FRONT_AND_BACK, gl.GL_FILL),
    },
    'polygon': {
        gl.GL_DEPTH_TEST: True,
        gl.GL_BLEND: False,
        # gl.GL_ALPHA_TEST: False,
        gl.GL_CULL_FACE: False,
        'glDepthMask': (gl.GL_TRUE,),
        'glPolygonMode': (gl.GL_FRONT_AND_BACK, gl.GL_LINE),
    }
}

# 图像标识符资源池, 最多支持200个图形项
GraphicsItemIdPool = NumberPool(200)


class GLGraphicsItem():

    def __init__(
        self,
        parentItem: 'GLGraphicsItem' = None,
        depthValue: int = 0,
    ):
        super().__init__()
        self.__parent: GLGraphicsItem | None = None
        self.__view = None
        self.__children: list[GLGraphicsItem] = list()
        self.__transform = Matrix4x4()
        self.__visible = True
        self.__initialized = False
        self.__glOpts = {}
        self.__depthValue = 0
        self.__label = GraphicsItemIdPool.gen_num()
        self._model_matrix = Matrix4x4()
        self.lights = []
        self.setParentItem(parentItem)
        self.setDepthValue(depthValue)

    @property
    def label(self):
        """唯一标识符"""
        return self.__label

    def setParentItem(self, item: 'GLGraphicsItem'):
        """Set this item's parent in the scenegraph hierarchy."""
        if item is None:
            return
        item.addChildItem(self)

    def addChildItem(self, item: 'GLGraphicsItem'):
        if item is not None and item not in self.__children:
            self.__children.append(item)
            self.__children.sort(key=lambda a: a.depthValue())
            if item.__parent is not None:
                item.__parent.__children.remove(item)
            item.__parent = self

    def parentItem(self):
        """Return a this item's parent in the scenegraph hierarchy."""
        return self.__parent

    def childItems(self):
        """Return a list of this item's children in the scenegraph hierarchy."""
        return self.__children

    def recursiveChildItems(self):
        """Yield this item's children and their children, etc."""
        for child in self.__children:
            yield child
            yield from child.recursiveChildItems()

    def treeItems(self):
        """Yield this item and it's children and their children, etc."""
        yield self
        for child in self.__children:
            yield child
            yield from child.recursiveChildItems()

    def setGLOptions(self, opts: Union[str, dict]):
        """
        Set the OpenGL state options to use immediately before drawing this item.
        (Note that subclasses must call setupGLState before painting for this to work)

        The simplest way to invoke this method is to pass in the name of
        a predefined set of options (see the GLOptions variable):

        ============= ======================================================
        opaque        Enables depth testing and disables blending
        translucent   Enables depth testing and blending
                      Elements must be drawn sorted back-to-front for
                      translucency to work correctly.
        additive      Disables depth testing, enables blending.
                      Colors are added together, so sorting is not required.
        ============= ======================================================

        It is also possible to specify any arbitrary settings as a dictionary.
        This may consist of {'functionName': (args...)} pairs where functionName must
        be a callable attribute of OpenGL.GL, or {GL_STATE_VAR: bool} pairs
        which will be interpreted as calls to glEnable or glDisable(GL_STATE_VAR).

        For example::

            {
                GL_ALPHA_TEST: True,
                GL_CULL_FACE: False,
                'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
            }

        """
        if isinstance(opts, str):
            opts = GLOptions[opts]
        self.__glOpts = opts.copy()

    def updateGLOptions(self, opts: dict):
        """
        Modify the OpenGL state options to use immediately before drawing this item.
        *opts* must be a dictionary as specified by setGLOptions.
        Values may also be None, in which case the key will be ignored.
        """
        self.__glOpts.update(opts)

    def setView(self, v):
        self.__view = v

    def view(self):
        return self.__view

    def setDepthValue(self, value: int):
        """
        Sets the depth value of this item. Default is 0. Range is -1000 to 1000.
        This controls the order in which items are drawn--those with a greater depth value will be drawn later.
        Items with negative depth values are drawn before their parent.
        (This is analogous to QGraphicsItem.zValue)
        The depthValue does NOT affect the position of the item or the values it imparts to the GL depth buffer.
        """
        self.__depthValue = value

    def depthValue(self):
        """Return the depth value of this item. See setDepthValue for more information."""
        return self.__depthValue

    def setTransform(self, tr):
        """Set the local transform for this object.

        Parameters
        ----------
        tr : transform3d.Matrix4x4
            Tranformation from the local coordinate system to the parent's.
        """
        self.__transform = Matrix4x4(tr)

    def resetTransform(self):
        """Reset this item's transform to an identity transformation."""
        self.__transform = Matrix4x4()

    def transform(self, local=True) -> Matrix4x4:
        """
        返回物体相对于父物体的变换矩阵, 若 local 为 False, 则返回物体的世界变换矩阵.
        """
        if local:
            return self.__transform.copy()
        else:
            tf = Matrix4x4(self.__transform)
            parent = self.parentItem()
            while parent is not None:
                tf = parent.transform() * tf
                parent = parent.parentItem()
            return tf

    def setVisible(self, vis, recursive=False):
        """Set the visibility of this item."""
        self.__visible = vis
        if recursive:
            for child in self.recursiveChildItems():
                child.setVisible(vis, recursive=False)
        # self.update()

    def visible(self):
        """Return True if the item is currently set to be visible.
        Note that this does not guarantee that the item actually appears in the
        view, as it may be obscured or outside of the current view area."""
        return self.__visible

    def setupGLState(self):
        """
        This method is responsible for preparing the GL state options needed to render
        this item (blending, depth testing, etc). The method is called immediately before painting the item.
        """
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)  # 默认开启抗锯齿
        for k,v in self.__glOpts.items():
            if v is None:
                continue
            if isinstance(k, str):
                func = getattr(gl, k)
                func(*v)
            else:
                if v is True:
                    gl.glEnable(k)
                else:
                    gl.glDisable(k)

    def initialize(self):
        if self.__initialized:
            return

        try:
            self.initializeGL()
        except:
            printExc()
            print(f"Error while initializing item: {self}, label: {self.label}.")

        self.__initialized = True

    @property
    def isInitialized(self):
        return self.__initialized

    @isInitialized.setter
    def isInitialized(self, value):
        self.__initialized = value

    def update_model_matrix(self) -> Matrix4x4:
        """
        更新模型矩阵, 用于将物体的本地坐标转化为世界坐标.
        在自身或者子物体绘制前调用, 由于绘制时严格按照父子顺序, 所以不需要递归计算.

        Returns:
        - Matrix4x4, 模型矩阵
        """
        if self.__parent is None:
            self._model_matrix = self.transform()
        elif self.__parent.visible():
            self._model_matrix = self.__parent._model_matrix * self.transform()
        else:
            self._model_matrix = self.transform(False)
        return self._model_matrix

    def model_matrix(self) -> Matrix4x4:
        """
        返回模型矩阵, 用于将物体的本地坐标转化为世界坐标, 物体移动后, 使用前需要调用 update_model_matrix 方法.
        """
        return self._model_matrix

    def moveTo(self, x, y, z):
        """
        Move the object to the absolute position (x,y,z) in its parent's coordinate system.
        """
        self.__transform.moveto(x, y, z)

    def applyTransform(self, tr:Matrix4x4, local=False):
        """
        Apply the transform *tr* to this object's local transform.
        """
        if local:
            self.__transform = self.__transform * tr
        else:
            self.__transform = tr * self.__transform
        return self

    def flip(self, x, y, z, local=False):
        """
        Flip the object along the axis specified by (x,y,z).
        """
        v = np.array([x, y, z])
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            raise ValueError("Cannot flip along a zero-length vector.")
        v_hat = v / v_norm
        flip_mat = np.eye(3) - 2 * np.outer(v_hat, v_hat)
        flip_mat = Matrix4x4.fromRotTrans(flip_mat, np.zeros(3))
        if local:
            self.__transform = self.__transform * flip_mat
        else:
            self.__transform = flip_mat * self.__transform
        return self

    def translate(self, dx, dy, dz, local=False):
        """
        Translate the object by (*dx*, *dy*, *dz*) in its parent's coordinate system.
        If *local* is True, then translation takes place in local coordinates.
        """
        self.__transform.translate(dx, dy, dz, local=local)
        return self

    def rotate(self, angle, x, y, z, local=False):
        """
        Rotate the object around the axis specified by (x,y,z).
        *angle* is in degrees.

        """
        self.__transform.rotate(angle, x, y, z, local=local)
        return self

    def scale(self, x, y, z, local=True):
        """
        Scale the object by (*dx*, *dy*, *dz*) in its local coordinate system.
        If *local* is False, then scale takes place in the parent's coordinates.
        """
        self.__transform.scale(x, y, z, local=local)
        return self

    # The following methods must be implemented by subclasses:
    def initializeGL(self):
        """
        Called once in GLViewWidget.paintGL.
        The widget's GL context is made current before this method is called.
        (So this would be an appropriate time to generate lists, upload textures, etc.)
        """
        pass
        # raise NotImplementedError()

    def paint(self, camera):
        """
        Called by the GLViewWidget to draw this item.
        The widget's GL context is made current before this method is called.
        It is the responsibility of the item to set up its own modelview matrix,
        but the caller will take care of pushing/popping.
        """
        pass

    def paintWithShader(self, camera, shader, **kwargs):
        pass


    def __del__(self):
        GraphicsItemIdPool.free_num(self.__label)
