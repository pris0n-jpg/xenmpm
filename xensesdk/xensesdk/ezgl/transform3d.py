import numpy as np
from math import acos, degrees, atan2, asin, radians, cos, sin
from typing import Any, Union, List, Tuple, Sequence
from .functions import dispatchmethod
from qtpy.QtGui import QQuaternion, QMatrix4x4, QVector3D, QMatrix3x3, QVector4D


def mat33ToEuler(mat33: np.ndarray) -> np.ndarray:
    """ zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] """
    if mat33[2, 0] > 0.9999:
        x = 0
        y = -np.pi/2
        z = atan2(-mat33[0, 1], -mat33[0, 2])
    elif mat33[2, 0] < -0.9999:
        x = 0
        y = np.pi/2
        z = atan2(-mat33[0, 1], mat33[0, 2])
    else:
        x = atan2(mat33[2, 1], mat33[2, 2])
        y = atan2(-mat33[2, 0], np.sqrt(mat33[2, 1]**2 + mat33[2, 2]**2))
        z = atan2(mat33[1, 0], mat33[0, 0])
    angle = np.rad2deg(np.array([x, y, z]))
    eq_angle = np.array([angle[0]-180, 180-angle[1], angle[2]-180])
    if eq_angle[0] < -180:
        eq_angle[0] += 360
    if eq_angle[1] > 180:
        eq_angle[1] -= 360
    if eq_angle[2] < -180:
        eq_angle[2] += 360
    if np.linalg.norm(eq_angle) < np.linalg.norm(angle):
        return eq_angle
    return angle

def eulerToMat33(x, y, z) -> np.ndarray:
    "zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] "
    cx, sx = cos(radians(x)), sin(radians(x))
    cy, sy = cos(radians(y)), sin(radians(y))
    cz, sz = cos(radians(z)), sin(radians(z))
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


class Quaternion(QQuaternion):
    """
    Extension of QQuaternion with some helpful methods added.
    """

    # constructors
    @dispatchmethod
    def __init__(self):
        super().__init__()

    @__init__.register(np.ndarray)
    def _(self, array: np.ndarray):
        quat_values = array.astype('f4').flatten()
        super().__init__(*quat_values)

    @__init__.register(tuple)
    @__init__.register(list)
    def _(self, array):
        super().__init__(*array)

    @__init__.register(QQuaternion)
    def _(self, quat):
        super().__init__(quat.scalar(), quat.x(), quat.y(), quat.z())

    def __repr__(self) -> str:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        return f"Quaternion(w: {self.scalar()}, x: {self.x()}, y: {self.y()}, z: {self.z()})"

    def copy(self):
        return Quaternion(self)

    def conjugate(self):
        """conjugate of quaternion"""
        return Quaternion(super().conjugated())

    def inverse(self):
        """conjugate of quaternion"""
        return Quaternion(super().inverted())

    def normalize(self):
        return Quaternion(super().normalized())

    def toEulerAngles(self):
        """ zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] """
        mat33 = self.toRotationMatrix()
        return mat33ToEuler(mat33)

    def toRotationMatrix(self) -> np.ndarray:
        mat33 = super().toRotationMatrix()
        return np.array(mat33.data()).reshape(3, 3).T

    def toMatrix4x4(self):
        matrix3x3 = super().toRotationMatrix()
        matrix = np.identity(4)
        matrix[:3, :3] = np.array(matrix3x3.data()).reshape(3, 3).T
        return Matrix4x4(matrix)

    @classmethod
    def fromEulerAngles(cls, x, y, z):
        "zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] "
        return cls.fromMatrix3x3(eulerToMat33(x, y, z))

    @classmethod
    def fromEulerAnglesOrder(cls, x, y, z, order=[2, 1, 0]):
        " euler system (degree), default zyx system "
        cx, sx = cos(radians(x)), sin(radians(x))
        cy, sy = cos(radians(y)), sin(radians(y))
        cz, sz = cos(radians(z)), sin(radians(z))
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        rots = [rot_x, rot_y, rot_z]
        mat33 = rots[order[0]] @ rots[order[1]] @ rots[order[2]]
        return cls.fromMatrix3x3(mat33)

    @classmethod
    def fromAxisAndAngle(cls, x=0., y=0., z=0., angle=0.):
        return cls(QQuaternion.fromAxisAndAngle(x, y, z, angle))

    @classmethod
    def fromMatrix4x4(cls, matrix: 'Matrix4x4'):
        matrix3x3 = QMatrix3x3(matrix.matrix33.flatten())
        return cls(QQuaternion.fromRotationMatrix(matrix3x3))

    @classmethod
    def fromMatrix3x3(cls, matrix: np.ndarray):
        matrix3x3 = QMatrix3x3(matrix.flatten())
        return cls(QQuaternion.fromRotationMatrix(matrix3x3))

    # operators
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(super().__mul__(other))
        # elif isinstance(other, Matrix4x4):
        #     mat = self.toMatrix4x4()
        #     return mat * other
        elif isinstance(other, QVector3D):
            return super().__mul__(other)

        # apply rotation to vectors np.array([[x1,y1,z1], [x2,y2,z2], ...])
        mat = self.toMatrix4x4()
        return mat * other

    def slerp(self, other, t) -> 'Quaternion':
        """
        计算两个四元数之间的插值点, t in [0, 1]

        Parameters:
        - other : Quaternion,
        - t : float in [0, 1]
        """
        return Quaternion(super().slerp(self, other, t))

    def dot(self, other) -> float:
        return super().dotProduct(self, other)


class Matrix4x4(QMatrix4x4):
    """
    Extension of QMatrix4x4 with some helpful methods added.
    """

    # constructors
    @dispatchmethod
    def __init__(self):
        super().__init__()

    @__init__.register(list)
    @__init__.register(tuple)
    @__init__.register(np.ndarray)
    def _(self, array):
        """row-major order"""
        matrix_values = np.array(array).astype('f4').flatten()
        super().__init__(*matrix_values)

    @__init__.register(QMatrix4x4)
    def _(self, matrix):
        super().__init__(matrix.copyDataTo())

    def copy(self):
        return Matrix4x4(self)

    def __len__(self):
        return 16

    def __getitem__(self, i):
        if isinstance(i, tuple):
            return super().__getitem__(i)
        return self.data()[i]

    def __setitem__(self, i, value):
        row, col = i
        rowdata = self.row(row)
        if col == 0:
            rowdata.setX(value)
        elif col == 1:
            rowdata.setY(value)
        elif col == 2:
            rowdata.setZ(value)
        elif col == 3:
            rowdata.setW(value)
        self.setRow(row, rowdata)

    @property
    def matrix33(self):
        m = np.array(self.copyDataTo()).reshape(4,4)
        return m[:3,:3]

    @property
    def matrix44(self):
        return np.array(self.copyDataTo()).reshape(4,4)

    @property
    def array(self):
        return np.array(self.copyDataTo()).reshape(16)

    @property
    def xyz(self):
        trans = self.column(3)
        return np.array([trans.x(), trans.y(), trans.z()])

    @property
    def quat(self):
        """convert to quaternion"""
        return Quaternion.fromMatrix4x4(self)

    @property
    def euler(self):
        """degree"""
        return self.toEulerAngles()

    @property
    def glData(self):
        """convert to column-major order for use with OpenGL"""
        return self.data()

    @classmethod
    def dist(cls, m1, m2):
        return (m1.column(3) - m2.column(3)).length()

    def __repr__(self) -> str:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        return  f"Matrix4x4(\n{self.matrix44}\n)"

    @dispatchmethod
    def rotate(self, q:Quaternion, local=True) -> "Matrix4x4":
        """rotate by quaternion in local space, it will change the current matrix"""
        if local:
            super().rotate(q)
        else:
            self.setData(q * self)
        return self

    @rotate.register(float)
    @rotate.register(int)
    def _(self, angle, x, y, z, local=True):
        """rotate by angle(degree) around x,y,z in local space, it will change the current matrix"""
        if local:
            super().rotate(angle, x, y, z)
        else:
            self.setData(Matrix4x4.fromAxisAndAngle(x, y, z, angle) * self)
        return self

    def translate(self, x, y, z, local=True):
        """translate by x,y,z in local space, it will change the current matrix"""
        if local:
            super().translate(x, y, z)
        else:
            self.setData(Matrix4x4.fromTranslation(x, y, z) * self)
        return self

    def scale(self, x, y, z, local=True):
        """scale by x,y,z in local space, it will change the current matrix"""
        if local:
            super().scale(x, y, z)
        else:
            self.setData(Matrix4x4.fromScale(x, y, z) * self)
        return self

    def moveto(self, x, y, z):
        """move to x,y,z in world space, it will change the current matrix"""
        self.setColumn(3, QVector4D(x, y, z, 1))
        return self

    def setData(self, matrix: Union[np.ndarray, QMatrix4x4]):
        """set matrix data"""
        for i in range(4):
            self.setRow(i, matrix.row(i))

    def applyTransform(self, matrix, local=False):
        """apply transform to self"""
        if local:
            return self * matrix
        else:
            return matrix * self

    @classmethod
    def fromRotTrans(cls, R: np.ndarray, t=None):
        """rotate by R(3x3) and translate by t"""
        if t is None:
            t = np.zeros(3, dtype='f4')
        data = np.zeros((4,4), dtype='f4')
        data[:3,:3] = R
        data[:3,3] = t
        data[3,3] = 1
        return cls(data)

    @classmethod
    def fromEulerAngles(cls, x, y, z):
        """ zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] """
        mat44 = np.identity(4)
        mat44[:3, :3] = eulerToMat33(x, y, z)
        return cls(mat44)

    @classmethod
    def fromTranslation(cls, x=0., y=0., z=0.):
        """translate by x,y,z"""
        return cls().moveto(x, y, z)

    @classmethod
    def fromScale(cls, x=1., y=1., z=1.):
        """scale by x,y,z"""
        return cls().scale(x, y, z)

    @classmethod
    def fromAxisAndAngle(cls, x=0., y=0., z=0., angle=0.):
        """rotate by angle(degree) around x,y,z"""
        if angle==0 or x==0 and y==0 and z==0:
            return cls()
        return cls().rotate(angle, x, y, z)

    @classmethod
    def fromQuaternion(cls, q):
        """rotate by quaternion"""
        return cls().rotate(q)

    @classmethod
    def fromVector6d(cls, x, y, z, a, b, c) -> "Matrix4x4":
        """ zyx euler system (degree): Rot(z) * Rot(y) * Rot(x), [degrees] """
        return cls.fromEulerAngles(a, b, c).moveto(x, y, z)

    @classmethod
    def fromVector7d(cls, x, y, z, qw, qx, qy, qz) -> "Matrix4x4":
        """rotate by quaternion"""
        return cls.fromQuaternion(Quaternion([qw, qx, qy, qz])).moveto(x, y, z)

    def inverse(self):
        mat, ret = self.inverted()
        assert ret, "matrix is not invertible"
        return Matrix4x4(mat)

    def transpose(self) -> 'Matrix4x4':
        """transpose the matrix"""
        return Matrix4x4(super().transposed())

    def toQuaternion(self):
        """convert to quaternion"""
        return Quaternion.fromMatrix4x4(self)

    def toEulerAngles(self):
        """degree"""
        return mat33ToEuler(self.matrix33)

    def toTranslation(self):
        trans = self.column(3)
        return np.array([trans.x(), trans.y(), trans.z()])

    def toVector6d(self):
        """zyx euler system (degree), return [x, y, z, pitch, yaw, roll]"""
        t = self.toTranslation()
        abc = self.toEulerAngles()
        return np.array([*t, *abc])

    def toVector7d(self):
        """return list [x, y, z, qw, qx, qy, qz]"""
        t = self.toTranslation()
        q = self.toQuaternion()
        return np.array([*t, q.scalar(), q.x(), q.y(), q.z()])

    def __mul__(self, other):
        if isinstance(other, Matrix4x4):
            return Matrix4x4(super().__mul__(other))
        elif isinstance(other, Quaternion):
            mat = Matrix4x4(self)
            mat.rotate(other)
            return mat

        if isinstance(other, (list, tuple)):
            other = np.array(other, dtype='f4')
        elif isinstance(other, Vector3):
            other = other.xyz
        assert isinstance(other, np.ndarray), f"unsupported type {type(other)}"

        # apply rotation to vectors v = np.array([[x1,y1,z1], [x2,y2,z2], ...]), n,3
        # v * R.T + t
        origin_shape = other.shape
        if other.ndim == 1:
            other = other[None, :]

        channels = other.shape[1]

        if channels == 3:
            other = np.pad(other, ((0, 0), (0, 1)), 'constant', constant_values=1.0)

        ret = np.matmul(other, self.matrix44.T)
        ret /= np.maximum(ret[:, [3]], 1e-8)
        ret[:, 3] = 1
        if channels == 3:
            return ret[:, :3].reshape(origin_shape)
        else:
            return ret.reshape(origin_shape)

    def interp(self, other:'Matrix4x4', t) -> 'Matrix4x4':
        """
        计算两个点之间的插值点， t in [0, 1]

        Parameters:
        - other : Matrix4x4, 另一个点
        - t : float, 插值系数, other 所占的比例

        Returns:
        - Matrix4x4
        """
        quat : QQuaternion = QQuaternion.slerp(self.quat, other.quat, t)
        xyz = self.xyz * (1-t) + other.xyz * t

        return Matrix4x4().rotate(quat).moveto(*xyz)

    @classmethod
    def perspective(cls, angle: float, aspect: float, near: float, far: float) -> 'Matrix4x4':
        proj = cls()
        super(cls, proj).perspective(angle, aspect, near, far)
        return proj

    @classmethod
    def ortho(cls, left, right, bottom, top, near, far) -> 'Matrix4x4':
        proj = cls()
        super(cls, proj).ortho(left, right, bottom, top, near, far)
        return proj

    @classmethod
    def lookAt(cls, eye: Sequence, center: Sequence, up: Sequence):
        mat = cls()
        super(cls, mat).lookAt(QVector3D(*eye), QVector3D(*center), QVector3D(*up))
        return mat


class Vector3():

    @dispatchmethod
    def __init__(self, x: float = 0., y: float = 0., z: float = 0.):
        self._data = np.array([x, y, z], dtype='f4')

    @__init__.register(np.ndarray)
    @__init__.register(list)
    @__init__.register(tuple)
    def _(self, data):
        self._data = np.array(data, dtype='f4').flatten()[:3]

    @__init__.register(QVector3D)
    def _(self, data):
        self._data = np.array([data.x(), data.y(), data.z()], dtype='f4')

    def copy(self):
        return Vector3(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return 3

    def __repr__(self):
        return f"Vec3({self.x:.3g}, {self.y:.3g}, {self.z:.3g})"

    def __sub__(self, other):
        return Vector3(self._data - other._data)

    def __add__(self, other):
        return Vector3(self._data + other._data)

    def __isub__(self, other):
        self._data -= other._data
        return self

    def __iadd__(self, other):
        self._data += other._data
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self._data * other)
        elif isinstance(other, (Vector3, np.ndarray, list, tuple)):
            return Vector3(self._data[0] * other[0],
                           self._data[1] * other[1],
                           self._data[2] * other[2])
        else:
            raise TypeError(f"unsupported type {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self._data *= other
        elif isinstance(other, (Vector3, np.ndarray, list, tuple)):
            self._data[0] *= other[0]
            self._data[1] *= other[1]
            self._data[2] *= other[2]
        else:
            raise TypeError(f"unsupported type {type(other)}")
        return self

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self._data / other)
        elif isinstance(other, (Vector3, np.ndarray, list, tuple)):
            return Vector3(self._data[0] / other[0],
                           self._data[1] / other[1],
                           self._data[2] / other[2])
        else:
            raise TypeError(f"unsupported type {type(other)}")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            self._data /= other
        elif isinstance(other, (Vector3, np.ndarray, list, tuple)):
            self._data[0] /= other[0]
            self._data[1] /= other[1]
            self._data[2] /= other[2]
        else:
            raise TypeError(f"unsupported type {type(other)}")
        return self

    def __neg__(self):
        return Vector3(-self._data)

    @property
    def xyz(self):
        return self._data

    @property
    def x(self):
        return self._data[0]

    @x.setter
    def x(self, val):
        self._data[0] = val

    @property
    def y(self):
        return self._data[1]

    @y.setter
    def y(self, val):
        self._data[1] = val

    @property
    def z(self):
        return self._data[2]

    @z.setter
    def z(self, val):
        self._data[2] = val

    @property
    def norm(self):
        return np.linalg.norm(self._data)

    def normalize(self, length=1):
        self._data = self._data * (length / self.norm)
        return self

    def __getitem__(self, i):
        if i > 2:
            raise IndexError("Point has no index %s" % str(i))
        return self._data[i]

    def __setitem__(self, i, x):
        self._data[i] = x

    @classmethod
    def fromPolar(cls, r, theta, phi) -> 'Vector3':
        """theta, phi in degrees"""
        theta = np.radians(theta)
        phi = np.radians(phi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return cls(x, y, z)

    def toPolar(self) -> tuple:
        """theta, phi in degrees"""
        r = np.linalg.norm(self._data)
        theta = degrees(acos(self._data[0] / r))
        phi = degrees(acos(self._data[2] / r))
        return r, theta, phi

@Vector3.__init__.register
def _(self, v: Vector3):
    self._data = np.array(v._data, dtype='f4')


if __name__ == '__main__':
    na = np.array([0,1,2,3,
                   4,5,6,7,
                   8,9,10,11,
                   0,0,0,1], dtype=np.float32).reshape(4,4)
    a = Matrix4x4(na)
    nt = Matrix4x4([ 1, 0, 0, 1,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1
    ])
    # nt = Matrix4x4.fromAxisAndAngle(1, 0, 0, 30)
    q = Quaternion.fromAxisAndAngle(0,1,1,34)
    q = Quaternion()
