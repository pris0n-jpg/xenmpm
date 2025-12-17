from enum import Enum

class ConnectionType(Enum):
    Local = 1
    Virtual = 2
    Remote = 3

class SensorType(Enum):
    Omni = 1
    VecTouch = 2
    Finger = 3
    OmniB = 4

class OutputType(Enum):
    Raw = 1000
    Rectify = 1
    Difference = 2  # ImgObjEnhance
    Depth = 3
    Marker2D = 4
    Marker3D = 5
    Marker3DFlow = 6
    Marker3DInit = 7
    MarkerUnorder = 8
    Force = 9
    ForceResultant = 16
    ForceNorm = 10
    Mesh3D = 11
    Mesh3DFlow = 12
    Mesh3DInit = 13
    Marker2DInit = 14
    Marker2DFlip = 15
    # -- Private Types
    Mesh3DNorm = 101
    Flow = 103
    ImgFloat = 104
    ImgMarkerFree = 105
    ImgMarkerEnhance = 106
    ImgObjEnhance = 107


class InferType(Enum):
    ONNX = 0
    Torch = 1
    RKNN = 2
    NoInfer = 3  # 无需推理


class DAGType(Enum):
    Split = 0
    AllInOne = 1


class MachineType(Enum):
    X86 = 0
    Jetson = 1
    RK3588 = 2
    RDK_X5 = 3
    RK3576 = 4

class CameraSource(Enum):
    CV2_MSMF = 1    #WIN
    CV2_DSHOW = 2   #WIN
    CV2_V4L2 = 3    #LINUX
    AV_V4L2 = 4     #LINUX