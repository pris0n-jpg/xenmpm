from pathlib import Path
from dataclasses import dataclass
from typing import Union, List, Dict, Tuple, Optional, Sequence
import xml.etree.ElementTree as ET
import numpy as np

from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Quaternion
from .GLModelItem import GLModelItem
from .GLMeshItem import GLMeshItem
from .GLAxisItem import GLAxisItem
from .MeshData import cube, cylinder, sphere, Material

__all__ = ['GLURDFItem', 'GLLinkItem', 'Joint']


rad2deg = 180 / np.pi


def fromRpyXyz(rpy: List[float], xyz: List[float]) -> Matrix4x4:
    """
    rotate by roll-x, pitch-y, yaw-z: return Rot(z) * Rot(y) * Rot(x), ie. zyx euler system (degree)
    and then translate by xyz
    """
    quat = Quaternion.fromEulerAnglesOrder(rpy[0], rpy[1], rpy[2], order=[2,1,0])
    return Matrix4x4().rotate(quat).moveto(xyz[0], xyz[1], xyz[2])

def parse_origin(origin: ET.Element):
    rpy_str = origin.get('rpy', '0 0 0').split()
    rpy = [float(val) * rad2deg for val in rpy_str]  # degree
    xyz_str = origin.get('xyz', '0 0 0').split()
    xyz = [float(val) for val in xyz_str]
    return rpy, xyz

class GLLinkItem(GLGraphicsItem):

    def __init__(
        self,
        name: str = None,
        mesh: Union[GLGraphicsItem, Sequence[GLGraphicsItem]] = None,
        axis_visiable: bool = False,
        parentItem: GLGraphicsItem=None,
    ):
        super().__init__(parentItem=parentItem)
        self.name = name
        # axis
        self.axis = GLAxisItem(size=(0.12, 0.12, 0.12), tip_size=0.12)
        self.axis.setVisible(axis_visiable)
        self.addChildItem(self.axis)
        self._visual_models = []

        # visual model: optional, support multiple models
        if mesh is not None:
            if isinstance(mesh, GLGraphicsItem):
                mesh = [mesh]
            for m in mesh:
                self._visual_models.append(m)
                self.addChildItem(m)

    def childLinks(self) -> List['GLLinkItem']:
        return [item for item in self.childItems() if isinstance(item, GLLinkItem)]

    def add_visual_model(self, mesh: GLGraphicsItem):
        self._visual_models.append(mesh)
        self.addChildItem(mesh)

    def set_data(
        self,
        axis_visiable: bool = None,
        visual_visiable: bool = None,
    ):
        if axis_visiable is not None:
            self.axis.setVisible(axis_visiable)
            if len(self._visual_models):
                for m in self._visual_models:
                    m.setMaterialData(opacity=0.5 if axis_visiable else 1)

        if visual_visiable is not None and len(self._visual_models):
            for m in self._visual_models:
                m.setVisible(visual_visiable)

@dataclass
class Joint:
    name: str
    parent: GLLinkItem
    child: GLLinkItem
    type: str  # revolute, prismatic. fixed
    axis: np.ndarray = None
    limit: np.ndarray = None
    origin: Matrix4x4 = None # child axis relative to parent axis
    _value: float = 0  # rad or m

    def __post_init__(self):
        assert self.type in ['revolute', 'prismatic', 'fixed', 'continuous'], \
            "type must be revolute, prismatic or fixed"
        if self.origin is None:
            self.origin = Matrix4x4()

        self.set_value(self._value)
        self.parent.addChildItem(self.child)  # 添加父子关系

    @property
    def value(self):
        return self._value

    def set_value(self, value):  # rad or m
        if np.isnan(value):
            return
        self._value = np.clip(value, self.limit[0], self.limit[1])
        tf = Matrix4x4()
        # 根据关节类型设置关节值
        if self.type in ['revolute', 'continuous']:
            tf = tf.fromAxisAndAngle(self.axis[0], self.axis[1],
                                     self.axis[2], rad2deg*self._value)
        elif self.type == 'prismatic':
            t = self.axis * value
            tf = tf.moveto(t[0], t[1], t[2])
        self.child.setTransform(self.origin * tf)

    def set_origin(self, origin: Matrix4x4):
        self.origin = origin
        self.set_value(self._value)


class GLURDFItem(GLGraphicsItem):
    """ Displays a GelSlim model with a surface plot on top of it."""

    def __init__(
        self,
        urdf_path: Union[str, Path],
        lights: Sequence,
        up_axis='z',
        glOptions: str = "translucent",
        parentItem=None,
        axis_visiable=False,
        **kwargs # 传递给 GLLinkItem.set_data
    ):
        super().__init__(parentItem=parentItem)

        # 解析xml文件
        self._lights = lights
        self._up_axis = up_axis
        self._glOptions = glOptions
        path = Path(urdf_path)
        if path.is_file():
            urdf_path = path
        else:
            urdf_path = path / (path.stem + ".urdf")
        self._urdf_path = urdf_path
        self._base_dir = self._urdf_path.parent
        self._urdf = ET.parse(urdf_path)
        self._links: Dict[str, GLLinkItem] = dict()
        self._joints: Dict[str, Joint] = dict()

        # 遍历每个link元素
        for link in self._urdf.findall('link'):
            name = link.get('name')
            self._links[name] = self._parse_link(name, link)
            self._links[name].set_data(axis_visiable=axis_visiable, **kwargs)

        # 遍历每个joint元素
        for joint in self._urdf.findall('joint'):
            name = joint.get('name')
            self._joints[name] = self._parse_joint(name, joint)

        # 设置 base_link
        self.base_link = None
        for link in self._links.values():
            if link.parentItem() is None:
                self.addChildItem(link)
                self.base_link = link
                break

        if self.base_link is None:
            raise ValueError("No base link found")

    def set_joint(self, name: Union[int, str], value):
        """ 设置活动关节的值, 若 name 为 int, 表示活动关节序号 """
        joint = self.get_joint(name)
        joint.set_value(value)

    def get_joint(self, name: Union[int, str]) -> Joint:
        """ 返回关节实例, 若 name 为 int, 表示活动关节序号 """
        if isinstance(name, int):
            return self.get_joints(movable=True)[name]
        else:
            return self._joints[name]

    def set_joints(self, values: Sequence):
        """ 设置所有活动关节 """
        view = self.view()
        if view is not None:
            view.rw_lock.acquire_write()

        for joint, val in zip(self.get_joints(movable=True), values):
            joint.set_value(val)

        if view is not None:
            view.rw_lock.release_write()

    def get_joints(self, movable=True) -> List[Joint]:
        """ 默认返回所有活动关节 """
        if movable:
            return [joint for joint in self._joints.values() if joint.type != 'fixed']
        else:
            return list(self._joints.values())

    def get_joints_attr(self, attr: str, movable=True) -> List:
        """ 所有活动关节的属性, attr: name, value, axis, limit"""
        joints = self.get_joints(movable)
        return [getattr(joint, attr) for joint in joints]

    def get_links_name(self) -> List[str]:
        return list(self._links.keys())

    def set_link(self, name: Union[int, str], **kwargs): # axis_visiable, visual_visiable
        if isinstance(name, int):
            name = list(self._links.keys())[name]
        self._links[name].set_data(**kwargs)

    def get_link(self, name: Union[int, str]) -> GLLinkItem:
        if isinstance(name, int):
            name = list(self._links.keys())[name]
        return self._links[name]

    def get_link_tf(self, name: Union[int, str]) -> Matrix4x4:
        """返回连体坐标系到世界坐标系的变换矩阵"""
        return self.get_link(name).axis.transform(local=False)

    def add_link(self, link: GLLinkItem,
                 parent_link: Union[int, str],
                 joint_name: str,
                 joint_type: str,
                 joint_axis: Tuple[float, float, float],
                 joint_limit: Tuple[float, float],
                 origin: Matrix4x4 = None):
        """添加一个link和joint到urdf模型中"""

        self._links[link.name] = link

        self._joints[joint_name] = Joint(
            joint_name,
            parent = self.get_link(parent_link),
            child = link,
            type = joint_type,
            axis = np.array(joint_axis),
            limit = np.array(joint_limit),
            origin = origin
        )

    def print_links(self):
        """dfs print"""
        prefix = ' '
        print(self._urdf_path)
        stack = [(self.base_link, 1)]  # node, 缩进级别

        while(stack):
            node, level = stack.pop()
            print(prefix * level + node.name)

            for child in reversed(node.childLinks()):
                    stack.append((child, level+1))
        print()

    def print_joints(self):
        for name, joint in self._joints.items():
            print(f"{name} | {joint.type} | val: {joint.value} | "
                  f"axis: {joint.axis} | limit: {joint.limit}")
        print()

    def _parse_link(self, name: str, link_elem: ET.Element) -> GLLinkItem:
        """
        url: https://wiki.ros.org/urdf/XML/link
        """
        ets_visual = link_elem.findall('visual')
        visual_models = []
        for et_visual in ets_visual:
            visual = self._parse_visual(et_visual)
            if visual is not None:
                visual_models.append(visual)

        return GLLinkItem(name, visual_models)

    def _parse_visual(self, et_visual: ET.Element) -> GLGraphicsItem:
        origin = et_visual.find('origin')
        if origin is not None:
            origin_tf = fromRpyXyz(*parse_origin(origin))
        else:
            origin_tf = Matrix4x4()

        # -- parse geometry: mesh, box, cylinder, sphere
        geometry = et_visual.find('geometry')  # required

        # mesh
        et_mesh = geometry.find('mesh')
        if et_mesh is not None:
            mesh_path = self._base_dir / et_mesh.get('filename')
            scale = [float(val) for val in et_mesh.get('scale', '1 1 1').split()]
            visual_mesh =  GLModelItem(mesh_path, lights=self._lights, up_axis=self._up_axis, glOptions=self._glOptions)
            return visual_mesh.scale(*scale).applyTransform(origin_tf)

        # material
        material = Material(ambient=[0.4, 0.4, 0.4], diffuse=[0.7, 0.7, 0.7])
        et_material = et_visual.find('material')
        if et_material is not None:
            et_color = et_material.find('color')  # optional
            if et_color is not None:
                rgba = np.array([float(val) for val in et_color.get('rgba').split()])
            else:
                rgba = np.array([0.7, 0.7, 0.7, 1])
            material.set_data(ambient=rgba[:3]*0.6, diffuse=rgba[:3], opacity=rgba[3])

        # box
        et_box = geometry.find('box')
        if et_box is not None:
            size = [float(val) for val in et_box.get('size').split()]
            vertexes, normals, _ = cube(*size)
            return GLMeshItem(vertexes, normals=normals, lights=self._lights, material=material, glOptions=self._glOptions).applyTransform(origin_tf)

        # cylinder
        et_cylinder = geometry.find('cylinder')
        if et_cylinder is not None:
            radius = float(et_cylinder.get('radius'))
            length = float(et_cylinder.get('length'))
            vertexes, faces = cylinder([radius, radius], length, cols=20)
            vertexes[:, 2] -= length / 2
            return GLMeshItem(vertexes, faces, lights=self._lights, material=material, glOptions=self._glOptions).applyTransform(origin_tf)

        # sphere
        et_sphere = geometry.find('sphere')
        if et_sphere is not None:
            radius = float(et_sphere.get('radius'))
            vertexes, faces, _, normals = sphere(radius, 16, 16, calc_uv_norm=True)
            return GLMeshItem(vertexes, faces, normals, lights=self._lights, material=material, glOptions=self._glOptions).applyTransform(origin_tf)

        return None


    def _parse_joint(self, name: str, joint_elem: ET.Element):
        type = joint_elem.get('type')
        origin = fromRpyXyz(*parse_origin(joint_elem.find('origin')))
        parent = self._links[joint_elem.find('parent').get('link')]
        child = self._links[joint_elem.find('child').get('link')]
        if type in ['revolute', 'prismatic']:
            axis = np.array(joint_elem.find('axis').get('xyz').split(), dtype=float)
            limit = np.array([joint_elem.find('limit').get('lower'), joint_elem.find('limit').get('upper')], dtype=float)
        elif type == 'continuous':
            axis = np.array(joint_elem.find('axis').get('xyz').split(), dtype=float)
            limit = np.array([-1000, 1000], dtype=float)
        elif type == 'fixed':
            axis = None
            limit = np.array([0, 0], dtype=float)
        return Joint(name, parent, child, type, axis, limit, origin)