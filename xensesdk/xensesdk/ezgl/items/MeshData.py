import numpy as np
import math
from typing import List, Union, Sequence
from pathlib import Path
import OpenGL.GL as gl
import assimp_py as assimp
from time import time
from ctypes import c_float, sizeof, c_void_p, Structure
from .shader import Shader
from .Buffer import VAO, VBO, EBO, GLDataBlock
from .texture import Texture2D
from ..transform3d import Vector3, Matrix4x4
from ..functions import dispatchmethod

__all__ = [
    "Mesh", "Material", "direction_matrixs", "vertex_normal",
    "sphere", "cylinder", "cube", "cone", "plane", "axis_mesh", "arrow_mesh",
    "make_color", "fit_3d_spline_curve"
]

Vec2 = (2 * c_float)
Vec3 = (3 * c_float)


TextureType = {
    assimp.TextureType_DIFFUSE: "tex_diffuse",  # map_Kd
    assimp.TextureType_SPECULAR: "tex_specular",  # map_Ks
    assimp.TextureType_AMBIENT: "tex_ambient",  # map_Ka
    assimp.TextureType_HEIGHT: "tex_height",  # map_Bump
}

class Material():

    @dispatchmethod
    def __init__(
        self,
        ambient = [0.4, 0.4, 0.4],
        diffuse = [1.0, 1.0, 1.0],
        specular = [0.2, 0.2, 0.2],
        shininess: float = 10.,
        opacity: float = 1.,
        textures: Sequence[Texture2D] = list(),
        textures_paths: dict = dict(),
        directory: Union[str, Path] = Path(),
    ):
        self.ambient = Vector3(ambient)
        self.diffuse = Vector3(diffuse)
        self.specular = Vector3(specular)
        self.shininess = shininess
        self.opacity = opacity
        self.textures: List[Texture2D] = list()
        self.textures.extend(textures)
        self.texture_paths = textures_paths
        self.directory = directory

    @__init__.register(dict)
    def _(self, material_dict: dict, directory=None):
        self.__init__(
            material_dict.get("COLOR_AMBIENT", [0.4, 0.4, 0.4]),  # Ka
            material_dict.get("COLOR_DIFFUSE", [1.0, 1.0, 1.0]),  # Kd
            material_dict.get("COLOR_SPECULAR", [0.2, 0.2, 0.2]),  # Ks
            material_dict.get("SHININESS", 10),
            material_dict.get("OPACITY", 1),
            textures_paths = material_dict.get("TEXTURES", dict()),
            directory = directory,
        )

    def load_textures(self):
        """在 initializeGL() 中调用 """
        if self.texture_paths is None:
            return
        for type, path in self.texture_paths.items():
            if isinstance(path, list):
                path = path[0]
            if type in TextureType.keys():
                type = TextureType[type]
            self.textures.append(
                Texture2D(self.directory / path, tex_type=type)
            )

    def set_uniform(self, shader: Shader, name: str):
        use_texture = False
        for texture in self.textures:
            if texture.type == "tex_diffuse":
                shader.set_uniform(f"{name}.tex_diffuse", texture.bindTexUnit(), "sampler2D")
                use_texture = True
        shader.set_uniform(name+".ambient", self.ambient, "vec3")
        shader.set_uniform(name+".diffuse", self.diffuse, "vec3")
        shader.set_uniform(name+".specular", self.specular, "vec3")
        shader.set_uniform(name+".shininess", self.shininess, "float")
        shader.set_uniform(name+".opacity", self.opacity, "float")
        shader.set_uniform(name+".use_texture", use_texture, "bool")

    def set_data(self, ambient=None, diffuse=None, specular=None, shininess=None, opacity=None):
        if ambient is not None:
            self.ambient = Vector3(ambient)
        if diffuse is not None:
            self.diffuse = Vector3(diffuse)
        if specular is not None:
            self.specular = Vector3(specular)
        if shininess is not None:
            self.shininess = shininess
        if opacity is not None:
            self.opacity = opacity

    def __repr__(self) -> str:
        return f"Material(ambient={self.ambient}, diffuse={self.diffuse}, specular={self.specular}, shininess={self.shininess}, opacity={self.opacity})"


class Mesh():

    def __init__(
        self,
        vertexes = None,
        indices = None,
        texcoords = None,
        normals = None,
        material = None,
        directory = None,
        usage = gl.GL_STATIC_DRAW,
        calc_normals = True,
        mode = gl.GL_TRIANGLES,  # gl.GL_TRIANGLES, gl.GL_LINES, gl.GL_QUADS
    ):
        self._usage = usage
        self._mode = mode
        self._indices_layout = {gl.GL_TRIANGLES: 3, gl.GL_QUADS: 4}[mode]
        self._vertexes = GLDataBlock(np.float32, 3)
        self._indices = GLDataBlock(np.uint32, self._indices_layout)
        self._normals = GLDataBlock(np.float32, 3)
        self._texcoords = GLDataBlock(np.float32, 2)
        self.vbo = None

        if vertexes is not None:
            self._vertexes.set_data(np.array(vertexes, dtype=np.float32))

        if indices is not None:
            try:
                self._indices.set_data(np.array(indices, dtype=np.uint32))
            except: # assimp 的 indices 有时出错, 例如 [(0, 1), (2, 3), (4, 5, 6), (7, 8, 9) ...]
                indices = [item for item in indices if len(item)==3]
                self._indices.set_data(np.array(indices, dtype=np.uint32))

        if calc_normals and normals is None and indices is not None:
            self._normals.set_data(vertex_normal(self._vertexes.data, self._indices.data))
        else:
            self._normals.set_data(normals)

        if texcoords is not None:
           texcoords = np.array(texcoords, dtype=np.float32)[..., :2]
        self._texcoords.set_data(texcoords)

        if isinstance(material, dict):
            self._material = Material(material, directory)
        elif isinstance(material, Material):
            self._material = material
        elif material is None:
            self._material = Material()

    def scale(self, scale):
        self._vertexes.set_data(self._vertexes.data * scale)

    def initializeGL(self):
        if self.vbo is not None:
            return

        with VAO() as self.vao:
            self.vbo = VBO(
                [self._vertexes, self._normals, self._texcoords],
                expandable = True,
                usage = self._usage
            )
            self.vbo.setAttrPointer([0, 1, 2], attr_id=[0, 1, 2])
            self.ebo = EBO(self._indices)
            self._material.load_textures()

    def paint(self, shader):
        self.vbo.commit()
        self.ebo.commit()
        self._material.set_uniform(shader, "material")

        if self.vbo.size(2) == 0:  # 没有纹理坐标
            shader.set_uniform("material.use_texture", False, 'bool')

        with self.vao:
            if self.ebo.size() > 0:
                gl.glDrawElements(self._mode, self.ebo.size(), gl.GL_UNSIGNED_INT, c_void_p(0))
            else:
                gl.glDrawArrays(self._mode, 0, self.vbo.count(0))

    def paintShadow(self):
        self.vbo.commit()
        self.ebo.commit()
        with self.vao:
            if self.ebo.size() > 0:
                gl.glDrawElements(self._mode, self.ebo.size(), gl.GL_UNSIGNED_INT, c_void_p(0))
            else:
                gl.glDrawArrays(self._mode, 0, self.vbo.count(0))

    def setMaterial(self, material=None):
        if isinstance(material, dict):
            self._material = Material(material)
        elif isinstance(material, Material):
            self._material = material

    def getMaterial(self):
        return self._material

    @classmethod
    def load_model(cls, path: Union[str, Path], up_axis, gen_normals=True) -> List["Mesh"]:
        """
        Load a model from file and return a list of Mesh instances.

        Parameters:
        - path: str or Path, the path to the model file.
        - up_axis: str, the up axis of the model, 'z' or 'y', 若为 'y' 则读入顶点xyz顺序和模型文件中完全一致,
            若为 'z' 则模型中的xyz顺序会变为 x, -z, y
        - gen_normals: bool, default: True, 是否在加载模型时计算法线. 若为 True, 由于存在重复点, assimp 必须复制一些顶点，
            以便正确计算法线, 因此加载的顶点数可能会比原始模型的顶点数多。若为 False, 顶点数量和模型文件中一致, 但是加载的顶点顺
            序与模型文件中顶点顺序不相同, 与顶点在文件faces中出现的顺序相同
        """
        meshes = list()
        directory = Path(path).parent
        face_num = 0

        # start_time = time()
        post_process = (assimp.Process_Triangulate |
                        assimp.Process_FlipUVs|
                        assimp.Process_PreTransformVertices|
                        assimp.Process_JoinIdenticalVertices
                        )
                        # assimp.Process_CalcTangentSpace 计算法线空间
        if gen_normals:
            post_process |= assimp.Process_GenNormals
        try:
            scene = assimp.ImportFile(str(path), post_process)
        except Exception as e:
            raise ValueError(f"{e} \nERROR: Assimp model failed to load, {path}")

        for m in scene.meshes:

            verts = np.array(m.vertices, dtype=np.float32).reshape(-1, 3)
            if m.normals is not None:
                norms = np.array(m.normals, dtype=np.float32).reshape(-1, 3)
            else:
                norms = None

            if up_axis == 'z': # x, y, z -> x, -z, y
                verts = np.stack((verts[:, 0], -verts[:, 2], verts[:, 1]), axis=1)
                if norms is not None:
                    norms = np.stack((norms[:, 0], -norms[:, 2], norms[:, 1]), axis=1)

            meshes.append(
                cls(
                    verts,
                    m.indices,
                    m.texcoords[0] if len(m.texcoords) > 0 else None,
                    norms,
                    scene.materials[m.material_index],
                    directory=directory,
                )
            )
            face_num += len(m.indices)

        # print(f"Took {round(time()-start_time, 3)}s to load {path} (faces: {face_num})")
        return meshes

def cone(radius, height, slices=12):
    slices = max(3, slices)
    vertices = np.zeros((slices+2, 3), dtype="f4")
    vertices[-2] = [0, 0, height]
    step = 360 / slices  # 圆每一段的角度
    for i in range(0, slices):
        p = step * i * 3.14159 / 180  # 转为弧度
        vertices[i] = [radius * math.cos(p), radius * math.sin(p), 0]
    # 构造圆锥的面索引
    indices = np.zeros((slices*6, ), dtype=np.uint32)
    for i in range(0, slices):
        indices[i*6+0] = i
        indices[i*6+1] = (i+1) % slices
        indices[i*6+2] = slices
        indices[i*6+3] = i
        indices[i*6+5] = (i+1) % slices
        indices[i*6+4] = slices+1
    return vertices, indices.reshape(-1, 3)


def direction_matrixs(starts, ends, move_to_ends=True):
    """
    返回 shape=(n, 4, 4) 的变换矩阵, 用于 GLArrowPlotItem
    """
    starts = np.array(starts)
    ends = np.array(ends)
    arrows = ends - starts
    arrows = arrows.reshape(-1, 3)
    # 处理零向量，归一化
    arrow_lens = np.linalg.norm(arrows, axis=1)  # (n,)
    zero_idxs = arrow_lens < 1e-3
    arrows[zero_idxs] = [0, 0, 1]

    arrow_lens_den = arrow_lens.copy()
    arrow_lens_den[zero_idxs] = 1
    arrows = arrows / arrow_lens_den[:, np.newaxis]
    # 构造标准箭头到目标箭头的旋转矩阵
    T = np.zeros_like(arrows)
    B = np.zeros_like(arrows)
    mask = arrows[:, 2] < -0.99999
    T[mask, 1] = -1
    B[mask, 0] = -1
    mask = np.logical_not(mask)
    a = 1 / (1 + arrows[mask, 2])
    b = -arrows[mask, 0] * arrows[mask, 1] * a
    T[mask] = np.stack((
        1 - arrows[mask, 0] * arrows[mask, 0] * a,
        b,
        -arrows[mask, 0],
    ), axis=1)
    B[mask] = np.stack((
        b,
        1 - arrows[mask, 1] * arrows[mask, 1] * a,
        -arrows[mask, 1],
    ), axis=1)
    # 转化成齐次变换矩阵
    transforms = np.zeros((len(arrows), 4, 4), dtype=np.float32)

    if move_to_ends:  # 将箭头的起始点移动到目标位置
        transforms4x3 = np.stack((T, B, arrows, ends.reshape(-1, 3)), axis=1)  # (n, 4(new), 3)
    else:
        transforms4x3 = np.stack((T, B, arrows, starts.reshape(-1, 3)), axis=1)  # (n, 4(new), 3)

    transforms[:, :, :3] = transforms4x3
    transforms[:, 3, 3] = 1
    return transforms, arrow_lens

def get_sphere_uv(verts):
    """采样球面贴图 verts: [n, 3]"""
    r = np.sqrt(verts[0, 0]**2 + verts[0, 1]**2 + verts[0, 2]**2)
    v = verts / r
    uv = np.zeros((verts.shape[0], 2), dtype=np.float32)
    uv[:, 0] = np.arctan2(v[:, 0], v[:, 1])
    uv[:, 1] = np.arcsin(v[:, 2])
    uv *= np.array([1 / (2 * np.pi), 1 / np.pi])
    uv += 0.5
    return uv

def sphere(radius=1.0, rows=12, cols=12, calc_uv_norm=False):
    """
    Return a MeshData instance with vertexes and faces computed
    for a spherical surface.
    """
    verts = np.empty((rows+1, cols+1, 3), dtype=np.float32)

    # compute vertexes
    phi = np.linspace(0, np.pi, rows+1, dtype=np.float32).reshape(rows+1, 1)
    s = radius * np.sin(phi)
    verts[...,2] = radius * np.cos(phi)

    th = np.linspace(0, 2 * np.pi, cols+1, dtype=np.float32).reshape(1, cols+1)
    verts[...,0] = s * np.cos(th)
    verts[...,1] = s * np.sin(th)
    verts = verts.reshape(-1, 3)

    # compute faces
    faces = np.empty((rows, 2, cols, 3), dtype=np.uint32)
    rowtemplate1 = np.arange(cols).reshape(1, cols, 1) + np.array([[[0     , cols+1, 1]]]) # 1, cols, 3
    rowtemplate2 = np.arange(cols).reshape(1, cols, 1) + np.array([[[cols+1, cols+2, 1]]]) # 1, cols, 3
    rowbase = np.arange(rows).reshape(rows, 1, 1) * (cols+1)  # nrows, 1, 1
    faces[:, 0] = (rowtemplate1 + rowbase)  # nrows, 1, ncols, 3
    faces[:, 1] = (rowtemplate2 + rowbase)

    faces = faces.reshape(-1, 3)
    faces = faces[cols:-cols]  # cut off zero-area triangles at top and bottom

    # 去掉上下重合顶点
    faces = np.clip(faces, cols, verts.shape[0]-cols-1) - cols
    verts = verts[cols:-cols]

    # compute uv and normals
    if calc_uv_norm:
        uv = np.zeros((rows+1, cols+1, 2), dtype=np.float32)
        uv[..., 0] = th / (2 * np.pi)
        uv[..., 1] = phi / np.pi
        uv = uv.reshape(-1, 2)[cols:-cols]
        norms = verts / radius  # rows, cols, 2
        return verts, faces, uv, norms

    return verts, faces


def cylinder(radius=[1.0, 1.0], length=1.0, rows=1, cols=12, offset=False):
    """
    Return a MeshData instance with vertexes and faces computed
    for a cylindrical surface.
    The cylinder may be tapered with different radii at each end (truncated cone)
    """
    verts = np.empty(((rows+3)*cols+2, 3), dtype=np.float32)  # 顶面的点和底面的点重复一次, 保证法线计算正确
    verts1 = verts[:(rows+1)*cols, :].reshape(rows+1, cols, 3)
    if isinstance(radius, int):
        radius = [radius, radius] # convert to list
    ## compute vertexes
    th = np.linspace(2 * np.pi, (2 * np.pi)/cols, cols).reshape(1, cols)
    r = np.linspace(radius[0],radius[1],num=rows+1, endpoint=True).reshape(rows+1, 1) # radius as a function of z
    verts1[...,2] = np.linspace(0, length, num=rows+1, endpoint=True).reshape(rows+1, 1) # z
    if offset:
        th = th + ((np.pi / cols) * np.arange(rows+1).reshape(rows+1,1))  ## rotate each row by 1/2 column
    verts1[...,0] = r * np.cos(th) # x = r cos(th)
    verts1[...,1] = r * np.sin(th) # y = r sin(th)
    verts1 = verts1.reshape((rows+1)*cols, 3) # just reshape: no redundant vertices...
    # 顶面, 底面
    verts[(rows+1)*cols:(rows+2)*cols] = verts1[-cols:]
    verts[(rows+2)*cols:-2] = verts1[:cols]
    verts[-2] = [0, 0, 0] # zero at bottom
    verts[-1] = [0, 0, length] # length at top

    ## compute faces
    num_side_faces = rows * cols * 2
    num_cap_faces = cols
    faces = np.empty((num_side_faces + num_cap_faces*2, 3), dtype=np.uint32)
    rowtemplate1 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 0, 1]])) % cols) + np.array([[0, cols, 0]])
    rowtemplate2 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) + np.array([[cols, cols, 0]])
    for row in range(rows):
        start = row * cols * 2
        faces[start:start+cols] = rowtemplate1 + row * cols
        faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols

    # Bottom face
    bottom_start = num_side_faces
    bottom_row = np.arange((rows+2) * cols, (rows+3) * cols)
    bottom_face = np.column_stack((bottom_row, np.roll(bottom_row, -1), np.full(cols, (rows+3) * cols)))
    faces[bottom_start : bottom_start + num_cap_faces] = bottom_face

    # Top face
    top_start = num_side_faces + num_cap_faces
    top_row = np.arange((rows+1) * cols, (rows+2) * cols)
    top_face = np.column_stack((np.roll(top_row, -1), top_row, np.full(cols, (rows+3) * cols+1)))
    faces[top_start : top_start + num_cap_faces] = top_face

    return verts, faces

def cube(x, y, z):
    """
    Return a MeshData instance with vertexes and normals computed
    for a rectangular cuboid of the given dimensions.
    """
    vertices = np.array( [
        # 顶点坐标             # 法向量       # 纹理坐标
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
         0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 0.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
        -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0,

        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,
         0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
        -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 1.0,
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,

        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,
        -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  1.0, 1.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
        -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  0.0, 0.0,
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,

         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
         0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 1.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
         0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 0.0,

        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0,
         0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0, 1.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
        -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 0.0,
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0,

        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
         0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 1.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
        -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0, 0.0,
    ], dtype="f4").reshape(-1, 8)
    verts = vertices[:, :3] * np.array([x,y,z], dtype="f4")
    normals = vertices[:, 3:6]
    texcoords = vertices[:, 6:]
    return verts, normals, texcoords


def plane(x, y):
    vertices = np.array([
        # 顶点坐标             # 法向量       # 纹理坐标
        -0.5, -0.5, 0.0,  0.0,  0.0, 1.0,  0.0, 0.0,
         0.5, -0.5, 0.0,  0.0,  0.0, 1.0,  1.0, 0.0,
         0.5,  0.5, 0.0,  0.0,  0.0, 1.0,  1.0, 1.0,
         0.5,  0.5, 0.0,  0.0,  0.0, 1.0,  1.0, 1.0,
        -0.5,  0.5, 0.0,  0.0,  0.0, 1.0,  0.0, 1.0,
        -0.5, -0.5, 0.0,  0.0,  0.0, 1.0,  0.0, 0.0,
    ], dtype=np.float32).reshape(-1, 8)
    verts = vertices[:, :3] * np.array([x,y,1.0], dtype="f4")
    normals = vertices[:, 3:6]
    texcoords = vertices[:, 6:]
    return verts, normals, texcoords


def vertex_normal(vert, ind):
    """
    计算每个顶点的法向量

    Parameters:
    - vert : np.ndarray, float32, shape=(n, 3), 顶点坐标
    - ind : np.ndarray, uint32, shape=(m, 3) or (m, 4), 面索引

    Returns:
    - norm, np.ndarray, float32, shape=(n, 3), 顶点法向量
    """
    nv = len(vert) # 顶点的个数
    norm = np.zeros((nv, 3), np.float32) # 初始化每个顶点的法向量为零向量

    # 处理三角形
    tri_mask = (ind.shape[1] == 3)
    if tri_mask:
        v0 = vert[ind[:, 0]]
        v1 = vert[ind[:, 1]]
        v2 = vert[ind[:, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        np.add.at(norm, ind[:, 0], normal)
        np.add.at(norm, ind[:, 1], normal)
        np.add.at(norm, ind[:, 2], normal)

    # 处理四边形
    quad_mask = (ind.shape[1] == 4)
    if quad_mask:
        v0 = vert[ind[:, 0]]
        v1 = vert[ind[:, 1]]
        v2 = vert[ind[:, 2]]
        v3 = vert[ind[:, 3]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal1 = np.cross(edge1, edge2)
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal2 = np.cross(edge1, edge2)
        np.add.at(norm, ind[:, [0, 1, 2]], normal1[:, None])
        np.add.at(norm, ind[:, [1, 2, 3]], normal2[:, None])

    norm_len = np.linalg.norm(norm, axis=1, keepdims=True) # 计算每个顶点的法向量的长度
    norm_len[norm_len < 1e-5] = 1  # 处理零向量
    norm = norm / norm_len  # 归一化每个顶点的法向量

    return norm


def surface(zmap, xy_size):
    x_size, y_size = xy_size
    h, w = zmap.shape
    scale = x_size / w
    zmap *= scale

    x = np.linspace(-x_size/2, x_size/2, w, dtype='f4')
    y = np.linspace(y_size/2, -y_size/2, h, dtype='f4')

    xgrid, ygrid = np.meshgrid(x, y, indexing='xy')
    verts = np.stack([xgrid, ygrid, zmap.astype('f4')], axis=-1).reshape(-1, 3)

    # calc indices
    ncol = w - 1
    nrow = h - 1
    if ncol == 0 or nrow == 0:
        raise Exception("cols or rows is zero")

    faces = np.empty((nrow, 2, ncol, 3), dtype=np.uint32)
    rowtemplate1 = np.arange(ncol).reshape(1, ncol, 1) + np.array([[[0     , ncol+1, 1]]])  # 1, ncols, 3
    rowtemplate2 = np.arange(ncol).reshape(1, ncol, 1) + np.array([[[ncol+1, ncol+2, 1]]])
    rowbase = np.arange(nrow).reshape(nrow, 1, 1) * (ncol+1)  # nrows, 1, 1
    faces[:, 0] = (rowtemplate1 + rowbase)  # nrows, 1, ncols, 3
    faces[:, 1] = (rowtemplate2 + rowbase)

    return verts, faces.reshape(-1, 3)


def grid3d(grid):
    # grid: (h, w, 3)
    h, w = grid.shape[:2]
    ncol, nrow = w-1, h-1

    rowtemplate = np.arange(ncol, dtype=np.uint32).reshape(1, ncol, 1) + \
        np.array([[[0, ncol+1, ncol+2, 1]]])  # 1, ncols, 4
    rowbase = np.arange(nrow, dtype=np.uint32).reshape(nrow, 1, 1) * (ncol+1)  # nrows, 1, 1
    faces = (rowtemplate + rowbase).reshape(-1, 4).astype(np.uint32)

    return grid.reshape(-1, 3).astype(np.float32), faces


def mesh_concat(verts: Sequence, faces: Sequence):
    """合并多个网格"""

    vert_nums = [len(v) for v in verts]
    id_bias = np.cumsum(vert_nums, dtype=np.uint32)
    for i in range(1, len(faces)):
        faces[i] += id_bias[i-1]

    verts = np.concatenate(verts, axis=0)
    faces = np.concatenate(faces, axis=0).astype(np.uint32)

    return verts, faces

def make_color(color: Union[list, np.ndarray], opacity=None) -> np.ndarray:
    """
    将颜色转化成统一格式 float16 RGBA

    Parameters:
    - color : Union[list, np.ndarray], 可以为 0-255 或 0-1, shape=(3,) or (4,) or (n, 3) or (n, 4)
    - opacity : float, default=None, 透明度, 若不为 None 优先级高于 color 的 alpha 通道, 若无 alpha 通道, 则默认为 1

    Returns:
    - np.ndarray of float16, shape=(n, 4)
    """
    if color is None or len(color) == 0:
        return None

    if isinstance(color, (list, tuple)):
        color = np.array(color, dtype=np.float16)

    color = color.astype(np.float16)
    channels = color.shape[-1]
    color = color.reshape(-1, channels)

    if np.max(color) > 10:  # 有时候需要设置 > 1, 因为shader可能会缩小颜色值
        color = color / 255

    if channels == 3:
        opacity = 1 if opacity is None else opacity
        color = np.concatenate(
            [color, np.full((color.shape[0], 1), opacity, dtype=np.float16)],
            axis=1
        )

    elif opacity is not None:
        color[:, 3] = opacity

    return color


def arrow_mesh(length=1.0, width=0.03, tip_width=2, tip_length=0.1):
    """
    生成箭头 Mesh 数据

    Parameters:
    - length : float, optional, default: 1.0, 箭头的长度
    - width : float, optional, default: 0.03, 箭头柄的半径
    - tip_width : float, optional, default: 2, 箭头头部的宽度相对于 width 的比例
    - tip_length : float, optional, default: 0.1, 箭头头部的长度相对于 length 的比例

    Returns:
    - verts, faces
    """
    cylinder_verts, cylinder_faces = cylinder([width, width], length*(1-tip_length), 1, 12)
    cone_verts, cone_faces = cylinder([width*tip_width, 0], length*tip_length, 1, 20)
    cone_verts[:, 2] += length * (1 - tip_length)

    verts, faces = mesh_concat([cylinder_verts, cone_verts], [cylinder_faces, cone_faces])
    return verts, faces


def axis_mesh(length=1.0, width=0.03, tip_width=2, tip_length=0.1):
    """
    生成坐标轴 Mesh 数据

    Parameters:
    - length : float, optional, default: 1.0, 箭头的长度
    - width : float, optional, default: 0.03, 箭头柄的半径
    - tip_width : float, optional, default: 2, 箭头头部的宽度相对于 width 的比例
    - tip_length : float, optional, default: 0.1, 箭头头部的长度相对于 length 的比例

    Returns:
    - verts, faces, color
    """
    cylinder_verts, cylinder_faces = cylinder([width, width], length*(1-tip_length), 1, 12)
    cone_verts, cone_faces = cylinder([width*tip_width, 0], length*tip_length, 1, 20)
    cone_verts[:, 2] += length * (1 - tip_length)

    verts_z, faces_z = mesh_concat([cylinder_verts, cone_verts], [cylinder_faces, cone_faces])
    verts_x = Matrix4x4.fromAxisAndAngle(0, 1, 0, 90) * verts_z
    verts_y = Matrix4x4.fromAxisAndAngle(1, 0, 0, -90) * verts_z

    verts, faces = mesh_concat([verts_x, verts_y, verts_z], [faces_z, faces_z.copy(), faces_z.copy()])
    color = np.concatenate([np.tile([1, 0, 0, 1], (len(verts_z), 1)),
                            np.tile([0, 1, 0, 1], (len(verts_z), 1)),
                            np.tile([0, 0, 1, 1], (len(verts_z), 1))], axis=0)
    return verts, faces, color


def cubic_spline(t, values, num_points):
    """
    用三次样条插值拟合数据点
    """
    n = len(t)
    A = np.zeros((n, n))
    B = np.zeros(n)

    # 构造三次样条方程的系数矩阵A和右边的向量B
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n-1):
        h1 = t[i] - t[i-1]
        h2 = t[i+1] - t[i]
        A[i, i-1] = h1
        A[i, i] = 2 * (h1 + h2)
        A[i, i+1] = h2
        B[i] = 3 * ( (values[i+1] - values[i]) / h2 - (values[i] - values[i-1]) / h1 )

    # 求解方程 A * c = B，得到样条曲线的二阶导数c
    c = np.linalg.solve(A, B)

    # 计算样条曲线的系数
    a = values[:-1]
    b = (values[1:] - values[:-1]) / (t[1:] - t[:-1]) - (t[1:] - t[:-1]) * (c[1:] + 2*c[:-1]) / 3
    d = (c[1:] - c[:-1]) / (3 * (t[1:] - t[:-1]))
    c = c[:-1]  # 去掉最后一个c值，因为它是虚拟的，用于方程的计算

    # 根据新参数生成拟合曲线
    t_new = np.linspace(t[0], t[-1], num_points)
    values_fit = np.zeros(num_points)

    for i in range(1, n):
        idx = (t_new >= t[i-1]) & (t_new <= t[i])
        dt = t_new[idx] - t[i-1]
        values_fit[idx] = a[i-1] + b[i-1]*dt + c[i-1]*dt**2 + d[i-1]*dt**3

    return t_new, values_fit


def fit_3d_spline_curve(pts, num_points=100):
    """
    使用三次多项式插值拟合三维折线，返回一个光滑曲线。

    :param pts: 输入的三维点，形状为 (n, 3)，每个点 (x, y, z)
    :param num_points: 拟合后的输出曲线的点数
    :return: 光滑曲线输出，形状为 (num_points, 3)
    """
    # 获取输入点的坐标
    pts = np.unique(pts, axis=0)

    # 计算累积弧长（用来作为参数 t）
    dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))  # 点与点之间的距离
    t = np.insert(np.cumsum(dists), 0, 0)  # 累积弧长 t

    # 对x、y、z维度分别进行插值
    num_dimension = pts.shape[1]
    output = np.zeros((num_points, num_dimension), dtype=np.float32)
    for i in range(num_dimension):
        _, output[:, i] = cubic_spline(t, pts[:, i], num_points)

    return output