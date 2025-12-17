import numpy as np
import OpenGL.GL as gl
from typing import List, Set, Sequence
from .shader import Shader
from .Buffer import VAO, VBO, EBO, GLDataBlock
from ..transform3d import Vector3, Matrix4x4
from ..GLGraphicsItem import GLGraphicsItem
from .MeshData import sphere
from .FrameBufferObject import FBO
from .camera import Camera
from .render import RenderGroup

__all__ = ["PointLight", "LineLight", "LightMixin"]

class PointLight(GLGraphicsItem):

    Type = 0

    def __init__(
        self,
        pos = (0.0, 0.0, 1.0),
        ambient = (0.2, 0.2, 0.2),
        diffuse = (0.8, 0.8, 0.8),
        specular = (1.0, 1.0, 1.0),
        constant = 1.0,
        linear = 0.01,
        quadratic = 0.001,
        visible = False,
        directional = False, # 平行光源选项
        render_shadow = False,
        shadow_size = (1600, 1600), # 阴影贴图大小
        light_frustum_visible = False,  # 是否绘制光源的视锥体, 只用于调试阴影
        glOptions = "opaque",
    ):
        super().__init__(parentItem=None)
        self.setGLOptions(glOptions)
        # 阴影相机
        self.item_group = RenderGroup()  # 光源照射的物体
        self.camera = Camera(proj_type="ortho", parentItem=self)
        self.setShadow(bias=0.05, sampleNum=6)
        self.shadow_fbo = None  # 阴影贴图

        # 光源属性
        self.moveTo(*pos)
        self._update_camera()  # 避免在这里写成 self.position = pos, 否则影响子类的初始化, 因为子类的 position.setter 可能实现不同
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic
        self.setVisible(visible)
        self.directional = directional
        self.render_shadow = render_shadow  # 是否渲染阴影
        self.shadow_size = shadow_size    # 阴影贴图大小
        self.camera.setVisible(light_frustum_visible, True)

    @property
    def position(self) -> np.ndarray:
        """
        光源在世界坐标系中的位置

        Returns:
        - np.ndarray, shape=(3,)
        """
        return self.transform(False).toTranslation()

    @position.setter
    def position(self, pos: np.ndarray):
        self.moveTo(*pos)
        self._update_camera()

    def set_uniform(self, shader: Shader, name: str):
        shader.set_uniform(name + ".position", self.position, "vec3")
        shader.set_uniform(name + ".ambient", self.ambient, "vec3")
        shader.set_uniform(name + ".diffuse", self.diffuse, "vec3")
        shader.set_uniform(name + ".specular", self.specular, "vec3")
        shader.set_uniform(name + ".constant", self.constant, "float")
        shader.set_uniform(name + ".linear", self.linear, "float")
        shader.set_uniform(name + ".quadratic", self.quadratic, "float")
        shader.set_uniform(name + ".directional", self.directional, "bool")
        shader.set_uniform(name + ".renderShadow", self.render_shadow and self.shadow_fbo is not None, "bool")
        shader.set_uniform(name + ".type", self.Type, "int")
        if self.shadow_fbo is not None:
            shader.set_uniform(name + ".shadowMap", self.shadow_fbo.depth_texture.bindTexUnit(), "sampler2D")
            shader.set_uniform(name + ".lightSpaceMatrix", self.camera.get_proj_view_matrix().glData, "mat4")
            shader.set_uniform(name + ".bias", self._bias, "float")
            shader.set_uniform(name + ".sampleNum", self._sampleNum, "float")

    def set_data(self, pos=None, ambient=None, diffuse=None, specular=None, visible=None, render_shadow=None):
        if pos is not None:
            self.position = pos
        if ambient is not None:
            self.ambient = np.array(ambient)
        if diffuse is not None:
            self.diffuse = np.array(diffuse)
        if specular is not None:
            self.specular = np.array(specular)
        if visible is not None:
            self.setVisible(visible)
        if render_shadow is not None:
            self.render_shadow = render_shadow

    def translate(self, dx, dy, dz, local=False):
        super().translate(dx, dy, dz, local)
        self._update_camera()
        return self

    def rotate(self, angle, x, y, z, local=False):
        super().rotate(angle, x, y, z, local)
        self._update_camera()
        return self

    def moveTo(self, x, y, z):
        super().moveTo(x, y, z)
        self._update_camera()
        return self

    def scale(self, x, y, z, local=False):
        super().scale(x, y, z, local)
        self._update_camera()
        return self

    def add_shadow_item(self, item: GLGraphicsItem):
        self.item_group.add(item)

    def add_shadow_items(self, items: Sequence[GLGraphicsItem]):
        self.item_group |= items

    def remove_shadow_item(self, item: GLGraphicsItem):
        self.item_group.remove(item)

    def initializeGL(self):
        """初始化阴影渲染数据"""
        _light_vert, _light_idx = sphere(0.05, 12, 12)
        self._light_vert = GLDataBlock(np.float32, 3, _light_vert)
        self._light_idx = GLDataBlock(np.uint32, 3, _light_idx)
        with VAO() as self._light_vao:
            self._light_vbo = VBO([self._light_vert], False)
            self._light_vbo.setAttrPointer([0], [0])
            self._light_ebo = EBO(self._light_idx)
            self._light_shader = Shader(vertex_shader, fragment_shader)

    def renderShadow(self):
        """更新阴影贴图"""
        if not self.render_shadow:
            return

        if self.shadow_fbo is None:
            self._shadow_shader = Shader(shadow_vertex_shader, empty_fragment_shader)
            self.shadow_fbo = FBO(*self.shadow_size, type=FBO.Type.DEPTH)

        with self.shadow_fbo:
            gl.glDepthMask(gl.GL_TRUE)   # 保证深度缓冲区可写
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glEnable(gl.GL_DEPTH_TEST)
            # 解决阴影失真的一种方案, 不可与阴影偏移 bais 同时使用
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_BACK)

            # gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            # gl.glPolygonOffset(1., 1.)
            # 渲染阴影
            self.item_group.render(self.camera, self._shadow_shader)
            # gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

    def setShadow(self, left=-4, right=4, bottom=-4, top=4, near=-20, far=20, bias=None, sampleNum=None):
        self.camera.set_proj_matrix(ortho_space=[left, right, bottom, top, near, far])
        if bias is not None:
            self._bias = bias / (far - near)  # 阴影偏移量(防止阴影失真), 默认为 0.05
        if sampleNum is not None:
            self._sampleNum = sampleNum

    def _update_camera(self):
        """
        在移动光源后, 确保深度相机的观察点在原点
        """
        eye = self.position
        length = np.linalg.norm(eye)
        if length == 0:
            eye[2] = 1

        up = np.array([0, 0, 1])
        if eye[0] == 0 and eye[1] == 0:
            up = np.array([0, 1, 0])

        self.camera.set_view_matrix(Matrix4x4.lookAt(eye, [0, 0, 0], up))

    def paint(self, camera: Camera):
        if not self.visible():
            return

        self.setupGLState()
        with self._light_shader, self._light_vao:
            self._light_shader.set_uniform("proj_view", camera.get_proj_view_matrix().glData, "mat4")
            self._light_shader.set_uniform("model", self.model_matrix().glData, "mat4")
            self._light_shader.set_uniform("_lightColor", self.diffuse, "vec3")
            gl.glDrawElements(gl.GL_TRIANGLES, self._light_ebo.size(), gl.GL_UNSIGNED_INT, None)

    def loadDict(self, data: dict):
        for key, value in data.items():
            if key == "visible":
                self.setVisible(value)
            elif key == "light_frustum_visible":
                self.camera.setVisible(value, True)
            else:
                if isinstance(value, Sequence):
                    value = np.array(value)
                setattr(self, key, value)

    def toDict(self) -> dict:
        return {
            "position": list(self.position),
            "ambient": list(self.ambient),
            "diffuse": list(self.diffuse),
            "specular": list(self.specular),
            "constant": self.constant,
            "linear": self.linear,
            "quadratic": self.quadratic,
            "visible": self.visible(),
            "directional": self.directional,
            "render_shadow": self.render_shadow,
            "shadow_size": list(self.shadow_size),
            "light_frustum_visible": self.camera.visible(),
        }


class LineLight(PointLight):

    Type = 1

    def __init__(
        self,
        pos = (0.0, 0.0, 1.0),
        pos2 = (0.0, 0.0, 2.0),
        ambient = (0.2, 0.2, 0.2),
        diffuse = (0.8, 0.8, 0.8),
        specular = (1.0, 1.0, 1.0),
        constant = 1.0,
        linear = 0.01,
        quadratic = 0.001,
        visible = False,
        directional = False, # 平行光源选项
        render_shadow = False,
        shadow_size = (1600, 1600), # 阴影贴图大小
        light_frustum_visible = False,  # 是否绘制光源的视锥体, 只用于调试阴影
        glOptions = "opaque",
    ):
        self.dpos = np.array(pos2) - np.array(pos)
        super().__init__(pos, ambient, diffuse, specular, constant, linear, quadratic, visible, directional, render_shadow, shadow_size, light_frustum_visible, glOptions)

        self._light_vert = GLDataBlock(np.float32, 3, np.array([[0, 0, 0], self.dpos], dtype=np.float32))

    @property
    def position(self) -> np.ndarray:
        return super().position

    @position.setter
    def position(self, pos: np.ndarray):
        # 为了保证修改 position1 不影响 position2, 重新计算 dpos, tf * dpos = pos2
        pos2 = self.position2
        self.moveTo(*pos)
        self.dpos = self.transform(False).inverse() * pos2
        self._light_vert.set_data(np.array([[0, 0, 0], self.dpos], dtype=np.float32))
        self._update_camera()

    @property
    def position2(self) -> np.ndarray:
        """
        光源终点在世界坐标系中的位置

        Returns:
        - np.ndarray, shape=(3,)
        """
        return self.transform(False) * self.dpos

    @position2.setter
    def position2(self, pos2: np.ndarray):
        self.dpos = self.transform(False).inverse() * pos2
        self._light_vert.set_data(np.array([[0, 0, 0], self.dpos], dtype=np.float32))
        self._update_camera()

    def set_uniform(self, shader: Shader, name: str):
        shader.set_uniform(name + ".position2", self.position2, "vec3")
        return super().set_uniform(shader, name)

    def _update_camera(self):
        """
        在移动光源后, 确保深度相机的观察点在原点
        """
        pos1 = self.position
        pos2 = self.position2
        eye = (pos1 + pos2) / 2
        length = np.linalg.norm(eye)
        if length == 0:
            eye[2] = 1

        # 计算 center, 保证视线 eye-center 垂直于 pos-pos2, 且朝向坐标原点
        x_dir = (pos2 - pos1) / np.linalg.norm(pos2 - pos1)
        center = np.dot(eye, x_dir) * x_dir
        up = np.cross(eye - center, x_dir)

        self.camera.set_view_matrix(Matrix4x4.lookAt(eye, center, up))

    def initializeGL(self):
        """初始化阴影渲染数据"""
        with VAO() as self._light_vao:
            self._light_vbo = VBO([self._light_vert], False)
            self._light_vbo.setAttrPointer([0], [0])
            self._light_shader = Shader(vertex_shader, fragment_shader)
            self._shadow_shader = Shader(shadow_vertex_shader, empty_fragment_shader)

    def paint(self, camera: Camera):
        if not self.visible():
            return

        self.setupGLState()
        with self._light_shader, self._light_vao:
            self._light_vbo.commit()
            self._light_shader.set_uniform("proj_view", camera.get_proj_view_matrix().glData, "mat4")
            self._light_shader.set_uniform("model", self.model_matrix().glData, "mat4")
            self._light_shader.set_uniform("_lightColor", self.diffuse, "vec3")
            gl.glLineWidth(2)
            gl.glDrawArrays(gl.GL_LINES, 0, 2)

    def toDict(self) -> dict:
        ret = super().toDict()
        ret["position2"] = list(self.position2)
        return ret

class LightMixin():

    @property
    def light_count(self):
        return len(self.lights)

    def addLight(self, light: Sequence[PointLight]):
        if isinstance(light, PointLight):
            light = [light]
        self.lights.extend(light)

        for l in light:
            l.add_shadow_item(self)

    def removeAllLight(self):
        """
        移除所有光源
        """
        for light in self.lights:
            light.remove_shadow_item(self)
        self.lights.clear()

    def removeLight(self, light):
        """
        移除一个光源
        """
        if light in self.lights:
            light.remove_shadow_item(self)
            self.lights.remove(light)

    def setupLight(self, shader: Shader):
        """设置光源 uniform 属性
        """
        for i, light in enumerate(self.lights):
            light.set_uniform(shader, "pointLight[" + str(i) + "]")
        shader.set_uniform("nr_point_lights", len(self.lights), "int")



vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 proj_view;
uniform mat4 model;

void main()
{
    gl_Position = proj_view * model * vec4(aPos, 1.0f);
}
"""

shadow_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

void main()
{
    gl_Position = proj * view * model * vec4(aPos, 1.0);
    gl_Position = gl_Position / gl_Position.w;
    if(gl_Position.z < -1.0 || gl_Position.z > 1.0)
        gl_Position.z = 1.0;
}
"""

empty_fragment_shader = """
#version 330 core
void main()
{
    // gl_FragDepth = gl_FragCoord.z;
}
"""


fragment_shader = """
#version 330 core
out vec4 FragColor;

uniform vec3 _lightColor;

void main() {
    FragColor = vec4(_lightColor, 1.0);
}
"""

light_fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;
in vec4 oColor;

uniform vec3 ViewPos;

struct Material {
    bool disable;  // 禁用材质时使用 oColor
    float opacity;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    bool use_texture;
    sampler2D tex_diffuse;
};
uniform Material material;

struct PointLight {
    int type;
    vec3 position;
    vec3 position2;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool directional;

    bool renderShadow;
    sampler2D shadowMap;
    mat4 lightSpaceMatrix;
    float bias;
    float sampleNum;
};
#define MAX_POINT_LIGHTS 10
uniform PointLight pointLight[MAX_POINT_LIGHTS];
uniform int nr_point_lights;

vec2 poissonDisk[16] = vec2[](
   vec2( -0.94201624, -0.39906216 ),    vec2( 0.94558609, -0.76890725 ),
   vec2( -0.094184101, -0.92938870 ),   vec2( 0.34495938, 0.29387760 ),
   vec2( -0.91588581, 0.45771432 ),     vec2( -0.81544232, -0.87912464 ),
   vec2( -0.38277543, 0.27676845 ),     vec2( 0.97484398, 0.75648379 ),
   vec2( 0.44323325, -0.97511554 ),     vec2( 0.53742981, -0.47373420 ),
   vec2( -0.26496911, -0.41893023 ),    vec2( 0.79197514, 0.19090188 ),
   vec2( -0.24188840, 0.99706507 ),     vec2( -0.81409955, 0.91437590 ),
   vec2( 0.19984126, 0.78641367 ),      vec2( 0.14383161, -0.14100790 )
);

// 伪随机数生成器 based on a vec3 and an int.
float random(vec3 seed, int i){
	vec4 seed4 = vec4(seed,i);
	float dot_product = dot(seed4, vec4(12.9898,78.233,45.164,94.673));
	return fract(sin(dot_product) * 43758.5453);
}


float ShadowCalculation(PointLight light, vec3 fragPos)
{
    vec4 fragPosLightSpace = light.lightSpaceMatrix * vec4(fragPos, 1.0);
    // 执行透视除法
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // 变换到[0,1]的范围
    projCoords = projCoords * 0.5 + 0.5;
    // 取得最近点的深度(使用[0,1]范围下的fragPosLight当坐标)
    float closestDepth = texture(light.shadowMap, projCoords.xy).r;
    // 取得当前片段在光源视角下的深度
    float currentDepth = projCoords.z;
    // 检查当前片段是否在阴影中
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(light.shadowMap, 0);

    for(int i=0; i<light.sampleNum; i++){
        int index = int(16.0*random(floor(fragPos.xyz*1000.0), i)) % 16;
        shadow += currentDepth - light.bias > texture( light.shadowMap, vec2(projCoords.xy + poissonDisk[index]*texelSize)).r ? 0.9 : 0.0;
        //shadow += 1 - texture( light.shadowMap, vec3(projCoords.xy + poissonDisk[index]*texelSize,  projCoords.z-0.004));
    }
    shadow /= light.sampleNum;
    if(projCoords.z > 1.0)
        shadow = 0.0;
    return shadow;
}

vec3 closestPointOnLine(vec3 pt1, vec3 ptA, vec3 ptB) {
    /// 计算线段 ptA-ptB 上到 pt1 的最近点 pt2

    vec3 lineVec = normalize(ptB - ptA);
    vec3 pointVec = pt1 - ptA;
    float projectionLength = dot(pointVec, lineVec);
    projectionLength = clamp(projectionLength, 0.0, length(ptB - ptA));
    return ptA + lineVec * projectionLength;
}

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewPos)
{
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 lightDir = vec3(0);
    vec3 lightPos = vec3(0);
    float attenuation = 1.0;
    float distance = 0.0;

    // 计算光源位置
    if (light.type == 0)
        lightPos = light.position;
    else if (light.type == 1)
        lightPos = closestPointOnLine(fragPos, light.position, light.position2);

    if (light.directional)
        lightDir = normalize(lightPos);
    else{
        lightDir = normalize(lightPos - fragPos);
        distance = length(lightPos - fragPos);
        attenuation = 1.0 / (light.constant + light.linear * distance +
                     light.quadratic * (distance * distance));
    }

    //vec3 halfwayDir = normalize(lightDir + viewDir);
    vec3 reflectDir = reflect(-lightDir, normal);
    // 漫反射着色
    float diff = max(dot(normal, lightDir), 0.0);
    // 镜面光着色
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), max(material.shininess, 1.0));

    // 合并结果
    vec3 ambient  = vec3(0);
    vec3 diffuse  = vec3(0);
    vec3 specular = vec3(0);
    if (material.disable) {
        ambient  = light.ambient  * oColor.rgb * 0.4;
        diffuse  = light.diffuse  * diff * oColor.rgb * 0.6;
        specular = light.specular * spec * oColor.rgb * 0.2;
    } else if (material.use_texture) {
        ambient  = light.ambient  * vec3(texture(material.tex_diffuse, TexCoords));
        diffuse  = light.diffuse  * diff * vec3(texture(material.tex_diffuse, TexCoords));
        specular = light.specular * spec * vec3(texture(material.tex_diffuse, TexCoords));
    } else {
        ambient  = light.ambient  * material.ambient;
        diffuse  = light.diffuse  * diff * material.diffuse;
        specular = light.specular * spec * material.specular;
    }

    // 阴影
    float shadow = 0.0;
    if (light.renderShadow) {
        //float bias = max(0.01 * (1.0 - dot(normal, lightDir)), 0.005);
        shadow = ShadowCalculation(light, fragPos);
    }
    return attenuation * (ambient + (1-shadow) * (specular + diffuse));
}

void main() {
    vec3 result = vec3(0);
    float opacity = 0;
    for(int i = 0; i < nr_point_lights; i++)
        result += CalcPointLight(pointLight[i], Normal, FragPos, ViewPos);

    if (nr_point_lights == 0){
        if(material.disable)
            result = oColor.rgb;
        else if(material.use_texture)
            result = vec3(texture(material.tex_diffuse, TexCoords));
        else
            result = material.diffuse;
    }

    if (material.disable) {
        opacity = oColor.a;
    } else {
        opacity = material.opacity;
    }

    FragColor = vec4(result, opacity);
    FragColor = clamp(FragColor, 0.0, 1.0);
    //float gamma = 2.2;
    //FragColor = vec4(pow(result, vec3(1.0 / gamma)), material.opacity);
}
"""
