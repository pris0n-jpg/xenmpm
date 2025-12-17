from .GLViewWidget import GLViewWidget
from .transform3d import Matrix4x4
from .items import *
from qtpy.QtCore import Qt
import numpy as np
from .utils.colormap import cm
import cv2

__all__ = [
    "DefaultViewWidget",
    "QSurfaceWidget",
    "QPointCloudWidget",
    "QQuiverWidget",
    "QSurfaceQuiverWidget",
    "QGelSlimWidget"
]

class DefaultViewWidget(GLViewWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.camera.set_vector6d((0,0,-1000, 0, -75, -90))
        self.camera.perspective_far = 600
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # 坐标轴
        self.axis = GLAxisItem(size=(100, 100, 100), tip_size=40)
        self.axis_fixed = GLAxisItem(size=(0.7, 0.7, 0.7), tip_size=1, fix_to_corner=True)
        self.axis.translate(-240, -320, 0)
        self.addItems([self.axis, self.axis_fixed])
        # 网格曲面
        self.grid1 = GLGridItem(size=(480, 640), spacing=(40, 40), color=(0.8, 0.8, 0.8, 1))
        self.grid1.rotate(90, 1, 0, 0)
        self.grid1.translate(0, 0, -5)
        self.addItem(self.grid1)
        self.grid1.setVisible(True)

        self.grid2 = GLGridItem(size=(480, 640), spacing=(160, 160), color=(0.8, 0.8, 0.8, 1))
        self.grid2.rotate(90, 1, 0, 0)
        self.grid2.translate(0, 0, 255)
        self.addItem(self.grid2)
        self.grid2.setVisible(False)

    def set_grid(self, grid1, grid2, grid1_z, grid2_z):
        self.grid1.setVisible(grid1)
        self.grid1.resetTransform()
        self.grid1.translate(0, 0, grid1_z)

        self.grid2.setVisible(grid2)
        self.grid2.resetTransform()
        self.grid2.translate(0, 0, grid2_z)


def get_color(zmap, colormap='coolwarm') -> np.ndarray:
    """
    Get color from zmap.

    Parameters:
    - zmap : 2d array, 深度图
    - colormap : str, optional, default: 'coolwarm', 颜色映射

    Returns:
    - np.ndarray of float32,  (N, 3) 的颜色数组
    """
    colormap = eval("cm." + colormap)
    return colormap(zmap).reshape(-1, 3)


class QSurfaceWidget(DefaultViewWidget):

    def __init__(self, parent=None, x_size=400):
        super().__init__(parent=parent)
        self.surface = GLColorSurfaceItem(x_size=x_size)
        self.surface.rotate(90, 0, 0, 1)
        self.addItem(self.surface)

    def setData(self, img, channel=0, colormap:str='coolwarm', cm_scale=1., cm_bias=0.):
        if img.shape[-1] == 3:
            img = img[..., channel]
        color = get_color(img / cm_scale + cm_bias, colormap)

        self.surface.setData(img, color)
        self.update()


class QPointCloudWidget(DefaultViewWidget):

    def __init__(self, parent=None, x_size=640, point_size=50, xy_cnt=40):
        super().__init__(parent)
        self.pc = GLScatterPlotItem(size=point_size, auto_size=True)
        self.pc.translate(-x_size*3/8, -x_size/2, 0)
        self.pc.applyTransform(
            Matrix4x4(np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])),
            local=True
        )
        self.addItem(self.pc)
        self.xy_cnt = xy_cnt
        self.x_size = x_size

    def setData(self, img, channel=0, colormap:str='coolwarm', cm_scale=1., cm_bias=0.):
        if img.shape[-1] == 3:
            img = img[..., channel]

        scale = self.x_size / img.shape[1]

        x_range = np.linspace(0, img.shape[1]-1, self.xy_cnt, dtype=int)
        y_range = np.linspace(0, img.shape[0]-1, self.xy_cnt, dtype=int)
        x, y = np.meshgrid(x_range, y_range)
        zmap = img[y, x]

        color = get_color(zmap / cm_scale + cm_bias, colormap)

        point_cloud = np.stack([x.ravel() * scale, y.ravel() * scale, zmap.ravel()], axis=1)

        self.pc.setData(point_cloud, color)
        self.update()


class QQuiverWidget(DefaultViewWidget):

    def __init__(self, parent=None, tip_size=[1.2, 1.5], width=2.5):
        super().__init__(parent)
        vert, ind, uv, norm = sphere(width, 8, 8, calc_uv_norm=True)
        sphere_mesh = Mesh(vert, ind, normals=norm)
        self.sphere = GLInstancedMeshItem(sphere_mesh, color=[0.1,0.1,0.1,1], glOptions="ontop")
        self.sphere.setDepthValue(1)
        self.quiver = GLArrowPlotItem(tip_size=tip_size, color=None, width=width)

        tr = Matrix4x4.fromTranslation(-240, -320, 0).applyTransform(
            Matrix4x4(np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])),
            local=True
        )

        self.sphere.applyTransform(tr)
        self.quiver.applyTransform(tr)

        self.addItem(self.sphere)
        self.addItem(self.quiver)

    def setData(self, start_pts, end_pts):
        arrow_lengths = np.linalg.norm(end_pts - start_pts, axis=1)
        arrow_colors = cm.autumn(0.8 - (arrow_lengths / 50))

        self.quiver.setData(start_pts, end_pts, arrow_colors)
        self.sphere.setData(pose = np.array(start_pts).reshape(-1, 3))
        self.update()


class QSurfaceQuiverWidget(QSurfaceWidget):

    def __init__(self, parent=None, x_size=640, tip_size=[1.2, 1.5], width=2.5):
        super().__init__(parent, x_size)
        vert, ind, uv, norm = sphere(width, 8, 8, calc_uv_norm=True)
        sphere_mesh = Mesh(vert, ind, normals=norm)
        self.sphere = GLInstancedMeshItem(sphere_mesh, color=[0.1,0.1,0.1,1], glOptions="ontop")
        self.quiver = GLArrowPlotItem(tip_size=tip_size, color=None, width=width)
        self.sphere.setDepthValue(1)

        tr = Matrix4x4.fromTranslation(-x_size*3/8, -x_size/2, 0).applyTransform(
            Matrix4x4(np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])),
            local=True
        )
        self.sphere.applyTransform(tr)
        self.quiver.applyTransform(tr)

        self.addItems([self.sphere, self.quiver])

    def setData(self, img, start_pts, end_pts,
                channel=0, colormap:str="coolwarm", cm_scale=1, cm_bias=0, **kwargs):
        if img.shape[-1] == 3:
            img = img[..., channel]

        color = get_color(img / cm_scale + cm_bias, colormap)

        arrow_lengths = np.linalg.norm(end_pts - start_pts, axis=1)
        arrow_colors = cm.autumn(0.8 - (arrow_lengths / 50))

        self.surface.setData(zmap=img, color=color)
        self.quiver.setData(start_pos=end_pts, end_pos=2*end_pts-start_pts, color=arrow_colors)
        self.sphere.setData(pose = end_pts)
        self.update()


class QGelSlimWidget(GLViewWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.camera.set_vector6d((-0.08, 0.4, -4, -35, -16, -20))
        self.camera.perspective_far = 100

        self.light = PointLight(pos=[0,0, 1], ambient=(0.2,0.2,0.2), diffuse=(1, 1, 1), directional=True, render_shadow=False, visible=False)
        self.ax = GLAxisItem(size=(0.4, 0.4, 0.4), tip_size=0.2).translate(-0.9, -0.9, 0)
        self.axis_fixed = GLAxisItem(size=(0.7, 0.7, 0.7), tip_size=1, fix_to_corner=True)
        self.grid = GLGridItem(size=(20, 20), spacing=(1, 1), lights=self.light).rotate(90, 1, 0, 0).translate(0, 0, -1.5)

        self.model = GLGelSlimItem(lights=[self.light])

        vert, ind, uv, norm = sphere(0.012, calc_uv_norm=True)
        sphere_mesh = Mesh(vert, ind, normals=norm)
        self.pointcloud = GLInstancedMeshItem(sphere_mesh, lights=[self.light], color=[1,0.7,0.,1], glOptions='translucent')
        self.arrow = GLArrowPlotItem(None, None, tip_size=[0.004, 0.007], tip_pos=-0.02, width=2, color=[1, 0, 0, 1])

        tr = Matrix4x4.fromTranslation(-0.675*3/4, -0.675, 0).applyTransform(
            Matrix4x4(np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])),
            local=True
        )
        self.pointcloud.applyTransform(tr)
        self.arrow.applyTransform(tr)

        self.addItems([self.light, self.ax, self.axis_fixed, self.grid, self.model, self.pointcloud, self.arrow])

    def setData(self, zmap, start_pts, end_pts, **kwargs):
        if zmap.ndim == 3:
            return
        zmap = zmap.astype(np.float32)
        h, w = zmap.shape[0:2]
        if h > 400:
            zmap = cv2.resize(zmap, (480, 360), interpolation=cv2.INTER_NEAREST)
        zmap = np.pad(zmap, ((1, 1), (1, 1)), mode='constant', constant_values=0)

        scale = self.model.gelslim_gel._x_size / w
        self.model.setDepth(zmap=-zmap)

        if start_pts is not None and end_pts is not None:
            self.pointcloud.setVisible(True)
            self.arrow.setVisible(True)
            start_pts[:, 2] = -start_pts[:, 2]
            end_pts[:, 2] = -end_pts[:, 2]
            self.pointcloud.setData(pose = np.array(end_pts*scale).reshape(-1, 3))
            self.arrow.setData(start_pos=start_pts*scale, end_pos=end_pts*scale)
        else:
            self.pointcloud.setVisible(False)
            self.arrow.setVisible(False)
        self.update()