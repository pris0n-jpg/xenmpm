import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator

import sys
from xensesdk import ExampleView
from xensesdk import Sensor
import cv2

flag = 1

class MarkerInterpolator:
    def __init__(self, marker_init, marker_row_col = [20, 11], marker_dx_dy_mm = [1.31, 1.31]):
        self.marker_row_col = np.array(marker_row_col)
        self.marker_dx_dy_mm = np.array(marker_dx_dy_mm)
        self.center_offset = np.array([199, 349])
        self.ratio = np.array([17.2/400, 28.5/700])
        self.marker_init = marker_init
        self.source_grid = self.get_source_grid(marker_init)
        self.target_grid = self.get_target_grid()
        self.source_xy = self.source_grid.reshape(-1, 2)
        self.target_xy = self.target_grid.reshape(-1, 2)
        self.grid_shape = self.target_grid.shape[:2]

    def get_target_grid(self):
        y_ed, x_ed = self.marker_dx_dy_mm*(self.marker_row_col-1)/2
        x = np.linspace(-x_ed,x_ed,self.marker_row_col[1])
        y = np.linspace(-y_ed,y_ed,self.marker_row_col[0])
        X,Y = np.meshgrid(x, y)
        return np.stack([X,Y],axis=2)
    
    def get_source_grid(self, marker_init):
        return (marker_init-self.center_offset)*self.ratio
    
    def interpolate(self, marker_uv):
        marker_uv_mm = (marker_uv - self.marker_init)*self.ratio
        source_uv = marker_uv_mm.reshape(-1, 2)
        rbf_u = RBFInterpolator(self.source_xy, source_uv[:, 0], kernel='thin_plate_spline', smoothing=0.0)
        rbf_v = RBFInterpolator(self.source_xy, source_uv[:, 1], kernel='thin_plate_spline', smoothing=0.0)
        interpolated_u = rbf_u(self.target_xy)
        interpolated_v = rbf_v(self.target_xy)
        interpolated_values = np.stack([interpolated_u, interpolated_v], axis=1)
        interpolated_values = interpolated_values.reshape(self.grid_shape[0], self.grid_shape[1], 2)
        return interpolated_values



def interpolate_to_regular_grid(source_points, source_values, target_points):
    """
    将不规则分布的源网格上的数值插值到规则目标网格上
    
    Parameters:
    - source_points: 源网格点坐标，形状为(H,W,2)，包含(x,y)坐标
    - source_values: 源网格上的数值，形状为(H,W,2)，包含(u,v)位移
    - target_points: 目标规则网格点坐标，形状为(H,W,2)，包含(x,y)坐标
    
    Returns:
    - interpolated_values: 插值后的数值，形状为(H,W,2)
    """
    # 获取网格形状
    grid_shape = target_points.shape[:2]
    
    # 将源点和数值展平
    source_xy = source_points.reshape(-1, 2)  # (N, 2)
    source_uv = source_values.reshape(-1, 2)  # (N, 2)
    target_xy = target_points.reshape(-1, 2)  # (M, 2)
    
    # 分别对u和v分量进行插值
    interpolated_u = griddata(
        points=source_xy, 
        values=source_uv[:, 0], 
        xi=target_xy, 
        method='linear',
        fill_value=0.0  # 使用0填充超出范围的点
    )
    
    interpolated_v = griddata(
        points=source_xy, 
        values=source_uv[:, 1], 
        xi=target_xy, 
        method='linear',
        fill_value=0.0
    )
    
    # 处理可能的NaN值，用最近邻插值填充
    nan_mask_u = np.isnan(interpolated_u)
    nan_mask_v = np.isnan(interpolated_v)
    
    if np.any(nan_mask_u):
        interpolated_u[nan_mask_u] = griddata(
            points=source_xy, 
            values=source_uv[:, 0], 
            xi=target_xy[nan_mask_u], 
            method='nearest'
        )
    
    if np.any(nan_mask_v):
        interpolated_v[nan_mask_v] = griddata(
            points=source_xy, 
            values=source_uv[:, 1], 
            xi=target_xy[nan_mask_v], 
            method='nearest'
        )
    
    # 重新组合并reshape
    interpolated_values = np.stack([interpolated_u, interpolated_v], axis=1)  # (M, 2)
    interpolated_values = interpolated_values.reshape(grid_shape[0], grid_shape[1], 2)
    
    return interpolated_values

def interpolate_to_regular_grid_fast(source_points, source_values, target_points):
    """
    快速插值版本，适用于源网格和目标网格形状相同且分布较规则的情况
    使用RBF (径向基函数) 插值，比griddata更快
    
    Parameters:
    - source_points: 源网格点坐标，形状为(H,W,2)
    - source_values: 源网格上的数值，形状为(H,W,2)
    - target_points: 目标规则网格点坐标，形状为(H,W,2)
    
    Returns:
    - interpolated_values: 插值后的数值，形状为(H,W,2)
    """
    from scipy.interpolate import RBFInterpolator
    
    # 获取网格形状
    grid_shape = target_points.shape[:2]
    
    # 将数据展平
    source_xy = source_points.reshape(-1, 2)
    source_uv = source_values.reshape(-1, 2)
    target_xy = target_points.reshape(-1, 2)
    
    # 使用RBF插值器，multiquadric核函数通常效果较好
    rbf_u = RBFInterpolator(source_xy, source_uv[:, 0], kernel='thin_plate_spline', smoothing=0.0)
    rbf_v = RBFInterpolator(source_xy, source_uv[:, 1], kernel='thin_plate_spline', smoothing=0.0)
    
    # 执行插值
    interpolated_u = rbf_u(target_xy)
    interpolated_v = rbf_v(target_xy)
    
    # 重新组合并reshape
    interpolated_values = np.stack([interpolated_u, interpolated_v], axis=1)
    interpolated_values = interpolated_values.reshape(grid_shape[0], grid_shape[1], 2)
    
    return interpolated_values


def main():
    sensor_0 = Sensor.create(0)
    View = ExampleView(sensor_0)
    View2d = View.create2d(Sensor.OutputType.Difference, Sensor.OutputType.Depth, Sensor.OutputType.Marker2D)

    def callback():
        force, res_force, mesh_init, rectify_real, diff, depth = sensor_0.selectSensorInfo(
            Sensor.OutputType.Force, 
            Sensor.OutputType.ForceResultant,
            Sensor.OutputType.Mesh3DInit,
            Sensor.OutputType.Rectify, 
            Sensor.OutputType.Difference, 
            Sensor.OutputType.Depth,
        )

        marker_img = sensor_0.drawMarkerMove(rectify_real)
        View2d.setData(Sensor.OutputType.Marker2D, marker_img)

        View2d.setData(Sensor.OutputType.Difference, diff)
        View2d.setData(Sensor.OutputType.Depth, depth)
        View.setForceFlow(force, res_force, mesh_init)
        View.setDepth(depth)
        

    # 获取其他信息
       
    global flag 

    ratio = [17.2/400, 28.5/700]
    marker_row_col = np.array([20, 11])
    marker_dx_dy_mm = [1.31, 1.31]
    center = [199, 349]
    y_ed, x_ed = marker_dx_dy_mm*(marker_row_col-1)/2
    print(y_ed,x_ed)
    x = np.linspace(-x_ed,x_ed,marker_row_col[1])
    y = np.linspace(-y_ed,y_ed,marker_row_col[0])
    X,Y = np.meshgrid(x, y)
    real_marker_mm = np.stack([X,Y],axis=2)
    # print(real_marker_mm.shape)
    # interpolator = MarkerInterpolator(marker_init, marker_row_col, marker_dx_dy_mm)
    

    if flag == 1:
        marker,marker_init= sensor_0.selectSensorInfo(Sensor.OutputType.Marker2D, Sensor.OutputType.Marker2DInit)
        print(marker.shape, marker_init.shape)

        flag = 0
        marker_geter = MarkerInterpolator(marker_init)
        marker_init_mm = (marker_init-center)*ratio
        marker_uv_mm = (marker - marker_init)*ratio

        # print('\nmarker_x',marker_init_mm[:,:,0])
        
        # 插值：将不规则分布的marker_init_mm上的数值marker_uv_mm插值到规则网格real_marker_mm上
        # 选择插值方法：
        # 1. interpolate_to_regular_grid: 通用方法，使用线性插值+最近邻填充
        # 2. interpolate_to_regular_grid_fast: 快速方法，使用RBF插值，适合规则网格
        # interpolated_uv_mm = interpolate_to_regular_grid(marker_init_mm, marker_uv_mm, real_marker_mm)
        # interpolated_uv_mm = interpolate_to_regular_grid_fast(marker_init_mm, marker_uv_mm, real_marker_mm)  # 备选快速方法
        interpolated_uv_mm = marker_geter.interpolate(marker)

        print(f"插值前后数值范围对比:")
        print(f"  原始u范围: [{np.min(marker_uv_mm[:,:,0]):.3f}, {np.max(marker_uv_mm[:,:,0]):.3f}]")
        print(f"  插值u范围: [{np.min(interpolated_uv_mm[:,:,0]):.3f}, {np.max(interpolated_uv_mm[:,:,0]):.3f}]")
        print(f"  原始v范围: [{np.min(marker_uv_mm[:,:,1]):.3f}, {np.max(marker_uv_mm[:,:,1]):.3f}]")
        print(f"  插值v范围: [{np.min(interpolated_uv_mm[:,:,1]):.3f}, {np.max(interpolated_uv_mm[:,:,1]):.3f}]")
        
        # 可选：可视化插值结果对比 (取消注释以启用)
        # visualize_interpolation_comparison(marker_uv_mm, interpolated_uv_mm, 'interpolation_comparison.png')


    View.setCallback(callback)
    View.show()
    sensor_0.release()
    sys.exit()


if __name__ == '__main__':
    main()







