import numpy as np
from scipy.interpolate import RBFInterpolator


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