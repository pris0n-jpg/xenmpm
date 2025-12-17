
from __future__ import annotations  # Python 3.7+
from typing import TYPE_CHECKING
import os
import numpy as np

if TYPE_CHECKING:
    from .XenseSensor import Sensor
from .sensorEnum import OutputType
from .dataProcessor import *
from .configManager import ConfigManager
from .sensorEnum import DAGType, ConnectionType
from .autoRefresher import AutoRefresher
from ..utils.decorator import deprecated


# Directed Acyclic Graph for image process 
class DAGBase:

    @property
    def reference_image(self):
        return self._reference_image
    
    @property
    def reference_image_binary(self):
        return self._reference_image_binary

    def _cal_rectify(self, cache):
        ret, frame = self.sensor._real_camera.get_raw_frame()  # NOTE check ret
        cache[OutputType.Rectify] = self.sensor._real_camera.rectify(frame)
        cache[OutputType.Raw] = frame
    
    def _cal_diff(self, cache):
        cache[OutputType.ImgFloat] = convertUint8ToInfer(cache[OutputType.Rectify], self.sensor.infer_size)
        cache[OutputType.ImgMarkerFree] = convertMarkerFree(self.sensor._infer_engine, cache[OutputType.ImgFloat])
        
        if self._reference_image is None:
            self._reference_image = cache[OutputType.ImgMarkerFree]
            
        cache[OutputType.ImgObjEnhance] = convertObjectEnhance(cache[OutputType.ImgMarkerFree], self._reference_image)
        cache[OutputType.Difference] = convertOriginSize(cache[OutputType.ImgObjEnhance] * 255, self.sensor.rectify_size).astype(np.uint8)
    
    def _cal_depth(self, cache):
        depth_float = convertDepth(self.sensor._infer_engine, cache[OutputType.ImgObjEnhance])
        cache[OutputType.Depth] = convertOriginSize(depth_float, self.sensor.rectify_size)

    def _cal_marker_enhance(self, cache):
        cache[OutputType.ImgMarkerEnhance] = convertMarkerEnhance(cache[OutputType.ImgFloat], cache[OutputType.ImgMarkerFree])
    
    def _cal_marker_2d(self, cache):        
        flow = predictMarker(self.sensor._infer_engine, cache[OutputType.ImgMarkerEnhance], self.reference_image_binary)
        self.sensor._flow_tracker.update(flow)
        cache[OutputType.Flow] = flow
        marker_2d_pos = np.array(self.sensor._flow_tracker.grid_curr, dtype=np.float32)
        cache[OutputType.Marker2D] = marker_2d_pos
        cache[OutputType.Marker2DFlip] = convertMarkerToFem(marker_2d_pos, self.sensor.grid_coord_size)
               
    def _cal_marker_3d(self, cache):
        cache[OutputType.Marker3DFlow] = self.sensor._fem.get_marker_flow(cache[OutputType.Marker2DFlip], np.flip(-cache[OutputType.Depth], 1))
        cache[OutputType.Marker3D] = self.sensor._fem.marker_xyz_init + cache[OutputType.Marker3DFlow]
    
    def _cal_marker_unorder(self, cache):
        cache[OutputType.MarkerUnorder] = detectMarker(
            cache[OutputType.ImgMarkerEnhance], 
            self.sensor._config_manager.marker_config.min_area,
            self.sensor.grid_coord_size
        )

    def _cal_infer(self, cache):
        cache[OutputType.ImgFloat] = convertUint8ToInfer(cache[OutputType.Rectify], self.sensor.infer_size)
        img_marker_free, img_obj_enhance, img_marker_enhance, img_depth, flow = self.sensor._infer_engine.inferAllInOne(
            cache[OutputType.ImgFloat], self._reference_image, self._reference_image_binary
        )
        cache[OutputType.ImgMarkerFree] = img_marker_free
        cache[OutputType.ImgMarkerEnhance] = img_marker_enhance
        cache[OutputType.Flow] = flow
        cache[OutputType.Depth] = img_depth  # float, [160, 96]
        cache[OutputType.Difference] = (img_obj_enhance * 255).astype(np.uint8)

    def _cal_marker_allinone(self, cache):
        self.sensor._flow_tracker.update(cache[OutputType.Flow])
        marker_2d_pos = np.array(self.sensor._flow_tracker.grid_curr, dtype=np.float32)
        cache[OutputType.Marker2D] = marker_2d_pos
        cache[OutputType.Marker2DFlip] = convertMarkerToFem(marker_2d_pos, self.sensor.grid_coord_size)
    
    @deprecated("Use _cal_force_approx instead")
    def _cal_force(self, cache):
        cache[OutputType.Force] = self.sensor._fem.get_mesh_force(cache[OutputType.Mesh3DFlow])

    @deprecated("Use _cal_force_approx instead")
    def _cal_force_norm(self, cache):
        cache[OutputType.ForceNorm] = self.sensor._fem.get_mesh_force(cache[OutputType.Mesh3DNorm])

    def _cal_force_approx(self, cache):
        cache[OutputType.Force], cache[OutputType.ForceNorm] = self.sensor._fem.get_approx_force(
            cache[OutputType.Marker2DFlip], 
            np.flip(-cache[OutputType.Depth], 1), 
            self.sensor._config_manager.sensor_config.force_calibrate_param
        )

    def _cal_force_resultant(self, cache):
        cache[OutputType.ForceResultant] = self.sensor._fem.get_force_resultant(cache[OutputType.Force])

    def _cal_mesh_flow(self, cache):
        cache[OutputType.Mesh3DFlow], cache[OutputType.Mesh3DNorm] = self.sensor._fem.get_mesh_flow(
            cache[OutputType.Marker2DFlip], 
            np.flip(-cache[OutputType.Depth], 1)
        )
        cache[OutputType.Mesh3D] = cache[OutputType.Mesh3DFlow] + self.sensor._fem.top_xyz

    def _get_marker_2d_init(self, cache):
        cache[OutputType.Marker2DInit] = self.sensor._flow_tracker.grid_init
    
    def _get_marker_3d_init(self, cache):
        cache[OutputType.Marker3DInit] = self.sensor._fem.marker_xyz_init

    def _get_mesh_3d_init(self, cache):
        cache[OutputType.Mesh3DInit] = self.sensor._fem.top_xyz  
                
    def __init__(self, sensor: Sensor):        
        self.sensor = sensor
        self._otvalue = int(os.environ.get("XENSE_OTVALUE", 100))
        self._reference_image = None
        # self._auto_refresher = AutoRefresher(self)
        if self.sensor._config_manager.sensor_config.connection_type != ConnectionType.Remote:
            self.resetRefernceImage()
            # self._auto_refresher.start()
    
    def resetRefernceImage(self, image_float=None, image_ref=None):
        if image_float is None:
            image = self.sensor._real_camera.get_frame()[1]
            image_float = convertUint8ToInfer(image, self.sensor.infer_size)

        if image_ref is None:
            image_ref = convertMarkerFree(self.sensor._infer_engine, image_float)
            
        self._reference_image = image_ref
        self._reference_image_binary = convertMarkerEnhance(image_float, image_ref)

    def selectSensorInfo(self, *args, **kwargs):
        if "cache" in kwargs:
            cache = kwargs["cache"]
            cache["Private"] = 1
        else:
            cache = {"Private": 1}
        ret = []
        for output_type in args:
            try:  # 禁止访问私有数据类型  
                assert isinstance(output_type, OutputType) and output_type.value < self._otvalue
            except Exception as e:
                raise Exception(f"Bad OutputType: {output_type}")
                        
            self._getOutputType(output_type, cache)
            ret.append(cache[output_type])

        if len(ret) == 1:
            return ret[0]
        return ret
    
    def _getOutputType(self, data_type: "Sensor.OutputType", cache):
        """
        计算 data_type 并填入 cache
        """
        try:
            assert "Private" in cache
        except:
            raise Exception("Private function cannot be called")
                
        if data_type in cache:
            return
        
        for require_type in self._dict_of_find[data_type]["require"]:
            if require_type not in cache:
                self._getOutputType(require_type, cache)
        
        self._dict_of_find[data_type]["function"](cache)

class SplitDAGRunner(DAGBase):
    def __init__(self, sensor):
        super().__init__(sensor)
        OT = OutputType
        self._dict_of_find = {            
            OT.Raw:              {"function": self._cal_rectify,        "require": []},
            OT.Rectify:          {"function": self._cal_rectify,        "require": []},
            OT.Difference:       {"function": self._cal_diff,           "require": [OT.Rectify]},
            OT.Depth:            {"function": self._cal_depth,          "require": [OT.Difference]},
            OT.Marker2D:         {"function": self._cal_marker_2d,      "require": [OT.ImgMarkerEnhance]},
            OT.Marker2DFlip:     {"function": self._cal_marker_2d,      "require": [OT.ImgMarkerEnhance]},
            OT.Marker2DInit:     {"function": self._get_marker_2d_init, "require": []},
            OT.Marker3D:         {"function": self._cal_marker_3d,      "require": [OT.Depth, OT.Marker2D]},
            OT.Marker3DFlow:     {"function": self._cal_marker_3d,      "require": [OT.Depth, OT.Marker2D]},
            OT.Marker3DInit:     {"function": self._get_marker_3d_init, "require": []},
            OT.MarkerUnorder:    {"function": self._cal_marker_unorder, "require": [OT.ImgMarkerEnhance]},
            OT.Force:            {"function": self._cal_force_approx,   "require": [OT.Depth, OT.Marker2D]}, 
            OT.ForceNorm:        {"function": self._cal_force_approx,   "require": [OT.Depth, OT.Marker2D]},        
            OT.ForceResultant:   {"function": self._cal_force_resultant,"require": [OT.Force]}, 
            OT.Mesh3D:           {"function": self._cal_mesh_flow,      "require": [OT.Depth, OT.Marker2D]},
            OT.Mesh3DFlow:       {"function": self._cal_mesh_flow,      "require": [OT.Depth, OT.Marker2D]},
            OT.Mesh3DInit:       {"function": self._get_mesh_3d_init,   "require": []},
            OT.ImgMarkerEnhance: {"function": self._cal_marker_enhance, "require": [OT.Difference]},
            OT.Flow:             {"function": self._cal_marker_2d,      "require": [OT.ImgMarkerEnhance]},
        }


class AllInOneDAGRunner(DAGBase):
    def __init__(self, sensor):
        super().__init__(sensor)
        OT = OutputType
        self._dict_of_find = {            
            OT.Raw:              {"function": self._cal_rectify,        "require": []},
            OT.Rectify:          {"function": self._cal_rectify,        "require": []},
            OT.Difference:       {"function": self._cal_infer,          "require": [OT.Rectify]},
            OT.Depth:            {"function": self._cal_infer,          "require": [OT.Rectify]},
            OT.Marker2D:         {"function": self._cal_marker_allinone,"require": [OT.Difference]},
            OT.Marker2DFlip:     {"function": self._cal_marker_allinone,"require": [OT.Difference]},
            OT.Marker2DInit:     {"function": self._get_marker_2d_init, "require": []},
            OT.Marker3D:         {"function": self._cal_marker_3d,      "require": [OT.Depth, OT.Marker2D]},
            OT.Marker3DFlow:     {"function": self._cal_marker_3d,      "require": [OT.Depth, OT.Marker2D]},
            OT.Marker3DInit:     {"function": self._get_marker_3d_init, "require": []},
            OT.MarkerUnorder:    {"function": self._cal_marker_unorder, "require": [OT.Difference]},
            OT.Force:            {"function": self._cal_force_approx,   "require": [OT.Depth, OT.Marker2D]}, 
            OT.ForceNorm:        {"function": self._cal_force_approx,   "require": [OT.Depth, OT.Marker2D]},        
            OT.ForceResultant:   {"function": self._cal_force_resultant,"require": [OT.Force]},         
            OT.Mesh3D:           {"function": self._cal_mesh_flow,      "require": [OT.Depth, OT.Marker2D]},
            OT.Mesh3DFlow:       {"function": self._cal_mesh_flow,      "require": [OT.Depth, OT.Marker2D]},
            OT.Mesh3DInit:       {"function": self._get_mesh_3d_init,   "require": []},
        }


class RemoteDAGRunner(SplitDAGRunner):
                
    def __init__(self, sensor): 
        super().__init__(sensor)
        OT = OutputType
        self._remote_types = [OT.Raw, OT.Rectify, OT.Difference, OT.Depth, OT.Marker2D, OT.Marker2DFlip]
        
        self._dict_of_find = {            
            OT.Marker2DInit:     {"function": self._get_marker_2d_init, "require": []},
            OT.Marker3D:         {"function": self._cal_marker_3d,      "require": [OT.Depth, OT.Marker2DFlip]},
            OT.Marker3DFlow:     {"function": self._cal_marker_3d,      "require": [OT.Depth, OT.Marker2DFlip]},
            OT.Marker3DInit:     {"function": self._get_marker_3d_init, "require": []},
            OT.MarkerUnorder:    {"function": self._cal_marker_unorder, "require": [OT.ImgMarkerEnhance]},
            OT.Force:            {"function": self._cal_force_approx,   "require": [OT.Depth, OT.Marker2D]}, 
            OT.ForceNorm:        {"function": self._cal_force_approx,   "require": [OT.Depth, OT.Marker2D]},        
            OT.ForceResultant:   {"function": self._cal_force_resultant,"require": [OT.Force]},     
            OT.Mesh3D:           {"function": self._cal_mesh_flow,      "require": [OT.Depth, OT.Marker2DFlip]},
            OT.Mesh3DFlow:       {"function": self._cal_mesh_flow,      "require": [OT.Depth, OT.Marker2DFlip]},
            OT.Mesh3DInit:       {"function": self._get_mesh_3d_init,   "require": []},
        }

    def resetRefernceImage(self):
        """
        通知远程端重置参考图像
        """
        self.sensor._real_camera.reset_reference()
    
    def selectSensorInfo(self, *args) -> list:
        return super().selectSensorInfo(*args, cache=self.__fetch_remote_data(args))

    def __fetch_remote_data(self, args: list):
        # 迭代代替 dfs
        types_from_remote = []
        types_visited = set(args)
        stack = list(args)
        
        while stack:
            data_type = stack.pop()
            if data_type in self._remote_types:
                types_from_remote.append(data_type)
                continue
            
            for require_type in self._dict_of_find[data_type]["require"]:
                if require_type not in types_visited:
                    types_visited.add(require_type)
                    stack.append(require_type)

        # 获取远程数据
        if types_from_remote:
            cache = self.sensor._real_camera.get_data(types_from_remote)
            if cache is None:
                raise Exception("Remote data fetch failed")
        else:
            cache = {}       
       
        return cache


class DAGManager(DAGBase):
    @classmethod
    def create(cls, sensor: Sensor):
        # check config for remote
        if sensor._config_manager.sensor_config.connection_type == ConnectionType.Remote:
            return RemoteDAGRunner(sensor)
        # check infer config for local
        if sensor._config_manager.infer_config.dag_type == DAGType.Split:
            return SplitDAGRunner(sensor)
        else:
            return AllInOneDAGRunner(sensor)
        