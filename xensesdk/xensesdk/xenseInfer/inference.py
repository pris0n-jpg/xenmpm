from enum import Enum
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..xenseInterface.configManager import ConfigManager
from xensesdk import PROJ_DIR, Path
from xensesdk.utils.encryptDB import EncryptDB
from xensesdk.utils.decorator import infer_singleton
from xensesdk.utils import getEnvBool

def load_onnx(path, use_gpu=True):
    """
    从字典中加载指定的键，并将其导入为 ONNX session
    """
    import onnxruntime
    from onnxruntime import set_default_logger_severity
    set_default_logger_severity(4)
    if use_gpu:
        session = onnxruntime.InferenceSession(str(Path(path)), providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        session = onnxruntime.InferenceSession(str(Path(path)), providers=[ 'CPUExecutionProvider'])
    return session

def load_torch(path, **kwargs):
    from xensesdk.xenseInfer.nn_block import BaseModel
    path = Path(path)
    
    model = BaseModel(path / "config.yaml")
    model.to("cuda")
    model.load(path / "checkpoint_best.pth")
    model.fuse().half().eval()
    return model

def load_models(model_paths, load_func):
    model_paths = [str(p) for p in model_paths]
    depth_key = [p for p in model_paths if "depth" in p]
    diff_key = [p for p in model_paths if "diff" in p]
    flow_key = [p for p in model_paths if "flow" in p]
    depth_model = load_func(depth_key[0]) if depth_key else None
    diff_model = load_func(diff_key[0]) if diff_key else None
    flow_model = load_func(flow_key[0]) if flow_key else None
    return depth_model, diff_model, flow_model


class InferBase:
    
    @classmethod
    def create(cls, config: "ConfigManager", use_gpu: bool=True):
        
        sensor_type = config.sensor_config.sensor_type.name.lower()
        infer_width = config.infer_config.width
        infer_height = config.infer_config.height
        infer_size_str = f"{infer_height}x{infer_width}"
        infer_type = config.infer_config.infer_type.name.lower()
        if infer_type == "onnx":
            suffix = "edb"
        elif infer_type == "torch":
            suffix = "tedb"
        elif infer_type == "rknn":
            suffix = "redb"

        # load edb file        
        try:
            edb_path = PROJ_DIR / "xenseInterface/guiConfig" / f"{sensor_type}.{suffix}"
            if not edb_path.exists():  # 尝试从 kwargs 获取模型
                print(f"Cannot find {sensor_type} infer config")
                edb = None
            else:
                password = 'Qz8mmWz2VEQ6X5Ic'
                edb = EncryptDB(input_path=str(edb_path), password=password)
                print("Init infer engine")
            
            if infer_type == "onnx":
                return OnnxInfer(edb, infer_size_str, use_gpu=use_gpu, **config.kwargs)
            elif infer_type == "torch":
                return TorchInfer(edb, infer_size_str, **config.kwargs)
            elif infer_type == "rknn":
                return RknnInfer(edb, infer_size_str, **config.kwargs)
            else:
                print(f"Unknown backend type: {infer_type}.")
                
        except Exception as e:
            if getEnvBool("XENSE_DEBUG", False):
                print(e)
            raise Exception("cannot init infer engine.")

    def inferDepth(self, image_diff: np.ndarray )-> np.ndarray:
        raise NotImplementedError("inferDepth not implemented")

    def inferDiff(self, image: np.ndarray)->np.ndarray:
        raise NotImplementedError("inferDiff not implemented")
    
    def inferMarker(self, image_deform: np.ndarray, image_ref_binary: np.ndarray):
        raise NotImplementedError("inferMarker not implemented")
    
    def inferAllInOne(self, img_float, img_ref_float, img_ref_binary):
        pass


@infer_singleton
class OnnxInfer(InferBase):

    def __init__(self, edb: EncryptDB, infer_size_str: str, use_gpu: bool, **kwargs):

        self._depth_model, self._diff_model, self._flow_model = None, None, None
        if "models" in kwargs:
            self._depth_model, self._diff_model, self._flow_model = load_models(kwargs["models"], load_onnx)
        elif edb is not None:
            model_names = list(edb.keys())
            depth_key = [k for k in model_names if ("depth-"+infer_size_str in k and "onnx" in k)]
            diff_key = [k for k in model_names if ("diff-"+infer_size_str in k and "onnx" in k)]
            flow_key = [k for k in model_names if ("flow-"+infer_size_str in k and "onnx" in k)]
            self._depth_model = edb.load_to_onnx(depth_key[0], use_gpu=use_gpu) if depth_key else None  
            self._diff_model = edb.load_to_onnx(diff_key[0], use_gpu=use_gpu) if depth_key else None  
            self._flow_model = edb.load_to_onnx(flow_key[0], use_gpu=use_gpu) if depth_key else None
        
        if self._depth_model is None or self._diff_model is None or self._flow_model is None:
            raise Exception(f"No {infer_size_str} model found in edb or kwargs.")
        
        if "CUDAExecutionProvider" in self._depth_model.get_providers():
            print("infer session using GPU")
        else:
            print("infer session using CPU")

    def inferDepth(self, image_diff: np.ndarray)-> np.ndarray:
        '''
        image_diff: numpy array with type np.float32, range from 0 to 1, size [h, w, 3]
        return: numpy array with type np.float32,range from 0 to 1, size [h, w]
        '''
        image_diff = image_diff.transpose(2, 0, 1)[None, ...]  # [1,3,h,w]
        return self._depth_model.run(["image_depth"], {"image_diff": image_diff.astype(np.float16)})[0].squeeze().astype(np.float32)
    
    def inferDiff(self, image: np.ndarray)->np.ndarray:
        '''
        image_plain: numpy array with type np.float32, range from 0 to 1, size [h, w, 3]
        return: numpy array with type np.float32, range from 0 to 1, size [h, w, 3]
        '''
        image = image.transpose(2, 0, 1)[None, ...]  # [1,3,h,w]
        return self._diff_model.run(["image_plain"], {"image": image.astype(np.float16)})[0].squeeze().transpose(1,2,0).astype(np.float32)
    
    def inferMarker(self, image_deform: np.ndarray, image_ref_binary: np.ndarray):
        '''
        image_deform: numpy array with type np.float32, range from 0 to 1, size [h, w, 1]
        image_ref_binary: numpy array with type np.float32, range from 0 to 1, size [h, w, 1]
        return: numpy array with type np.float32, range from 0 to 1, size [hf, wf, 2](flow)
        '''
        image_deform = image_deform.transpose(2, 0, 1)[None, ...]
        image_ref_binary = image_ref_binary.transpose(2, 0, 1)[None, ...]
        flow = self._flow_model.run(["flow"], {"image_ref": image_ref_binary.astype(np.float16), "image_deform": image_deform.astype(np.float16)})
        return flow[0].squeeze().transpose(1,2,0).astype(np.float32)
    

@infer_singleton
class TorchInfer(InferBase):

    def __init__(self, edb: EncryptDB, infer_size_str: str, **kwargs):
        import torch
        self.torch = torch  # HACK: 避免不支持 torch 的平台报错
        
        self._depth_model, self._diff_model, self._flow_model = None, None, None
        if "models" in kwargs:
            self._depth_model, self._diff_model, self._flow_model = load_models(kwargs["models"], load_torch)
        elif edb is not None:
            model_names = list(edb.keys())
            depth_key = [k for k in model_names if ("depth-"+infer_size_str in k and "onnx" in k)]
            diff_key = [k for k in model_names if ("diff-"+infer_size_str in k and "onnx" in k)]
            flow_key = [k for k in model_names if ("flow-"+infer_size_str in k and "onnx" in k)]
            self._depth_model = edb.load_to_torch(depth_key[0]).eval() if depth_key else None  
            self._diff_model = edb.load_to_torch(diff_key[0]).eval() if depth_key else None  
            self._flow_model = edb.load_to_torch(flow_key[0]).eval() if depth_key else None
        
        if self._depth_model is None or self._diff_model is None or self._flow_model is None:
            raise Exception(f"No {infer_size_str} model found in edb or kwargs.")

    def inferDepth(self, image_diff: np.ndarray)-> np.ndarray:
        '''
        image_diff: numpy array with type np.float32, range from 0 to 1, size [h, w, 3]
        return: numpy array with type np.float32,range from 0 to 1, size [h, w]
        '''
        image_diff = image_diff.transpose(2, 0, 1)[None, ...]  # [1,3,h,w]
        image_diff = self.torch.tensor(image_diff, dtype=self.torch.float16, device="cuda")
        image_depth = self._depth_model.predict(image_diff, 0)
        return image_depth.squeeze().cpu().numpy().astype(np.float32)
    
    def inferDiff(self, image: np.ndarray)->np.ndarray:
        '''
        image_plain: numpy array with type np.float32, range from 0 to 1, size [h, w, 3]
        return: numpy array with type np.float32, range from 0 to 1, size [h, w, 3]
        '''
        image = image.transpose(2, 0, 1)[None, ...]  # [1,3,h,w]
        image = self.torch.tensor(image, dtype=self.torch.float16, device="cuda")
        image_plain = self._diff_model.predict(image, 0)
        return image_plain.squeeze().permute(1,2,0).cpu().numpy().astype(np.float32)
    
    def inferMarker(self, image_deform: np.ndarray, image_ref_binary: np.ndarray):
        '''
        image_deform: numpy array with type np.float32, range from 0 to 1, size [h, w, 1]
        image_ref_binary: numpy array with type np.float32, range from 0 to 1, size [h, w, 1]
        return: numpy array with type np.float32, range from 0 to 1, size [hf, wf, 2](flow)
        '''
        image_deform = image_deform.transpose(2, 0, 1)[None, ...]
        image_deform = self.torch.tensor(image_deform, dtype=self.torch.float16, device="cuda")
        image_ref_binary = image_ref_binary.transpose(2, 0, 1)[None, ...]
        image_ref_binary = self.torch.tensor(image_ref_binary, dtype=self.torch.float16, device="cuda")
        
        flow = self._flow_model.predict([image_ref_binary, image_deform], 0)

        return flow.squeeze().permute(1,2,0).cpu().numpy().astype(np.float32)
    
  
@infer_singleton
class RknnInfer(InferBase):

    def __init__(self, edb: EncryptDB, infer_size_str: str, **kwargs):
        model_names = list(edb.keys())
        
        model_key = [k for k in model_names if (infer_size_str in k and "rknn" in k)]
        if len(model_key) == 0:
            raise Exception(f"Unsupported infer size: {infer_size_str}.")
        else:
            model_key = model_key[0]            

        self._model = edb.load_to_rknn(model_key)

    def inferAllInOne(self, img_float, img_ref_float, img_ref_binary):
        img_float = img_float[None, ...]
        img_ref_float = img_ref_float[None, ...]
        img_ref_binary = img_ref_binary[None, ...]
        
        img_marker_free, img_obj_enhance, img_marker_enhance, img_depth, flow = self._model.inference(
            data_format='nhwc', inputs=[img_float, img_ref_float, img_ref_binary]
        )

        return (
            img_marker_free.squeeze().transpose(1,2,0), 
            img_obj_enhance.squeeze().transpose(1,2,0), 
            img_marker_enhance.squeeze(),
            img_depth.squeeze(), 
            flow.squeeze().transpose(1, 2, 0)
        )

    def inferDiff(self, img_float):
        '''
        image_plain: numpy array with type np.float32, range from 0 to 1, size [h, w, 3]
        return: numpy array with type np.float32, range from 0 to 1, size [h, w, 3]
        '''
        img_ref_float = img_float.copy()
        img_ref_binary = np.zeros((*img_float.shape[:2], 1), np.float16)

        img_marker_free, _, _, _, _ = self.inferAllInOne(img_float, img_ref_float, img_ref_binary)
        return img_marker_free