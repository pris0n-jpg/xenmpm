import yaml
import torch
from torch import nn
import numpy as np
from pathlib import Path
from typing import Type, Dict, List, Union, Iterator
from copy import deepcopy

from ..general import colorstr, img_to_tensor, time_sync, initialize_weights
from .conv import *
from .block import *

class BaseLayer():

    _layer_classes: Dict[str, Type['BaseLayer']] = dict()  # layer_type: layer_class

    @staticmethod
    def init(base_model: 'BaseModel', sub_module: 'SubModule', id: int, input_id, layer_type, args) -> 'BaseLayer':
        """
        Initialize the layer.

        Parameters:
        - base_model : BaseModel, the base model
        - sub_module : SubModule, the submodule
        - id : int32, unique id of the layer
        - input_id : Union[int32, List(int32)], id of the input layer(s),
            valid values are -1, positive integers and negative integers
            indicating the relative position within the submodule
        - layer_type : str, type of the layer
        - args : list, arguments of the layer
        """
        # seperate args to args and kwargs
        args_, kwargs = [], {}
        for arg in args:  # parse args
            if isinstance(arg, dict):
                kwargs.update(arg)
            else:
                args_.append(arg)

        # wrap nn.Module as Layer.Moudle
        if layer_type not in BaseLayer._layer_classes:
            tmp_cls = eval(layer_type)
            BaseLayer._layer_classes[layer_type] =  type("Layer."+tmp_cls.__name__, (tmp_cls, BaseLayer), {})

        # create layer
        try:
            layer: 'BaseLayer' = BaseLayer._layer_classes[layer_type](*args_, **kwargs)
        except Exception as e:
            print(colorstr("red", f"Error occurs at layer: {id} {layer_type}, input layer: {input_id}\n"))
            raise e
        layer.set_id(id, base_model)

        # check input_id, set layers_need_cache
        input_id = input_id if isinstance(input_id, list) else [input_id]  # convert to list
        new_input_id = []
        for _id in input_id:
            if _id < -1 and abs(_id) > len(sub_module):
                raise ValueError(f"Input id {_id} is out of range of submodule {sub_module.name}.")
            elif _id < -1:
                _id = int(list(sub_module.keys())[_id])

            if _id != -1:
                base_model.layers_need_cache.add(_id)

            new_input_id.append(_id)

        layer.set_input_id(new_input_id)

        # print(f'{id:>4}{str(new_input_id):>15}{layer.num_parameters():14.0f}' +
        #       f'   {layer.class_name:<12}{str(args):<30}')
        return layer

    @property
    def class_name(self):
        return self.__class__.__name__.split('.')[-1]

    @property
    def id(self) -> int:
        return self._id

    def set_id(self, id: int, base_model: 'BaseModel') -> None:
        if id in base_model._layers:
            raise ValueError(f"Layer id {id} already exists.")
        base_model._layers[id] = self
        self._id = id

    @property
    def input_id(self) :
        return self._input_id

    def set_input_id(self, input_id):
        self._input_id: List[int] = input_id

    def num_parameters(self):
        """
        Returns the total number of parameters in the model.
        """
        return sum(x.numel() for x in self.parameters())

    def profile(self, x):
        """
        Profile the layer.
        """
        import thop
        if isinstance(x, torch.Tensor):
            x = x.clone()
        elif isinstance(x, list):
            x = [v.clone() if isinstance(v, torch.Tensor) else deepcopy(v) for v in x]

        flop = thop.profile(self, inputs=(x, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs

        t0 = time_sync()
        for _ in range(10):
            self(x)
        dt = (time_sync() - t0) * 100

        numel = self.num_parameters()
        print(f'{self.id:>4}{dt:11.2f} {flop:10.2f} {numel:10.0f}  {self.class_name}')
        return dt, flop, numel

    def feature_visualize(self, x, n=64, save_dir='runs/features'):
        """
        Visualize the features of the layer.

        Parameters:
        - x : Tensor, features to be visualized
        - n : int, optional, default: 64, maximum number of feature maps to plot
        - save_dir : str, optional, default: 'runs/features', directory to save the feature maps
        """
        import matplotlib.pyplot as plt

        save_dir =Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(x, torch.Tensor):  # do not visualize
            return

        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            filename = save_dir / f"id_{self.id}_{self.class_name}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu().detach(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            nrows = n // 8 if n % 8 == 0 else n // 8 + 1
            fig, ax = plt.subplots(nrows, 8, layout="constrained")  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            print(f'Saving {filename}... ({n}/{channels})')
            plt.savefig(filename, dpi=300)
            plt.close()


class SubModule(nn.ModuleDict):

    def __init__(self, base_model: 'BaseModel', name, layer_config_list: list):
        super(SubModule, self).__init__()
        self._name = name

        # parse layer_list
        for layer_args in layer_config_list:
            layer = BaseLayer.init(base_model, self, *layer_args)

            self[str(layer.id)] = layer  # add module

    @property
    def name(self):
        return self._name

    def __getitem__(self, key: Union[str, int]) -> BaseLayer:
        if isinstance(key, int):
            key = str(key)
        return super(SubModule, self).__getitem__(key)


class BaseModel(nn.ModuleDict):

    def __init__(self, yaml: Union[str, Path], str_byte=False):
        super(BaseModel, self).__init__()
        self._layers = {}  # layer_id: layer
        self.layers_need_cache: List[int] = set()  # layers that need cache

        self.forward_paths = None  # forward paths
        if str_byte:
            self.parse_yaml_str(yaml)
        else:
            self.parse_yaml(yaml)
        initialize_weights(self)

    @staticmethod
    def register(*layer_classes: List[Type['BaseLayer']]) -> None:
        """
        注册自定义 nn.Module
        """
        for layer_class in layer_classes:
            layer_name = layer_class.__name__
            BaseLayer._layer_classes[layer_name] = type("Layer."+layer_name, (layer_class, BaseLayer), {})

    def parse_yaml(self, yaml_path):
        """
        Parse a model.yaml dictionary.

        Parameters:
        - yaml_path : Union[str, Path], path to the model.yaml file
        """
        with open(yaml_path, "r", encoding='utf-8') as f:
            self.yaml = yaml.safe_load(f)  # model dict

        # Parse a model.yaml dictionary
        # print(colorstr('green', f"\nModel Config: {yaml_path}"))
        # print(f"{'id':>4}{'input_id':>15}{'params':>14}   {'module':<12}{'arguments':<30}")

        self.forward_paths = self.yaml["forwards"]

        submodule_names = set([smod for fp in self.forward_paths for smod in fp])
        for smod_name in submodule_names:
            # print(colorstr('blue', f"* {smod_name}: "))
            self[smod_name] = SubModule(self, smod_name, self.yaml[smod_name])
        # print()

    def parse_yaml_str(self, yaml_str):
        """
        Parse a model.yaml dictionary.

        Parameters:
        - yaml_path : Union[str, Path], path to the model.yaml file
        """
        
        self.yaml = yaml.safe_load(yaml_str.decode('utf-8'))

        # Parse a model.yaml dictionary
        print(f"{'id':>4}{'input_id':>15}{'params':>14}   {'module':<12}{'arguments':<30}")

        self.forward_paths = self.yaml["forwards"]

        submodule_names = set([smod for fp in self.forward_paths for smod in fp])
        for smod_name in submodule_names:
            print(colorstr('blue', f"* {smod_name}: "))
            self[smod_name] = SubModule(self, smod_name, self.yaml[smod_name])
        print()

    @torch.no_grad()
    def predict(self, x, path_id=0):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
            x: 4D torch tensor with shape (batch_size, channels, height, width)
            Return: D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x, path_id)

        return x

    @torch.no_grad()
    def predict_img(self, img, path_id=0) -> torch.Tensor:
        if self.training:
            self.eval()

        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype
        if isinstance(img, np.ndarray):
            img = img_to_tensor(img).to(device, dtype)
        elif isinstance(img, (list, tuple)):
            img[0] = img_to_tensor(img[0]).to(device, dtype)

        pred = self.forward(img, path_id)
        return pred

    def forward(self, x, path_id=0):
        """
        运行模型

        Parameters:
        - x : Any,
        - path_id : int, optional, default: 0
        """
        output_cache = {}

        path = self.forward_paths[path_id]
        for submodule_name in path:
            submodule = self[submodule_name]
            for layer in submodule.values():
                # make input data
                inputs =  [x if j == -1 else output_cache[j] for j in layer.input_id]
                if len(inputs) == 1:
                    inputs = inputs[0]

                # forward one layer
                try:
                    x = layer(inputs)
                except Exception as e:
                    print(colorstr("red", f"Error occurs at layer: {layer.id} {layer.class_name}, "
                            f"input layer: {layer.input_id}, path_id: {path_id}\n"))
                    raise e

                if layer.id in self.layers_need_cache:
                    output_cache[layer.id] = x
        return x

    def profile(self, input_size=None, input=None, path_id=0, visualize=False):
        """
        分析模型

        Parameters:
        - input_size : list, optional, default: None, 数据形状的列表, 例如 [[1, 2, 10, 20], [1, 3, 10, 20]]
        - input : Tensor or List[Tensor], optional, default: None, 输入数据
        - path_id : int, optional, default: 0
        - visualize : bool, optional, default: False, 是否绘制特征
        """
        param = next(self.parameters())
        x = []

        # make input data
        if input is None:
            if input_size is None:
                input_size = [[1, 3, 480, 640]]

            for sz in input_size:
                x.append(torch.randn(sz, dtype=param.dtype, device=param.device))
            if len(x) == 1:
                x = x[0]
        else:
            x = input

        # forward and profile
        print(colorstr("green", f"Model Profile (path {path_id}, device {param.device}):"))
        print(f"{'id':>4}{'time(ms)':>11s} {'GFLOPs':>10s} {'params':>10s}  module")
        tmp_model = deepcopy(self)
        output_cache = {}
        dts, flops, params = [], [], []
        layer: BaseLayer = None
        path = self.forward_paths[path_id]
        for submodule_name in path:
            submodule = tmp_model[submodule_name]
            print(colorstr('blue', f"* {submodule_name}: "))
            for layer in submodule.values():
                inputs =  [x if j == -1 else output_cache[j] for j in layer.input_id]
                if len(inputs) == 1:
                    inputs = inputs[0]

                try:
                    dt, flop, numel = layer.profile(inputs)
                    dts.append(dt), flops.append(flop), params.append(numel)
                    x = layer(inputs)  # run
                except Exception as e:
                    print(colorstr("red", f"Error occurs at layer: {layer.id} {layer.class_name}, "
                            f"input layer: {layer.input_id}, path_id: {path_id}\n"))
                    raise e

                if layer.id in self.layers_need_cache:
                    output_cache[layer.id] = x

                if visualize:
                    if not hasattr(self, 'save_dir'):
                        self.save_dir = f'runs/features'
                    layer.feature_visualize(x, 64, save_dir=self.save_dir)
        dts = sum(dts)
        flops = sum(flops)
        params = sum(params)
        print(f"{dts:>15.2f} {flops:>10.2f} {params:>10.0f}  Total \n")
        return dts, flops, params

    def parameters(self, select: Union[str, int, list]=None, layer_id: list=None, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Select can be path_id or submodule_name"""
        if select is None and layer_id is None:
            yield from super().parameters(recurse)
        elif select is None and layer_id is not None:
            for id in layer_id:
                yield from self._layers[id].parameters(recurse)
        elif isinstance(select, list):
            for s in select:
                yield from self.parameters(s, recurse)
        elif isinstance(select, str):  # submodule
            yield from self[select].parameters(recurse)
        elif isinstance(select, int):  # path_id
            path = self.forward_paths[select]
            for submodule_name in path:
                yield from self[submodule_name].parameters(recurse)

    def freeze(self, requireds_grad=False, select: Union[str, int, list]=None, layer_id: list=None):
        # 冻结所有参数
        for param in self.parameters(select, layer_id):
            param.requires_grad = requireds_grad

    def load(self, pth_path, device="cuda"):
        state_dict = torch.load(pth_path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        self.to(device=device)
        print(f'Model loaded from {pth_path}')

    def count_conv_bn_layers(self):
        conv_count = 0
        bn_count = 0

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_count += 1
            elif isinstance(module, nn.BatchNorm2d):
                bn_count += 1
        print(f"Conv2d: {conv_count}, BatchNorm2d: {bn_count}")

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if hasattr(m, 'fuse') and not isinstance(m, BaseModel):
                m.fuse()
        return self