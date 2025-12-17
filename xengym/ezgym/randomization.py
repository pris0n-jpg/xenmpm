import numpy as np
from typing import Callable, Dict, List


class NoiseModel:

    def sample(self):
        """
        采样噪声
        """
        raise NotImplementedError


class GaussianNoiseModel(NoiseModel):

    def __init__(self, shape=(), mean=0, std=1, low=-np.inf, high=np.inf):
        """
        高斯噪声模型

        Parameters:
        - shape : tuple , optional, default: (), 默认数据形状, () 表示标量
        - mean : float or np.ndarray, optional, default: 0, float 或者 shape 形状的 np.ndarray, 均值
        - std : float or np.ndarray, optional, default: 1, float 或者 shape 形状的 np.ndarray, 标准差
        - low : float or np.ndarray, optional, default: -np.inf, float 或者 shape 形状的 np.ndarray, 最小值
        - high : float or np.ndarray, optional, default: np.inf, float 或者 shape 形状的 np.ndarray, 最大值
        """
        self._mean = mean
        self._std = std
        self._shape = shape
        self._low = low
        self._high = high

    def sample(self):
        return np.clip(np.random.normal(self._mean, self._std, self._shape), self._low, self._high)


class UniformNoiseModel(NoiseModel):

    def __init__(self, shape=(), low=0, high=1):
        """
        均匀分布噪声模型

        Parameters:
        - shape : tuple , optional, default: (), 默认数据形状, () 表示标量
        - low : float or np.ndarray, optional, default: 0, float 或者 shape 形状的 np.ndarray, 最小值
        - high : float or np.ndarray, optional, default: 1, float 或者 shape 形状的 np.ndarray, 最大值
        """
        self._low = low
        self._high = high
        self._shape = shape

    def sample(self):
        return np.random.uniform(self._low, self._high, self._shape)


class BoolNoiseModel(NoiseModel):

    def __init__(self, p=0.5):
        """
        二值噪声模型

        Parameters:
        - p : float, optional, default: 0.5, 二值分布概率
        """
        self._p = p

    def sample(self) -> bool:
        return np.random.rand() < self._p


class NoiseGroup:

    def __init__(self):
        self._noise_models: Dict[str, NoiseModel] = dict()
        self._setters: Dict[str, Callable] = dict()
        self._label_count: Dict[str, int] = dict()

    def add(self, label: str, noise_model: NoiseModel, setter: Callable):
        """
        添加噪声模型

        Parameters:
        - label : str, 噪声标签
        - noise_model : NoiseModel, _description_
        - setter : Callable, 设置函数
        """
        if self._label_count.get(label, 0) > 0:
            self._label_count[label] += 1
            label += f"_{self._label_count[label] - 1}"
        else:
            self._label_count[label] = 1

        self._noise_models[label] = noise_model
        self._setters[label] = setter

    def sample(self):
        """
        采样噪声
        """
        for key, noise_model in self._noise_models.items():
            self._setters[key](noise_model.sample())