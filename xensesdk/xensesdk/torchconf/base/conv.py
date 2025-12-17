import torch
import torch.nn as nn
import math
from typing import List


__all__ = [
    "Conv",
    "DWConv",
    "DwPwConv",
    "RepConv",
    "RepConv2",
    "SEModule",
    "SCSEModule",
    "Clamp",
    "Activation",
    "Attention",
    "Concat",
    "GroupConcat",
    "Slice",
    "Mux",
    "Demux",
    "Focus",
]

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def get_equivalent_kernel_bias(conv: nn.Conv2d, bn: nn.BatchNorm2d, kernel_size=None, groups=1):
    """
    Get equivalent kernel and bias from Conv2d and BatchNorm2d layers.

    Parameters:
    - conv : nn.Conv2d, 可以为 None
    - bn : nn.BatchNorm2d, 必须传入非 None 的 BatchNorm2d
    - kernel_size : int, optional, default: None, 表示 equivalent kernel 使用 conv 的 kernel_size, 否则指定 kernel_size
    - groups : int, optional, default: 1
    """
    if conv is not None:
        kernel = conv.weight
        kernel_size = conv.kernel_size[0] if kernel_size is None else kernel_size
        pad_size = (kernel_size - conv.kernel_size[0]) // 2
        kernel = torch.nn.functional.pad(kernel, [pad_size, pad_size, pad_size, pad_size])  # example: 1x1 pad to 3x3
    else:
        input_dim = bn.num_features // groups
        kernel = torch.zeros((bn.num_features, input_dim, kernel_size, kernel_size), dtype=bn.weight.dtype, device=bn.weight.device)
        for i in range(bn.num_features):
            kernel[i, i % input_dim, kernel_size // 2, kernel_size // 2] = 1

    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    # Fuse Conv2d() and BatchNorm2d() layers
    kernel, bias = get_equivalent_kernel_bias(conv, bn, kernel_size=conv.kernel_size[0])
    conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )
    conv.weight.data = kernel
    conv.bias.data = bias
    return conv

class Conv(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        kernel_size,
        stride=1,
        padding=None,
        use_batchnorm=True,
        activation="leakyrelu",
        groups=1,
        dilation=1,
        padding_mode="zeros",  # options: 'zeros', 'circular', 'reflect', 'replicate', 'constant'
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            bias=not (use_batchnorm),
            groups=groups,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        if use_batchnorm:
            self.bn = nn.BatchNorm2d(c2)
        else:
            self.forward = self.forward_fuse
        self.act = Activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def fuse(self):
        if hasattr(self, "bn"):
            # if hasattr(self, "id"):
            #     print(f"Fusing Conv {self.id}")
            # else:
            #     print(f"Fusing Conv")
            self.conv = fuse_conv_and_bn(self.conv, self.bn)
            self.__delattr__("bn")
            self.forward = self.forward_fuse


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(
        self,
        c1,
        c2,
        kernel_size=1,
        stride=1,
        dilation=1,
        activation="leakyrelu",
        padding_mode="zeros",
    ):
        super().__init__(c1, c2, kernel_size,
                         stride, groups=math.gcd(c1, c2),
                         dilation=dilation, activation=activation, padding_mode=padding_mode)


class DwPwConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, kernel_size=1, stride=1, activation="leakyrelu", padding_mode="zeros"):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv_dw = DWConv(c1, c1, kernel_size, stride=stride, activation=None, padding_mode=padding_mode)
        self.conv_pw = Conv(c1, c2, 1, activation=activation)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv_pw(self.conv_dw(x))


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block: act(bn(conv3x3) + bn(conv1x1)).
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1, branches=1, activation="silu", padding_mode="zeros"):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        self.kernel_size = kernel_size
        self.groups = groups
        self.c1 = c1
        self.c2 = c2
        self.act = Activation(activation)

        self.conv1 = Conv(c1, c2, kernel_size, stride, None, use_batchnorm=True, groups=groups, activation=None, padding_mode=padding_mode)
        self.conv1x1 = Conv(c1, c2, 1, stride, None, use_batchnorm=True, groups=groups, activation=None)

    def forward(self, x):
        """Forward process."""
        return self.act(self.conv1(x) + self.conv1x1(x))

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def fuse(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return

        # if hasattr(self, "id"):
        #     print(f"Fusing RepConv {self.id}")
        # else:
        #     print(f"Fusing RepConv")

        kernel3x3, bias3x3 = get_equivalent_kernel_bias(self.conv1.conv, self.conv1.bn, self.kernel_size)
        kernel1x1, bias1x1 = get_equivalent_kernel_bias(self.conv1x1.conv, self.conv1x1.bn, kernel_size=self.kernel_size, groups=self.groups)
        kernel = kernel3x3 + kernel1x1
        bias = bias3x3 + bias1x1

        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
            padding_mode=self.conv1.conv.padding_mode,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv1x1")
        self.forward = self.forward_fuse


class RepConv2(nn.Module):
    """
    RepConv with n conv3x3 branches: act(n * bn(conv3x3(x)) + bn(conv1x1(x)) + bn(x)).
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1, branches=2, activation="leakyrelu", padding_mode="zeros"):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.kernel_size = kernel_size
        self.groups = groups
        self.branches = branches

        self.act = Activation(activation)

        self.branch_skip = nn.BatchNorm2d(c1) if c1==c2 and stride==1 else None
        self.conv3x3 = nn.ModuleList()
        for i in range(branches):
            self.conv3x3.append(Conv(c1, c2, kernel_size, stride, None, use_batchnorm=True, groups=groups, activation=None, padding_mode=padding_mode))
        self.conv1x1 = Conv(c1, c2, 1, stride, None, use_batchnorm=True, groups=groups, activation=None)

    def forward(self, x):
        """Forward process."""
        out = self.conv1x1(x)
        for conv in self.conv3x3:
            out += conv(x)

        if self.branch_skip is not None:
            out += self.branch_skip(x)

        return self.act(out)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def fuse(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        # if hasattr(self, "id"):
        #     print(f"Fusing RepConv2 {self.id}")
        # else:
        #     print(f"Fusing RepConv2")

        kernel, bias = get_equivalent_kernel_bias(self.conv1x1.conv, self.conv1x1.bn, self.kernel_size, self.groups)
        for conv in self.conv3x3:
            _kernel, _bias = get_equivalent_kernel_bias(conv.conv, conv.bn, self.kernel_size, self.groups)
            kernel += _kernel
            bias += _bias
        if self.branch_skip is not None:
            _kernel, _bias = get_equivalent_kernel_bias(None, self.branch_skip, self.kernel_size, self.groups)
            kernel += _kernel
            bias += _bias

        self.conv = nn.Conv2d(
            in_channels=self.conv3x3[0].conv.in_channels,
            out_channels=self.conv3x3[0].conv.out_channels,
            kernel_size=self.conv3x3[0].conv.kernel_size,
            stride=self.conv3x3[0].conv.stride,
            padding=self.conv3x3[0].conv.padding,
            dilation=self.conv3x3[0].conv.dilation,
            groups=self.conv3x3[0].conv.groups,
            bias=True,
            padding_mode=self.conv3x3[0].conv.padding_mode,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()

        self.__delattr__("conv1x1")
        self.__delattr__("conv3x3")
        self.__delattr__("branch_skip")

        self.forward = self.forward_fuse


# Attentions
class SEModule(nn.Module):
    def __init__(self, c1, reduction_ratio=16):
        super(SEModule, self).__init__()
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction_ratio, c1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.channel_se(x)


class eSEModule(nn.Module):
    def __init__(self, c1):
        super(eSEModule, self).__init__()
        self.channel_ese = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.channel_ese(x)


class SCSEModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c1, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(c1, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


def Activation(name, inplace=True, **params):
    if name is None or name == "identity":
        return nn.Identity(**params)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax2d":
        return nn.Softmax(dim=1, **params)
    elif name == "softmax":
        return nn.Softmax(**params)
    elif name == "logsoftmax":
        return nn.LogSoftmax(**params)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    elif name == "silu":
        return nn.SiLU(inplace=inplace)
    elif name == "argmax":
        return ArgMax(**params)
    elif name == "argmax2d":
        return ArgMax(dim=1, **params)
    elif name == "clamp":
        return Clamp(**params)
    elif callable(name):
        return name(**params)
    else:
        raise ValueError(
            f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
            f"argmax/argmax2d/clamp/None; got {name}"
        )


def Attention(name, **params):
    if name is None:
        return nn.Identity(**params)
    elif name == "scse":
        return SCSEModule(**params)
    elif name == "se":
        return SEModule(**params)
    else:
        raise ValueError("Attention {} is not implemented".format(name))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class GroupConcat(nn.Module):
    """
    首先将每个输入 tensor 沿着 dimension 分组, 然后交替地将每个组的 tensor 沿着 dimension 拼接在一起.
    """
    def __init__(self, dimension=1, groups=1):
        super().__init__()
        self.d = dimension
        self.groups = groups

    def forward(self, x: List[torch.Tensor]):
        group_size = []  # 每个 tensor 的 group size
        for tensor in x:
            assert tensor.size(self.d) % self.groups == 0, f"Size of tensor {tensor.size()} is not divisible by groups {self.groups}"
            group_size.append(tensor.size(self.d) // self.groups)

        group_tensors = []
        for i in range(self.groups):
            for j, tensor in enumerate(x):
                group_tensors.append(tensor.narrow(self.d, i * group_size[j], group_size[j]))

        return torch.cat(group_tensors, self.d)


class Slice(nn.Module):
    def __init__(self, start, stop, step=1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, x):
        return x[:, self.start:self.stop:self.step, ...]

class Mux(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Demux(nn.Module):
    def __init__(self, x_i):
        super().__init__()
        self._x_i = x_i

    def forward(self, x):
        return x[self._x_i]


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(
        self,
        c1,
        c2,
        kernel_size=1,
        stride=1,
        padding=None,
        use_batchnorm=True,
        activation="leakyrelu",
        groups=1,
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, kernel_size,
                         stride, padding, use_batchnorm,
                         activation, groups)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2],
                                    x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1))