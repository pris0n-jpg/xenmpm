"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, DWConv, RepConv, RepConv2, autopad



class BottleneckC2(nn.Module):
    """Standard bottleneck."""
    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        expansion=0.5,
        kernel_size=(3, 3),
        activation="leakyrelu",
        groups=1,
        padding_mode="zeros"
    ):
        super().__init__()
        c_ = int(c2 * expansion)  # hidden channels
        self.cv1 = Conv(c1, c_, kernel_size[0], 1, activation=activation, padding_mode=padding_mode)
        self.cv2 = Conv(c_, c2, kernel_size[1], 1, groups=groups, activation=activation, padding_mode=padding_mode)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


# Stages
class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1,
        c2,
        n_layer=1,
        shortcut=True,
        expansion=0.5,
        kernel_size=(3, 3),  # kernel sizes of BottleneckC2
        activation="leakyrelu",
        groups=1,
        padding_mode="zeros"
    ):
        super().__init__()
        c_ = int(c2 * expansion)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1, 1, activation=activation)
        self.cv_out = Conv(2 * c_, c2, 1, activation=activation, groups=groups)
        self.m = nn.Sequential(*(BottleneckC2(c_, c_, shortcut, 1.0, kernel_size, activation, groups, padding_mode=padding_mode)
                                 for _ in range(n_layer)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv_out(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(
        self,
        c1,
        c2,
        n_layer=1,
        shortcut=False,
        expansion=0.5,
        kernel_size=(3, 3),  # kernel sizes of BottleneckC2
        activation="relu",
        groups=1,
        padding_mode="zeros"
    ):
        super().__init__()
        self.c_ = int(c2 * expansion)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1, activation=activation)
        self.cv2 = Conv((2 + n_layer) * self.c_, c2, 1, activation=activation)
        self.m = nn.ModuleList(BottleneckC2(self.c_, self.c_, shortcut, 1.0, kernel_size, activation, groups, padding_mode=padding_mode)
                               for _ in range(n_layer))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c_, self.c_), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


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

class InvertedRep(nn.Module):
    """
    MobileNetV3-like inverted residual block with Repconv.
    """
    def __init__(self, c1, c2, kernel_size=3, stride=1, c_mid=None, branches=3, activation='leakyrelu', padding_mode="zeros"):
        super().__init__()
        c_mid = c2 if c_mid is None else c_mid
        self.rep_pw1 = RepConv2(c1, c_mid, 1, 1, branches=branches, activation=activation)
        self.rep_dw = RepConv2(c_mid, c_mid, kernel_size, stride, groups=c_mid, branches=branches, activation=None, padding_mode=padding_mode)
        self.rep_pw2 = RepConv2(c_mid, c2, 1, 1, branches=branches, activation=activation)

    def forward(self, x):
        return self.rep_pw2(self.rep_dw(self.rep_pw1(x)))

class DwPwRep(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1, branches=3, activation='leakyrelu', padding_mode="zeros"):
        super().__init__()
        self.rep_dw = RepConv2(c1, c1, kernel_size, stride, groups=c1, branches=branches, activation=None, padding_mode=padding_mode)
        self.rep_pw = RepConv2(c1, c2, 1, 1, groups=groups, branches=branches, activation=activation)

    def forward(self, x):
        return self.rep_pw(self.rep_dw(x))

class SPPF(nn.Module):
    # Fast Spatial Pyramid Pooling
    def __init__(
        self,
        c1,
        c2,
        kernel_size=5,
        n_level=3,
    ):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.n_level = n_level
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n_level+1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y = [x]
        for _ in range(self.n_level):
            y.append(self.m(x))
        return self.cv2(torch.cat(y, 1))


class MarkerHead(nn.Module):
    def __init__(self, c1, c_mid, n_iter=1, grid_xy_range=(640, 480), cols=14, rows=10):
        super().__init__()
        self.grid_xy = grid_xy_range
        self.cols = cols
        self.rows = rows
        self.marker_interval = grid_xy_range[0] // cols
        self.scale = self.marker_interval // 5
        self.n_iter = n_iter

        self.grid_convs = nn.Sequential(
            RepConv(   c1,       c_mid, 3, branches=2),
            RepConv(c_mid,  c_mid // 2, 3, branches=2),
            Conv(c_mid // 2, 2, 3, 1, use_batchnorm=False, activation=None)
        )

        self.conf_convs = nn.Sequential(
            RepConv( c1+2,       c1, 3, branches=2),
            Conv(c1, c1//2, 3, 1,),
            Conv(c1//2, 1, 3, 1, use_batchnorm=False, activation='sigmoid')
        )

        grid_init = self.get_default_grid().float()
        self.grid_init = nn.Parameter(grid_init, requires_grad=False)

    def sample_once(self, feat, grid):
        grid1 = grid.permute(0, 2, 3, 1)  # to (N, H_out, W_out, 2)
        #
        grid1 = grid1 / torch.tensor([self.grid_xy[0]//2, self.grid_xy[1]//2], device=feat.device) -\
            torch.tensor([1, 1], device=feat.device)
        grid1 = grid1.to(dtype=feat.dtype)
        feat = F.grid_sample(feat, grid1, align_corners=True)

        grid_out = self.grid_convs(feat) * self.scale + grid

        return grid_out

    def forward(self, x):
        feat, grid = x
        for _ in range(self.n_iter):
            grid = self.sample_once(feat, grid)  # grid: (N, 2, H_out, W_out)

        # conf
        grid2 = grid.detach()
        grid_move = grid2 - self.grid_init

        grid2 = grid2 / torch.tensor([[[self.grid_xy[0]//2]], [[self.grid_xy[1]//2]]], device=feat.device) - \
            torch.tensor([[[1]],[[1]]], device=feat.device)
        grid2 = grid2.to(dtype=feat.dtype)
        feat2 = F.grid_sample(feat.detach(), grid2.permute(0, 2, 3, 1), align_corners=True)
        conf = self.conf_convs(torch.concat([feat2, grid_move], dim=1))

        return torch.concat([grid, conf], dim=1)

    def get_default_grid(self):
        w, h = self.grid_xy
        X = torch.linspace(0, self.marker_interval * (self.cols-1), self.cols)
        Y = torch.linspace(0, self.marker_interval * (self.rows-1), self.rows)
        x0 = (w - X[-1]) / 2
        y0 = (h - Y[-1]) / 2
        grid = torch.stack(torch.meshgrid(X+x0, Y+y0, indexing="xy"), dim=0).unsqueeze(0)
        return grid  # shape: (1, 2, row, col)


class GaussianBlurConv(nn.Module):
    def __init__(self, kernel_size, sigma):
        super().__init__()
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel.size(2), groups=3, bias=False,
                              padding=kernel.size(2) // 2, padding_mode='replicate')
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        # Create a 1D Gaussian kernel
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        x = x.float().to(torch.float32)
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()

        # Create a 2D Gaussian kernel by outer product
        gauss_kernel = torch.outer(gauss, gauss)
        gauss_kernel = gauss_kernel.unsqueeze(0).unsqueeze(0)

        # Expand to 2D convolution kernel with shape (out_channels, in_channels, kernel_size, kernel_size)
        gauss_kernel = gauss_kernel.expand(3, 1, kernel_size, kernel_size)
        return gauss_kernel


class DiffBlock(nn.Module):
    def __init__(self, c1, c2, stride, gaussian_kernel=31, sigma=15):
        super().__init__()
        assert c1 == 3
        self.blur = GaussianBlurConv(gaussian_kernel, sigma)
        self.cv = Conv(c1, c2, 3, stride=stride)
        self.cv2 = DwPwConv(c2, c2, 3)

    def forward(self, x):
        return self.cv2(self.cv(x - self.blur(x) + 0.5))
