from xensesdk.torchconf.base.block import *
from xensesdk.torchconf import BaseModel

class ADD(nn.Module):
    def __init__(self, sign = 1):
        super().__init__()
        self.sign = sign

    def forward(self, x):
        return x[0] + self.sign * x[1]


class DiffBlock2(nn.Module):
    def __init__(self, c1, c2, stride, gaussian_kernel=7, sigma=5):
        super().__init__()
        assert c1 == 3
        self.blur = GaussianBlurConv(gaussian_kernel, sigma)
        self.cv = Conv(c1*2, c2, 3, stride=stride)

    def forward(self, x):
        return self.cv(torch.concat([x - self.blur(x) + 0.5, x], dim=1))


class Detach(nn.Module):
    def forward(self, x):
        return x.detach()


def cst_grid_sample_bilinear(input, grid, **kwargs):
    """
    Custom implementation of grid_sample using bilinear interpolation.

    Args:
        input (Tensor): Input feature map of shape [batch, channels, height, width].
        grid (Tensor): Grid coordinates of shape [batch, out_height, out_width, 2].
    
    Returns:
        Tensor: Sampled feature map of shape [batch, channels, out_height, out_width].
    """
    batch, channels, height, width = input.shape
    _, out_height, out_width, _ = grid.shape

    # Normalize grid coordinates to pixel coordinates
    x = grid[..., 0]  # [batch, out_height, out_width]  
    y = grid[..., 1]  # [batch, out_height, out_width]
    
    x = ((x + 1) * (width - 1) / 2)
    y = ((y + 1) * (height - 1) / 2)

    # x 在 [0, width-1] 之间有效
    padding_mask = ((x < 0).float() + (x > width-1).float() + (y < 0).float() + (y > height-1).float())>0  # [batch, out_height, out_width]
    x0 = torch.floor(x).long().clamp(0, width - 2)  # Left x-coordinate
    y0 = torch.floor(y).long().clamp(0, height - 2)  # Top y-coordinate

    # Compute the distances between the grid points and the surrounding points
    dx0 = (x - x0).view(batch, 1, -1).expand(batch, channels, -1)  # Horizontal distance from x0
    dy0 = (y - y0).view(batch, 1, -1).expand(batch, channels, -1)  # Vertical distance from y0

    # Reshape input for indexing
    input = input.view(batch, channels, height * width)

    # Compute indices for the four surrounding points
    indices00 = y0 * width + x0  # [batch, out_height, out_width]
    indices01 = indices00 + 1  # Top-right corner index
    indices10 = indices00 + width  # Bottom-left corner index
    indices11 = indices10 + 1  # Bottom-right corner index

    # Gather the values for the four corners
    values00 = input.gather(2, indices00.view(batch, 1, -1).expand(batch, channels, -1))
    values01 = input.gather(2, indices01.view(batch, 1, -1).expand(batch, channels, -1))
    values10 = input.gather(2, indices10.view(batch, 1, -1).expand(batch, channels, -1))
    values11 = input.gather(2, indices11.view(batch, 1, -1).expand(batch, channels, -1))

    # Compute the bilinear interpolation
    top = values00 * (1 - dx0) + values01 * dx0  # Interpolation along x-axis
    bottom = values10 * (1 - dx0) + values11 * dx0  # Interpolation along x-axis

    output = top * (1 - dy0) + bottom * dy0  # Final interpolation along y-axis

    # Reshape output back to the desired shape
    output = output.view(batch, channels, out_height, out_width)
    output[padding_mask.view(batch, 1, out_height, out_width).expand_as(output)] = 0
    
    return output


class MarkerHead2(nn.Module):
    
    def __init__(self, c1, c_mid, n_iter=1, grid_xy_range=(640, 480), cols=14, rows=10, c_mid2=128):
        super().__init__()
        self.grid_sample = F.grid_sample
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

        self.conf_conv1 = Conv(c1, c1//2, 3)
        self.conf_conv2 = Conv(2, c_mid2, 5)
        self.conf_conv3 = nn.Sequential(
            Conv(c1//2 + c_mid2, c1//2, 3, 1,),
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
        feat = self.grid_sample(feat, grid1, align_corners=True)

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
        feat2 = self.grid_sample(feat.detach(), grid2.permute(0, 2, 3, 1), align_corners=True)
        conf = torch.concat([self.conf_conv1(feat2), self.conf_conv2(grid_move)], dim=1)
        conf = self.conf_conv3(conf)
        return torch.concat([grid, conf], dim=1)

    def get_default_grid(self):
        w, h = self.grid_xy
        X = torch.linspace(0, self.marker_interval * (self.cols-1), self.cols)
        Y = torch.linspace(0, self.marker_interval * (self.rows-1), self.rows)
        x0 = (w - X[-1]) / 2
        y0 = (h - Y[-1]) / 2
        grid = torch.stack(torch.meshgrid(X+x0, Y+y0, indexing="xy"), dim=0).unsqueeze(0)
        return grid  # shape: (1, 2, row, col)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # HACK 避免 pth 中的 grid_init 与 config 中的 grid_init 不一致, 默认使用 pth 中的 grid_init
        key = prefix + "grid_init"
        checkpoint_grid_init = state_dict[key]
        checkpoint_grid_init = checkpoint_grid_init.to(device=self.grid_init.device, dtype=self.grid_init.dtype)
        self.grid_init = nn.Parameter(checkpoint_grid_init, requires_grad=False)
        del state_dict[key]

        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                              missing_keys, unexpected_keys, error_msgs)        
    
class SrcVAE(nn.Module):
    def __init__(self, bottleneck_size, channel_size=1):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.channel_size    = channel_size
    
        self.encoder_layer = torch.nn.Sequential(
            nn.Conv2d(channel_size, 16, 3, stride=2, padding=1),  # [batch, 16,16,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [batch, 32, 8,8]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3,stride=2, padding=1),  # [batch, 64, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3),  # [batch, 128, 2, 2]
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder_layer = nn.Sequential(
            nn.ReLU(),
            # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
            nn.ConvTranspose2d(128, 64, 3 ),  # [batch, 64, 4,4]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3,stride=2,padding=1,output_padding=1),  # [batch, 32, 8, 8]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # [batch, 16, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, channel_size, 3, stride=2, padding=1, output_padding=1),  # [batch, 1, 32, 32]
            nn.Sigmoid(),
        )
        self.linear_encoder_mu = nn.Linear(128*2*2, self.bottleneck_size)
        self.linear_encoder_logvar = nn.Linear(128*2*2, self.bottleneck_size)
        self.linear_decoder = nn.Linear(self.bottleneck_size ,128*2*2)

    def encoder(self,image):
        
        last_layer = self.encoder_layer(image)
        flat_last = last_layer.view(last_layer.size(0), -1)
        mu = self.linear_encoder_mu(flat_last)
        logvar = self.linear_encoder_logvar(flat_last)
        std = logvar.mul(0.5).exp()

        return mu, std
    
    def decoder(self,code):
        first_layer = self.linear_decoder(code)
        unflat = first_layer.view(first_layer.size(0), 128, 2, 2)
        decoded_image = self.decoder_layer(unflat)
        return decoded_image
    
    def reparametrization(self,mu,std):
        z = torch.randn_like(std).mul(std) + mu
        return z
    
    def forward(self, image):
        #solution
        mu, std = self.encoder(image)
        z = self.reparametrization(mu, std)
        decoded_image = self.decoder(z)
        #end_solution
        return decoded_image, mu, std
    
    def reconstruction_loss(prediction, target):
        recon_loss = torch.mean(torch.sum((prediction - target)**2, (1,2,3))) 
        return recon_loss

    def kl_divergence_loss(mu,std):
        kl_loss = torch.mean(- 0.5 * torch.sum(1. + torch.log(torch.pow(std, 2)) - torch.pow(mu, 2) - torch.pow(std, 2), dim=1)) 
        return kl_loss

class DiffHead(nn.Module):
    def __init__(self, border_size=5):
        super().__init__()
        self.border = border_size
        
    def forward(self, x):
        left_part = x[:, :3, :, :self.border]
        middle_part = x[:, 3:, :, self.border:-self.border]
        right_part = x[:, :3, :, -self.border:]
        return torch.concat([left_part, middle_part, right_part], dim=3)

class Correlation(nn.Module):
    def __init__(self, max_displacement=3, stride=1):
        """
        max_displacement: 最大的位移, 相关范围为[-max_disp, max_disp]
        stride: 位移步长
        """
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.stride = stride
        self.padding = max_displacement

    def forward(self, feat1, feat2):
        """
        feat1, feat2: (B, C, H, W) 的特征图
        返回: (B, num_shifts, H, W)
        """
        B, C, H, W = feat1.shape
        max_disp = self.max_displacement
        stride = self.stride

        if self.padding > 0:
            feat2 = F.pad(feat2, [self.padding]*4, mode='constant', value=0)

        corr_tensors = []
        for dy in range(-max_disp, max_disp + 1, stride):
            for dx in range(-max_disp, max_disp + 1, stride):
                shifted = feat2[:, :, self.padding+dy:self.padding+dy+H, self.padding+dx:self.padding+dx+W]  # (B, C, H, W)

                corr = torch.sum(feat1 * shifted, dim=1, keepdim=True)  # (B, 1, H, W)
                corr_tensors.append(corr)

        out = torch.cat(corr_tensors, dim=1)  # (B, num_shifts, H, W)
        return out


class CorelationInput(nn.Module):
    def __init__(self, max_displacement=3, stride=1):
        super().__init__()
        self.max_displacement = max_displacement
        self.stride = stride

        self.net1 = nn.Sequential(
            RepConv( 1,  16, 5, 2, 1, branches=2),
            RepConv(16,  16, 3, 1, 1, branches=2),
            RepConv(16,  32, 3, 2, 1, branches=2),
            RepConv(32,  32, 3, 1, 1, branches=2),  # (B,32,60,36)
        )

        self.correlation = Correlation(max_displacement, stride)

    def forward(self, x):
        feat2 = self.net1(x[0])
        feat1 = self.net1(x[1])
        ret = self.correlation(feat1, feat2)  # (B,49,60,36)
        return torch.cat([feat1, ret], dim=1)        



BaseModel.register(MarkerHead2, Detach, DiffBlock2, ADD, DiffHead, Correlation, CorelationInput)
