import torch
import torch.nn as nn
import numpy as np
from time import time

__all__ = [
    'time_sync',
    'colorstr',
    'img_to_tensor',
    'tensor_to_numpy',
    'initialize_weights',
    'save_model',
]


def img_to_tensor(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()

    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.shape[2] < img.shape[0]:
        img = img.transpose((2, 0, 1))

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    return torch.as_tensor(img).contiguous().unsqueeze(0)

def tensor_to_numpy(tensor):
    if isinstance(tensor, (tuple, list)):
        return [tensor_to_numpy(t) for t in tensor]

    if tensor.ndim == 4:  # BCHW
        tensor = tensor.squeeze(0)
    return tensor.permute(1, 2, 0).cpu().numpy()

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time()

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def initialize_weights(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True


def save_model(log_dir, model, file_name):
    log_dir.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, str( log_dir / 'checkpoint_{}.pth'.format(file_name)))
