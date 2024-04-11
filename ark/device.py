
from typing import Union
import torch


def use_device(device: Union[int, str, torch.device, None] = 0):
    """尝试使用一个device， 若无法使用则使用cpu

    1. 当device是int类型， 返回第 device 个 gpu 或 cpu

    2. 当device是str类型，返回torch.device(device) 或 cpu

    3. 当device是torch.device类型, 返回本身 或 cpu
    """
    if device is None:
        device = 0

    try:
        if isinstance(device, torch.device):
            return device
        elif isinstance(device, str):
            return torch.device(device)
        elif isinstance(device, int) and torch.cuda.device_count() >= device + 1:
            return torch.device(f'cuda:{device}')
        else:
            raise RuntimeError
    except RuntimeError:
        return torch.device('cpu')