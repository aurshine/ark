import os
import time
from datetime import datetime
from typing import Union

import torch
from sklearn.metrics import confusion_matrix


class Timer:
    def __init__(self, name=None):
        self.name = name if name is not None else "Timer"

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{self.name} taken by {func.__name__}: {end_time - start_time} seconds")
            return result

        return wrapper


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


def get_metrics(epoch: int, y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
    """
    计算模型在指定 epoch 的指标, 并返回字符串格式的指标信息
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * precision * recall / (precision + recall)

    return (f'Epoch: {epoch}\t'
            f'Accuracy: {accuracy: 4f}\t'
            f'Precision: {precision: 4f}\t'
            f'Recall: {recall: 4f}\t'
            f'FPR: {fpr: 4f}\t'
            f'F1-score: {f1: 4f}\n')


def date_prefix_filename(filename: str) -> str:
    """
    为文件名添加日期前缀

    :param filename: 文件名
    """
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime('%Y%m%d%H%M')

    # 分离目录和文件名
    dir_path, file_name = os.path.split(filename)

    # 修改文件名，在前面加上当前时间
    new_file_name = current_time + file_name

    # 重新组合成新的路径
    new_file_path = os.path.join(dir_path, new_file_name)
    return new_file_path