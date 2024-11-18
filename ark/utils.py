import os
import time
from datetime import datetime
from typing import Union, Tuple
import shutil

import torch

from ark.setting import TRAIN_RESULT_PATH


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


def all_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float, float, float, float]:
    """
    计算各种指标，返回 accuracy, precision, recall, fpr, f1
    """
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    tn = torch.sum((y_pred == 0) & (y_true == 0)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, fpr, f1


def get_metrics_str(epoch: int, y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
    """
    计算模型在指定 epoch 的指标, 并返回字符串格式的指标信息
    """
    accuracy, precision, recall, fpr, f1 = all_metrics(y_true, y_pred)

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


def cpu_ts(ts: torch.Tensor) -> torch.Tensor:
    """
    将一个tensor转移到cpu上

    :param ts: 一个tensor
    """
    return ts.clone().detach().cpu()


class Timer:
    def __init__(self, name=None):
        self.name = name if name is not None else "Timer"
        self.start_time = None

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{self.name} taken by {func.__name__}: {end_time - self.start_time} seconds")
            return result

        return wrapper

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        print(f"{self.name} taken: {end_time - self.start_time} seconds")


def clear_train_result():
    """
    清理训练结果

    删除前缀有_的目录和文件
    """
    clear_dirs = [os.path.join(TRAIN_RESULT_PATH, d) for d in os.listdir(TRAIN_RESULT_PATH) if d.startswith('_')]
    for clear_dir in clear_dirs:
        if os.path.isdir(clear_dir):
            shutil.rmtree(clear_dir)
        else:
            os.remove(clear_dir)