from typing import Tuple, List

import torch
import pandas as pd
from numpy import int64 as np_int64

from ark.setting import *
from ark.utils import use_device


CURRENT_FOLDER = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def _get_text_label_from_df(df, text_col='text', label_col='label', device=None) -> Tuple[List[str], torch.Tensor]:
    """
    从dataframe中获取text和label
    """
    device = use_device(device)
    texts, labels = df[text_col].tolist(), torch.from_numpy(df[label_col].to_numpy(dtype=np_int64))

    return texts, labels.to(device)


def _load_dir(dir_path,
              text_col='text',
              label_col='label',
              sep=',',
              device=None):
    """
    读取一个文件夹的所有csv文件, 返回text和label

    :param dir_path: 文件夹路径

    :param text_col: 文本列名

    :param label_col: 标签列名

    :param sep: csv分隔符

    :param device: 设备
    """
    all_df = []
    for file in os.listdir(dir_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(dir_path, file), encoding='utf-8', sep=sep)
            all_df.append(df)

    return _get_text_label_from_df(pd.concat(all_df), text_col, label_col, device)


def load(dataset_path='train.csv',
         text_col='text',
         label_col='label',
         sep=',',
         device=None) -> Tuple[List[str], torch.Tensor]:
    """
    读取csv文件, 返回text和label两个列表

    :param dataset_path: 数据集路径，如果是一个文件，可以是相对于DATASET_PATH的路径，也可以是绝对路径。如果是一个文件夹，则读取文件夹下所有csv文件。

    :param text_col: 文本列名

    :param label_col: 标签列名

    :param device: 设备, 默认为None, 自动选择cpu/gpu

    :param sep: csv分隔符
    """
    if os.path.isdir(dataset_path):
        return _load_dir(dataset_path, text_col, label_col, sep, device)

    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(DATASET_PATH, dataset_path)

    df = pd.read_csv(dataset_path, encoding='utf-8', sep=sep)
    return _get_text_label_from_df(df, text_col, label_col, device)
