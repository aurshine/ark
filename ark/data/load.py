from typing import Tuple, List

import torch
import pandas as pd
from numpy import int64 as np_int64

from ark.setting import *
from ark.spider.classify import get_lines, write_lines, clear
from ark.device import use_device


CURRENT_FOLDER = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def load(file_path='tie-ba.csv',
         text_col='TEXT',
         label_col='label',
         sep=',',
         device=None) -> Tuple[List[str], torch.Tensor]:
    """
    读取csv文件, 返回text和label两个列表

    :param file_path: 文件路径，可以是相对于DATASET_PATH的路径，也可以是绝对路径

    :param text_col: 文本列名

    :param label_col: 标签列名

    :param device: 设备, 默认为None, 自动选择cpu/gpu

    :param sep: csv分隔符
    """
    device = use_device(device)
    if not os.path.exists(file_path):
        file_path = os.path.join(DATASET_PATH, file_path)

    df = pd.read_csv(file_path, encoding='utf-8', sep=sep)
    df = df.sample(frac=1, ignore_index=True)

    texts, labels = df[text_col].tolist(), torch.from_numpy(df[label_col].to_numpy(dtype=np_int64))

    return texts, labels.to(device)


def update_tie_ba(encoding='utf-8-sig'):
    """
    内部函数, 更新tie-ba数据集

    :param encoding: 编码方式
    """
    def dic(path, label):
        lines = get_lines(path, encoding=encoding)
        return {
            'TEXT': lines,
            'label': [label] * len(lines)
        }

    df1 = pd.DataFrame(dic(BAD_TXT_PATH, 1))
    df2 = pd.DataFrame(dic(NOT_BAD_TXT_PATH, 0))
    comment = pd.read_csv(TIE_BA_CSV_PATH, encoding=encoding)

    if df1.shape[0]:
        comment = pd.concat([df1, comment])

    if df2.shape[0]:
        comment = pd.concat([df2, comment])
    # comment = pd.concat([df1, df2, comment])
    comment = comment.sort_values(by=['label'], ascending=False)
    comment.to_csv(TIE_BA_CSV_PATH, index=False, encoding=encoding)
    clear(BAD_TXT_PATH), clear(NOT_BAD_TXT_PATH)

    update_tie_ba_split(encoding=encoding)
    print('更新贴吧数据集')


def update_tie_ba_split(encoding='utf-8-sig'):
    """
    内部函数

    将tie-ba数据集分为pos和nag两个数据集
    """
    path = os.path.join(CURRENT_FOLDER, 'DATASET')
    df = pd.read_csv(TIE_BA_CSV_PATH, encoding=encoding)

    pos = df[df['label'] == 0]
    neg = df[df['label'] == 1]

    pos.to_csv(os.path.join(path, 'tie-ba-pos.csv'), index=False, encoding=encoding)
    neg.to_csv(os.path.join(path, 'tie-ba-neg.csv'), index=False, encoding=encoding)


def update_vocab():
    """更新词表"""
    df = pd.read_csv(TIE_BA_CSV_PATH)
    vocab = set(get_lines(VOCAB_PATH))

    add_news = []
    for line in df['TEXT']:
        for c in line:
            if not c.isspace() and c not in vocab:
                vocab.add(c)
                add_news.append(c)

    write_lines(add_news, VOCAB_PATH, mode='a')