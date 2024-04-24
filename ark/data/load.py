from typing import Tuple, List, Union, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from ark.setting import *
from ark.spider.classify import get_lines, write_lines, clear


CURRENT_FOLDER = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def load_cold(file_name='train', num_texts=-1, replace=False) -> Tuple[List[str], List[int]]:
    """ 在COLD文件夹读取数据集

    :param file_name: 可选择 'train' 'test' 'dev' 'tie-ba' 'tie-ba-pos' 'tie-ba-neg', 'cold'

    :param num_texts: 选择数据集的数量

    :param replace: 是否有放回的取样

    :return: 返回 texts, labels
    """
    cold = pd.read_csv(os.path.join(CURRENT_FOLDER, f'COLD/{file_name}.csv'))

    if num_texts < 0:
        num_texts = cold.shape[0]
    cold = cold.iloc[np.random.choice(cold.shape[0], num_texts, replace=replace)]

    texts, labels = cold['TEXT'], cold['label']
    return texts.tolist(), labels.tolist()


def load_train_test_tieba(num_texts: Union[int, list] = -1,
                          replace=False,
                          test_size=None,
                          train_size=None,
                          random_state=None,
                          shuffle=True,
                          drop_test=False
                          ) -> Tuple[List[str], torch.LongTensor, Optional[List[str]], Optional[torch.LongTensor]]:
    """
    从 ark/data/COLD/tie-ba-pos.csv 和 ark/data/COLD/tie-ba-neg.csv 加载 训练集 和 测试集

    :param num_texts: 选择正样本和负样本的数量, 为 list 时应为 [正样本数, 负样本数]

    :param replace: 是否有放回的取样

    :param test_size: 测试集占比

    :param train_size: 训练集占比

    :param random_state: 随机种子

    :param shuffle: 是否打乱，默认 True

    :param drop_test: 是否抛弃测试集, 选择此参数时, 所有数据集将都作为训练集, 此时测试集返回 None

    :return: train_texts, train_labels, test_texts, test_labels
    """
    if isinstance(num_texts, (list, tuple)):
        pos_length, neg_length = num_texts
    else:
        pos_length, neg_length = num_texts, num_texts

    pos_texts, pos_labels = load_cold('tie-ba-pos', pos_length, replace)
    neg_texts, neg_labels = load_cold('tie-ba-neg', neg_length, replace)

    if drop_test:
        train_texts, train_labels = pos_texts + neg_texts, pos_labels + neg_labels
        indices = np.arange(len(train_texts))
        np.random.shuffle(indices)
        return [train_texts[i] for i in indices], torch.LongTensor([train_labels[i] for i in indices]), None, None

    train_texts, test_texts, train_labels, test_labels = train_test_split(pos_texts + neg_texts,
                                                                          pos_labels + neg_labels,
                                                                          test_size=test_size,
                                                                          train_size=train_size,
                                                                          random_state=random_state,
                                                                          shuffle=shuffle)

    return train_texts, torch.LongTensor(train_labels), test_texts, torch.LongTensor(test_labels)


def load_train_test_cold():
    """
    从 ark/data/COLD/train.csv 和 ark/data/COLD/test.csv 加载 训练集 和 测试集
    """
    train_texts, train_labels = load_cold('train')
    test_texts, test_labels = load_cold('test')

    return train_texts, torch.LongTensor(train_labels), test_texts, torch.LongTensor(test_labels)


def copy_csv(from_path: str, to_path: str, encoding='utf-8-sig'):
    """
    复制csv文件

    :param from_path: 源文件路径

    :param to_path: 目标文件路径

    :param encoding: 编码方式
    """
    csv = pd.read_csv(from_path, encoding=encoding)
    csv.to_csv(to_path, index=False, encoding=encoding)


def update_tie_ba(encoding='utf-8-sig'):
    """
    内部函数, 更新tie-ba数据集

    :param encoding: 编码方式
    """
    print('更新贴吧数据集')

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


def update_tie_ba_split(encoding='utf-8-sig'):
    """
    内部函数

    将tie-ba数据集分为pos和nag两个数据集
    """
    path = os.path.join(CURRENT_FOLDER, 'COLD')
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