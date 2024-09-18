from typing import Union

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset


def get_tensor_loader(*datas, batch_size, shuffle=True, drop_last=False, **kwargs):
    """得到一个简单的DataLoader对象

    :param datas: tensor组成的 list 或 tuple

    :param batch_size: 批量大小

    :param shuffle: 是否打乱

    :param drop_last: 对最后不足batch_size大小的批次选择丢弃或保留
    """
    dset = TensorDataset(*datas)
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)


class ArkDataSet(Dataset):
    def __init__(self, csv_: Union[str, pd.DataFrame], **kwargs):
        if isinstance(csv_, str):
            self.df = pd.read_csv(csv_, **kwargs)
        else:
            self.df = csv_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]


def ark_collate_fn(batch) -> pd.DataFrame:
    """ArkDataSet的collate_fn函数，用于将batch数据处理成tensor形式"""
    return pd.concat(batch, axis=0, ignore_index=True)


def get_ark_loader(csv_: Union[str, pd.DataFrame], sep=',', batch_size=32, shuffle=True, drop_last=False, num_workers=2, **kwargs):
    """得到一个ArkDataSet的DataLoader对象

    loader 返回的data的类型为pandas.DataFrame

    :param csv_: csv文件路径或DataFrame对象

    :param sep: csv文件分隔符

    :param batch_size: 批量大小

    :param shuffle: 是否打乱

    :param num_workers: 多线程加载数据

    :param drop_last: 对最后不足batch_size大小的批次选择丢弃或保留
    """
    return DataLoader(ArkDataSet(csv_, sep=sep),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      collate_fn=ark_collate_fn,
                      num_workers=num_workers,
                      **kwargs)