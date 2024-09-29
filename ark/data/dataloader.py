from typing import List, Dict, Union

import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset

from ark.device import use_device
from ark.nn.pinyin import translate_piny


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
    def __init__(self, csv_: Union[str, pd.DataFrame], tokenizer, max_length=128, device=None, **kwargs):
        """
        ArkDataSet 类用于处理文本数据集，包括数据集的读取、预处理、tokenizing等。

        :param csv_: csv文件路径或DataFrame对象

        :param tokenizer: 用于分词的Tokenizer对象

        :param max_length: 最大长度

        :param device: 加载数据的设备

        :param kwargs: read_csv的参数
        """
        if isinstance(csv_, str):
            self.df = pd.read_csv(csv_, **kwargs)
        else:
            self.df = csv_
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = use_device(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        data, text = {}, self.df.iloc[index]['TEXT']
        piny = translate_piny(text)
        letter = [p[0] for p in piny]

        kwargs = {
            'padding': 'max_length',
            'truncation': True,
            'max_length': self.max_length,
            'return_tensors': 'pt',
            'return_token_type_ids': False,
            'return_attention_mask': True,
            'return_length': False,
        }
        data['source_tokens']: Dict[str, torch.Tensor] = self.tokenizer.encode_plus(text=text, **kwargs)

        data['pinyin_tokens']: Dict[str, torch.Tensor] = self.tokenizer.encode_plus(text=piny,
                                                                                    is_split_into_words=True,
                                                                                    **kwargs)

        data['letter_tokens']: Dict[str, torch.Tensor] = self.tokenizer.encode_plus(text=letter,
                                                                                    is_split_into_words=True,
                                                                                    **kwargs)

        data['label']: torch.Tensor = torch.tensor([self.df.iloc[index]['label']], dtype=torch.int64)
        return data


def ark_collate_fn(batch_datas: List[Dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    ArkDataSet的collate_fn函数，用于将batch数据处理成dict形式

    :param batch_datas: 一个batch的数据, 每个元素为

    {source_tokens: {input_ids, attention_mask}, pinyin_tokens: {input_ids, attention_mask}, letter_tokens: {input_ids, attention_mask}, label: tensor}

    :return: {source_tokens: {input_ids, attention_mask}, pinyin_tokens: {input_ids, attention_mask}, letter_tokens: {input_ids, attention_mask}, label: tensor}

    """
    datas = {}
    tokens_name, other_name = [], []
    for key, value in batch_datas[0].items():
        if 'tokens' in key:
            datas[key] = {k: [] for k in value.keys()}
            tokens_name.append(key)
        else:
            datas[key] = []
            other_name.append(key)

    for token_name in tokens_name:
        for token_key in datas[token_name].keys():
            for data in batch_datas:
                datas[token_name][token_key].append(data[token_name][token_key])
            datas[token_name][token_key] = torch.cat(datas[token_name][token_key])
    for other_name in other_name:
        for data in batch_datas:
            datas[other_name].append(data[other_name])
        datas[other_name] = torch.cat(datas[other_name])

    return datas


def get_ark_loader(csv_: Union[str, pd.DataFrame],
                   tokenizer,
                   max_length: int,
                   sep: str = ',',
                   batch_size: int = 32,
                   shuffle=True,
                   drop_last=False,
                   device=None,
                   **kwargs):
    """得到一个ArkDataSet的DataLoader对象

    loader 返回的data的类型为pandas.DataFrame

    :param csv_: csv文件路径或DataFrame对象

    :param tokenizer: 用于分词的Tokenizer对象

    :param max_length: 词元的最大长度

    :param sep: csv文件分隔符

    :param batch_size: 批量大小

    :param shuffle: 是否打乱

    :param num_workers: 多线程加载数据

    :param drop_last: 对最后不足batch_size大小的批次选择丢弃或保留

    :param device: 加载数据的设备

    :param kwargs: DataLoader的其他参数
    """
    return DataLoader(ArkDataSet(csv_, sep=sep, tokenizer=tokenizer, max_length=max_length, device=device),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      collate_fn=ark_collate_fn,
                      **kwargs)