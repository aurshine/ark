from typing import List, Dict, Union

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import BatchEncoding

from ark.device import use_device
from ark.nn.pinyin import translate_piny


def collate_dict(batch_datas: List[Dict[str, Union[dict, torch.Tensor]]]) -> Dict[str, Union[dict, torch.Tensor]]:
    """
    将dict形式的batch数据合并, 并将tensor stack或cat到一起

    如果tensor的第一维是1，则cat，否则stack
    """
    datas = {k: [] for k in batch_datas[0].keys()}

    for k in datas.keys():
        for data in batch_datas:
            datas[k].append(data[k])

        if isinstance(datas[k][0], torch.Tensor):
            if datas[k][0].shape[0] == 1:
                datas[k] = torch.cat(datas[k])
            else:
                datas[k] = torch.stack(datas[k])
        elif isinstance(datas[k][0], (dict, BatchEncoding)):
            datas[k] = collate_dict(datas[k])
        else:
            raise TypeError(f"Unsupported type {type(datas[k][0])} in collate_dict")

    return datas


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

    def __getitem__(self, index) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        """
        返回一个batch的数据

        :return: {
                source_tokens: {input_ids: tensor(1, max_length), attention_mask: tensor(1, max_length),},
                pinyin_tokens: {input_ids: tensor(1, max_length),, attention_mask: tensor(1, max_length),},
                letter_tokens: {input_ids: tensor(1, max_length),, attention_mask: tensor(1, max_length),},
                label: tensor(1)
            }
        """
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
        data['label']: torch.Tensor = torch.tensor([self.df.iloc[index]['label']], dtype=torch.int64,
                                                   device=self.device)
        return data


class ArkPretrainDataSet(Dataset):
    def __init__(self, csv_: Union[str, pd.DataFrame], tokenizer, max_length=128, device=None, **kwargs):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


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

    :param drop_last: 对最后不足batch_size大小的批次选择丢弃或保留

    :param device: 加载数据的设备

    :param kwargs: DataLoader的其他参数
    """
    return DataLoader(ArkDataSet(csv_, sep=sep, tokenizer=tokenizer, max_length=max_length, device=device),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      collate_fn=collate_dict,
                      **kwargs)