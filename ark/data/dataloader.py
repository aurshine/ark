from typing import List, Dict, Union

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import BatchEncoding
from transformers import BertTokenizer

from ark.device import use_device
from ark.nn.pinyin import translate_piny, Style
from ark.nn.text_process import token_random_mask


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
    def __init__(self, csv_: Union[str, pd.DataFrame], tokenizer: BertTokenizer, max_length=128, device=None, **kwargs):
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
                initial_tokens: {input_ids: tensor(1, max_length),, attention_mask: tensor(1, max_length),},
                final_tokens: {input_ids: tensor(1, max_length),, attention_mask: tensor(1, max_length),},
                label: tensor(1)
            }
        """
        # 文本
        text = self.df.iloc[index]['TEXT']
        # 声母
        initial = translate_piny(text, Style.INITIALS)
        # 韵母
        final = translate_piny(text, Style.FINALS)

        kwargs = {
            'padding': 'max_length',
            'truncation': True,
            'max_length': self.max_length,
            'return_tensors': 'pt',
            'return_token_type_ids': False,
            'return_attention_mask': True,
            'return_length': False,
        }

        item = {
            'source_tokens': self.tokenizer.encode_plus(text=text, **kwargs),
            'initial_tokens': self.tokenizer.encode_plus(text=initial, is_split_into_words=True, **kwargs),
            'final_tokens': self.tokenizer.encode_plus(text=final, is_split_into_words=True, **kwargs),
            'label': torch.tensor([self.df.iloc[index]['label']], dtype=torch.int64, device=self.device)
        }

        return item


class ArkPretrainDataSet(Dataset):
    def __init__(self,
                 file_path_or_texts: Union[str, List[str]],
                 tokenizer: BertTokenizer,
                 num_pred_position=5,
                 max_length=128,
                 device=None):
        """
        ark 使用bert预训练模型进行训练的dataset类

        :param file_path_or_texts: 文本文件路径或文本列表

        :param tokenizer: 用于分词的Tokenizer对象

        :param num_pred_position: 需要mask的位置数量

        :param max_length: 词元的最大长度

        :param device: 加载数据的设备
        """
        if isinstance(file_path_or_texts, str):
            with open(file_path_or_texts, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f]
        else:
            self.texts = file_path_or_texts

        self.tokenizer = tokenizer
        self.num_pred_position = num_pred_position
        self.max_length = max_length
        self.device = use_device(device)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        this_num_pred_position = min(self.num_pred_position, len(self.texts[index]) // 5)
        # 文本
        token_list, masked_position, masked_token = token_random_mask(self.texts[index],
                                                                      pred_position=len(self.texts[index]),
                                                                      num_pred_position=this_num_pred_position,
                                                                      all_tokens=list(self.tokenizer.vocab.keys()),
                                                                      _mask_token='*',
                                                                      )
        # 文本
        text = ''.join(token_list)
        # 声母
        initial = translate_piny(text, Style.INITIALS)
        # 韵母
        final = translate_piny(text, Style.FINALS)
        #  将 * 替换为 [MASK]
        for i in masked_position:
            initial[i] = final[i] = self.tokenizer.mask_token

        # tokenizer后,首位词元被[BOS]填充
        for i in range(len(masked_position)):
            masked_position[i] = masked_position[i] + 1
        if self.num_pred_position > this_num_pred_position:
            masked_position.extend([0] * (self.num_pred_position - this_num_pred_position))

        kwargs = {
            'padding': 'max_length',
            'truncation': True,
            'max_length': self.max_length,
            'return_tensors': 'pt',
            'return_token_type_ids': False,
            'return_attention_mask': True,
            'return_length': False,
        }

        item = {
            'source_tokens': self.tokenizer.encode_plus(text=text, **kwargs),
            'initial_tokens': self.tokenizer.encode_plus(text=initial, is_split_into_words=True, **kwargs),
            'final_tokens': self.tokenizer.encode_plus(text=final, is_split_into_words=True, **kwargs),
            'masked_position': torch.tensor(masked_position, dtype=torch.int64, device=self.device),
        }

        # 只提出被mask的token的label
        item['label'] = item['source_tokens']['input_ids'][0, masked_position]

        return item


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