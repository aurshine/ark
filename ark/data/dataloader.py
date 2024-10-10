from typing import List, Dict, Union

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import BatchEncoding
from transformers import BertTokenizer

from ark.device import use_device
from ark.nn.pinyin import translate_piny, translate_char, Style
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
                self.texts = f.readlines()
        else:
            self.texts = file_path_or_texts

        self.tokenizer = tokenizer
        self.num_pred_position = num_pred_position
        self.max_length = max_length
        self.device = use_device(device)
        self.all_tokens = [k for k in self.tokenizer.vocab.keys() if len(k) == 1]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_length = len(self.texts[index])
        # 限制预测位置数量, 最多为文本长度的 1/5
        this_num_pred_position = min(self.num_pred_position, text_length // 5)
        # 文本
        masked_tokens, masked_position, real_tokens = token_random_mask(self.texts[index],
                                                                        # 需要填充[CLS][SEP], 所以max_length - 2
                                                                        pred_position=min(text_length,
                                                                                          self.max_length - 2),
                                                                        num_pred_position=this_num_pred_position,
                                                                        all_tokens=self.all_tokens,
                                                                        _mask_token=self.tokenizer.mask_token,
                                                                        )
        # 声母 韵母
        initials, finals = [], []
        for token in masked_tokens:
            if token == self.tokenizer.mask_token:
                token_initial = token_final = self.tokenizer.mask_token
            else:
                token_initial = translate_char(token, Style.INITIALS)
                token_final = translate_char(token, Style.FINALS)
                # 没有声母或韵母用[PAD]代替
                if len(token_initial) == 0:
                    token_initial = self.tokenizer.pad_token
                if len(token_final) == 0:
                    token_final = self.tokenizer.pad_token
            initials.append(token_initial)
            finals.append(token_final)

        # tokenizer后,首位词元被[CLS]填充, 所有位置需要右移一位
        for i in range(len(masked_position)):
            masked_position[i] = masked_position[i] + 1

        # 期望预测的数量 > 实际预测的数量, 则补齐
        # 全部预测[CLS]
        if self.num_pred_position > this_num_pred_position:
            num_extend = self.num_pred_position - this_num_pred_position
            masked_position.extend([0] * num_extend)
            real_tokens.extend([self.tokenizer.cls_token] * num_extend)

        kwargs = {
            'padding': 'max_length',
            'truncation': True,
            'max_length': self.max_length,
            'return_tensors': 'pt',
            'return_token_type_ids': False,
            'return_attention_mask': True,
            'return_length': False,
            'is_split_into_words': True
        }

        item = {
            'source_tokens': self.tokenizer.encode_plus(text=masked_tokens, **kwargs),
            'initial_tokens': self.tokenizer.encode_plus(text=initials, **kwargs),
            'final_tokens': self.tokenizer.encode_plus(text=finals, **kwargs),
            'masked_position': torch.Tensor(masked_position, dtype=torch.int32, device=self.device),
            'label': torch.LongTensor(self.tokenizer.convert_tokens_to_ids(real_tokens), device=self.device)
        }
        return item


def get_ark_loader(file_path_or_df: Union[str, pd.DataFrame],
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

    :param file_path_or_df: csv文件路径或DataFrame对象

    :param tokenizer: 用于分词的Tokenizer对象

    :param max_length: 词元的最大长度

    :param sep: csv文件分隔符

    :param batch_size: 批量大小

    :param shuffle: 是否打乱

    :param drop_last: 对最后不足batch_size大小的批次选择丢弃或保留

    :param device: 加载数据的设备

    :param kwargs: DataLoader的其他参数
    """
    return DataLoader(ArkDataSet(file_path_or_df, sep=sep, tokenizer=tokenizer, max_length=max_length, device=device),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      collate_fn=collate_dict,
                      **kwargs)


def get_ark_pretrain_loader(file_path_or_texts: Union[str, List[str]],
                            tokenizer,
                            num_pred_position: int,
                            max_length: int,
                            batch_size: int = 32,
                            shuffle=True,
                            drop_last=False,
                            device=None):
    """得到一个ArkPretrainDataSet的DataLoader对象

    loader 返回的data的类型为dict

    :param file_path_or_texts: 文本文件路径或文本列表

    :param tokenizer: 用于分词的Tokenizer对象

    :param num_pred_position: 需要mask的位置数量

    :param max_length: 词元的最大长度

    :param batch_size: 批量大小

    :param shuffle: 是否打乱

    :param drop_last: 对最后不足batch_size大小的批次选择丢弃或保留

    :param device: 加载数据的设备
    """
    return DataLoader(ArkPretrainDataSet(file_path_or_texts, tokenizer, num_pred_position, max_length, device),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      collate_fn=collate_dict)