import os
import json
from typing import List, Dict, Optional, Union

import torch


class Tokenizer:
    cls_token: str = '[CLS]'
    sep_token: str = '[SEP]'
    pad_token: str = '[PAD]'
    unk_token: str = '[UNK]'
    mask_token: str = '[MASK]'

    def __init__(self, tokenizer_path: Optional[str], max_length):
        """
        tokenizer_path: tokenizer的保存文件夹路径, 如果为None, 则创建一个空的tokenizer

        tokenizer_path 下应该包含以下文件:

        - vocab.json: 词表文件, key为词元, value为词元id
        - special_tokens.json: 特殊词元文件, key为特殊词元, value为词元id
        """
        if tokenizer_path is not None:
            with open(os.path.join(tokenizer_path, 'normal_tokens.json'), 'r', encoding='utf-8') as f:
                self._normal_tokens = json.load(f)
            with open(os.path.join(tokenizer_path, 'special_tokens.json'), 'r', encoding='utf-8') as f:
                self._special_tokens = json.load(f)
        else:
            self._normal_tokens = {}
            for token in self.__dict__.keys():
                if token.endswith('_token'):
                    self._special_tokens[token] = len(self._special_tokens)

        self.max_length = max_length
        self.all_tokens = ['' for _ in range(len(self._normal_tokens) + len(self._special_tokens))]
        for token, id_ in self._normal_tokens.items():
            self.all_tokens[id_] = token
        for token, id_ in self._special_tokens.items():
            self.all_tokens[id_] = token

    def __len__(self):
        return len(self.all_tokens)

    def tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        将token序列转换为id序列

        默认是已经分词

        如果传入的tokens是str, 则返回int
        如果传入的tokens是list, 则返回list
        """
        if isinstance(tokens, str):
            if tokens in self._normal_tokens:
                return self._normal_tokens[tokens]
            elif tokens in self._special_tokens:
                return self._special_tokens[tokens]
            else:
                return self.unk_token_id()
        else:
            return [self.tokens_to_ids(token) for token in tokens]

    def ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        将id序列转换为token序列

        默认是已经分词

        如果传入的ids是int, 则返回str

        如果传入的ids是list, 则返回list
        """
        if isinstance(ids, int):
            if ids < 0 or ids >= len(self):
                raise ValueError(f'id {ids} is out of range')
            return self.all_tokens[ids]
        else:
            return [self.ids_to_tokens(id_) for id_ in ids]

    def encode(self, text: Union[str, List[str]], device=None) -> Dict[str, torch.Tensor]:
        """
        将文本转换为id序列

        - 在文本前后分别添加cls_token和sep_token

        - 如果文本长度不足max_length, 则用pad_token填充， 如果长度超过max_length, 则截断

        :param text: 输入文本, 如果为str, 则表示单个文本, 如果为list, 则表示多个token序列

        :param device: 输出的tensor的device

        :return: 转换后的id序列和mask序列
        """
        if len(text) > self.max_length - 2:
            text = text[:self.max_length - 2]
            token_length = self.max_length - 2
        else:
            token_length = len(text)

        if isinstance(text, str):
            text = list(text)

        tokens = [self.cls_token] + text + [self.sep_token] + [self.pad_token] * (self.max_length - token_length - 2)
        masked_ts = torch.zeros(size=(self.max_length,), dtype=torch.bool, device=device)
        masked_ts[token_length + 2:] = True

        ids = self.tokens_to_ids(tokens)
        ids_ts = torch.tensor(ids, dtype=torch.int64, device=device)

        return {'input_ids': ids_ts, 'attention_mask': masked_ts}

    def token_exists(self, token: str) -> bool:
        return token in self._normal_tokens or token in self._special_tokens

    def is_special_token(self, token: str) -> bool:
        return token in self._special_tokens

    def add_tokens(self, tokens: List[str]):
        for token in tokens:
            self._add_token(token, self._normal_tokens)

    def add_special_tokens(self, special_tokens: List[str]):
        for token in special_tokens:
            self._add_token(token, self._special_tokens)

    def _add_token(self, token: str, where: dict):
        if not self.token_exists(token):
            where[token] = len(self)
            self.all_tokens.append(token)

    def save_tokenizer(self, tokenizer_path: str):
        os.makedirs(tokenizer_path, exist_ok=True)
        with open(os.path.join(tokenizer_path, 'normal_tokens.json'), 'w', encoding='utf-8') as f:
            json.dump(self._normal_tokens, f, ensure_ascii=False, indent=4)
        with open(os.path.join(tokenizer_path, 'special_tokens.json'), 'w', encoding='utf-8') as f:
            json.dump(self._special_tokens, f, ensure_ascii=False, indent=4)

    def cls_token_id(self):
        return self._special_tokens[self.cls_token]

    def sep_token_id(self):
        return self._special_tokens[self.sep_token]

    def pad_token_id(self):
        return self._special_tokens[self.pad_token]

    def unk_token_id(self):
        return self._special_tokens[self.unk_token]

    def mask_token_id(self):
        return self._special_tokens[self.mask_token]