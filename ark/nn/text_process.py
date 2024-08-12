import random
from collections import Counter
from typing import Tuple, Union, List, Optional

import torch
from torch import nn, Tensor
import jieba

from ark.spider.classify import get_lines, write_lines
from ark.setting import VOCAB_PATH
from ark.nn.pinyin import translate_piny, translate_into_other_piny


class Tokenize(nn.Module):
    def __init__(self, mode='char', keep_dim=True, **kwargs):
        super(Tokenize, self).__init__(**kwargs)
        modes = ['char', 'word', 'cutWord']
        if mode not in modes:
            raise RuntimeError(f'unknown mode {mode}, select mode of {modes}')

        self.mode, self.keep_dim = mode, keep_dim

    def forward(self, inputs, keep_dim=None) -> list:
        if keep_dim is None:
            keep_dim = self.keep_dim

        if isinstance(inputs, (list, tuple)):
            if keep_dim:
                return [self.forward(line) for line in inputs]
            else:
                ret = []
                for line in inputs:
                    ret += self.forward(line)
                return ret
        else:
            if self.mode == 'char':
                return list(inputs)
            elif self.mode == 'word':
                return inputs.split()
            elif self.mode == 'cutWord':
                return [line for line in jieba.cut(inputs)]


class TruncPadding(nn.Module):
    def __init__(self, max_length, pad, front_pad=True, **kwargs):
        """
        对字符串进行填充截断

        :param max_length: 最大长度, 超过截断不足填充
        :param pad: 填充元
        :param front_pad: 为True表示在前面填充, False表示在后面填充
        :param kwargs:
        """
        super(TruncPadding, self).__init__(**kwargs)
        assert max_length > 0
        self.max_length, self.pad, self.front_pad = max_length, pad, front_pad

    def forward(self, inputs: list):
        """传入一个 list, 对最后一维的list填充或截断"""
        ret = []
        if isinstance(inputs[0], (list, tuple)):
            for line in inputs:
                ret.append(self.forward(line))
        else:
            if len(inputs) >= self.max_length:
                ret = inputs[: self.max_length]
            else:
                if self.front_pad:
                    ret = (self.max_length - len(inputs)) * [self.pad] + inputs
                else:
                    ret = inputs + (self.max_length - len(inputs)) * [self.pad]
        return ret


class Vocab(nn.Module):
    """词表类, 把一个字词映射到数字"""
    def __init__(self, corpus=None, mode='char', min_freq=0, reserve=None, **kwargs):
        super(Vocab, self).__init__(**kwargs)
        if reserve is None:
            reserve = []

        self.mode = mode
        self._tokens = []
        self.tokenize = None
        if isinstance(corpus, list):
            self.tokenize = Tokenize(mode=mode, keep_dim=False)
            corpus = self.tokenize(corpus)

            self._freq = sorted(Counter(corpus).items(), key=lambda x: x[1])
            self._tokens += ['<unk>'] + reserve + [token for token, freq in self._freq if freq > min_freq]

        elif isinstance(corpus, str):
            self.load_vocab(corpus)
        elif corpus is not None:
            raise RuntimeError

        self._ids = {token: i for i, token in enumerate(self._tokens)}

    def forward(self, inputs: Union[list, tuple, str], dtype=torch.int32):
        """token 映射数字"""
        return torch.tensor(self.to_index(inputs), dtype=dtype)

    @property
    def id(self):
        return self._ids

    @property
    def token(self):
        return self._tokens

    @property
    def unk(self):
        """
        :return: 返回未知词元 <unk>
        """
        return '<unk>'

    @property
    def unk_index(self):
        return self.id[self.unk]

    def __len__(self):
        return len(self._tokens)

    def to_index(self, inputs: Union[list, tuple, str]) -> Union[list, int]:
        """
        词元映射到数字
        """
        ret = []
        if isinstance(inputs, (tuple, list)):
            for _input in inputs:
                ret.append(self.to_index(_input))
        else:
            return self.id.get(inputs, self.unk_index)
        return ret

    def load_vocab(self, path):
        """读取对应文件的词表, 每行表示一个词元"""
        self._tokens = [self.unk] + get_lines(path)

    def download(self, path=VOCAB_PATH):
        write_lines(self._tokens, path=path, encoding='utf-8', mode='w')


class TextProcess(nn.Module):
    def __init__(self, vocab: Vocab, mode, steps=128, front_pad=True):
        """
        对文本预处理, 词元化, 填充截断后映射到vocab Index里

        :param vocab: 词表

        :param mode: 词元化的模式, 如果不需要词元化传入 None

        :param steps: 文本保留长度

        :param front_pad: 为True时填充选择前填充
        """
        super(TextProcess, self).__init__()
        self.vocab = vocab
        self.mode = mode
        self.steps = steps
        self.tokenize = Tokenize(mode=mode) if mode is not None else None
        self.padding = TruncPadding(steps, pad=vocab.unk, front_pad=front_pad)

    def forward(self, X: List[str]) -> Tensor:
        """
        词元化并填充X

        :return: 形状为 (len(X), steps)
        """

        if self.tokenize is None:
            X = self.padding(X)
        else:
            X = self.padding(self.tokenize(X))

        return self.vocab(X)


def n_texts_process(texts: List[List[str]],
                    vocabs: Union[List[Vocab], Vocab],
                    steps: int,
                    modes: List[str],
                    front_pad: bool) -> Tuple[Tensor, Tensor]:
    """
    :return: 形状为 (num_text, num_channel, steps), 形状为 (num_text)
    """
    assert len(texts) == len(modes)

    num_channel = len(texts)
    if isinstance(vocabs, Vocab):
        vocabs = [vocabs] * num_channel

    ret, valid_len = [], [len(text) for text in texts[0]]
    for text, vocab, mode in zip(texts, vocabs, modes):
        process = TextProcess(vocab, mode=mode, steps=steps, front_pad=front_pad)
        ret.append(process(text))

    return torch.stack(ret).permute(1, 0, 2), torch.tensor(valid_len, dtype=torch.int32)


def texts_process(texts, vocab: Union[List[Vocab], Vocab], steps=128, front_pad=True) -> Tuple[Tensor, Tensor]:
    """将文本通过词表映射为3D-tensor

    形状为 (num_texts, 1, steps), 其中 1 表示通道数

    :return: 返回texts转化成的3D-tensor, 每个text 的 valid_len
    """
    return n_texts_process([texts], vocab, steps, ['char'], front_pad=front_pad)


def fusion_piny(texts, vocabs: Union[List[Vocab], Vocab], steps=128, front_pad=True) -> Tuple[Tensor, Tensor]:
    """将文本通过词表映射为3D-tensor, 其中融合拼音特征

    形状为 (num_texts, 2, steps), 其中 2 表示字和拼音双通道

    :return: 返回texts转化成的3D-tensor, 每个text 的 valid_len
    """
    piny = translate_piny(texts)
    return n_texts_process([texts, piny], vocabs, steps, modes=['char', None], front_pad=front_pad)


def fusion_piny_letter(texts, vocabs: Union[List[Vocab], Vocab], steps=128, front_pad=True) -> Tuple[Tensor, Tensor]:
    """将文本通过词表映射为3D-tensor, 其中融合拼音特征和拼音首字母特征

    形状为 (num_texts, 3, steps), 其中 3 表示字, 拼音, 拼音首字母三通道

    :return: 返回texts转化成的3D-tensor, 每个text 的 valid_len
    """
    from pypinyin import Style

    piny = translate_piny(texts, Style.NORMAL)
    letter = translate_piny(texts, Style.FIRST_LETTER)

    return n_texts_process([texts, piny, letter], vocabs, steps, modes=['char', None, None], front_pad=front_pad)


def data_augment_(texts: List[str], labels: List = None, choice_p: float = 0.2, mdf_p: float = 0.1) -> Tuple[List[str], Optional[List]]:
    """
    数据增广, 在原列表里操作

    :param texts: 所有文本

    :param labels: 文本对应的标签, 默认为None

    :param choice_p: 每个文本被选择的概率

    :param mdf_p: 每个词元被修改的概率

    :return:  返回增广后的数据
    """
    len_texts = len(texts)
    for i in range(len_texts):
        text, label = texts[i], (labels[i] if labels else None)

        u_choice = random.uniform(0, 1)
        if u_choice < choice_p:
            texts.append(translate_into_other_piny(text, mdf_p))
            if labels is not None:
                labels.append(label)

    return texts, labels


def data_augment(texts: List[str], labels: List = None, choice_p: float = 0.2, mdf_p: float = 0.1) -> Tuple[List[str], Optional[List]]:
    """
    数据增广, 在原列表里操作

    :param texts: 所有文本

    :param labels: 文本对应的标签, 默认为None

    :param choice_p: 每个文本被选择的概率

    :param mdf_p: 每个词元被修改的概率

    :return:  返回增广后的数据
    """
    texts_ = [text for text in texts]
    labels_ = [label for label in labels]
    return data_augment_(texts_, labels_, choice_p, mdf_p)

