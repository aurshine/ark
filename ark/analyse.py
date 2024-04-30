import os
import time
from enum import Enum
from typing import Union, List, Optional, Sequence
from ark.nn.module import AttentionArk
from ark.nn.text_process import Vocab, fusion_piny_letter
from ark.setting import MODEL_LIB, VOCAB_PATH


text_layer = fusion_piny_letter
__ARK__ = None


def reload(accuracy: int,
           vocab,
           hidden_size,
           in_channel,
           num_heads,
           en_num_layer,
           de_num_layer,
           dropout,
           num_class):
    global __ARK__
    __ARK__ = AttentionArk(vocab,
                           hidden_size=hidden_size,
                           in_channel=in_channel,
                           num_heads=num_heads,
                           en_num_layer=en_num_layer,
                           de_num_layer=de_num_layer,
                           dropout=dropout,
                           num_class=num_class)

    __ARK__.load(os.path.join(MODEL_LIB, f'ark-{accuracy}-{hidden_size}-{num_heads}-{en_num_layer}-{de_num_layer}.net'))
    return __ARK__


class ByType(Enum):
    BY_TEXT = 0
    BY_LABEL = 1


def analyse(comments: Union[str, List[str]], classes: Optional[Sequence[str]] = None, by: Optional[ByType] = None) -> list:
    """
    :param comments: 需要分析的文本, 可以是str 或 list[str]

    :param classes: 类别标签, 默认为 ['非恶意', '恶意'] 或 [0, 1]

    :param by: 每个文本对应的返回值型别, 可选 BY_TEXT: 返回文本类型, BY_LABEL: 返回数值类型

    :return: 以列表的形式返回每个输入的分析结果
    """
    if by is None:
        by = ByType.BY_TEXT
    if classes is None:
        classes = ['非恶意', '恶意']

    if isinstance(comments, str):
        comments = [comments]

    if by == ByType.BY_TEXT:
        pass
    elif by == ByType.BY_LABEL:
        classes = [_ for _ in range(len(classes))]
    else:
        raise RuntimeError(f'by value as {by} was not define')

    global __ARK__
    if __ARK__ is None:
        __ARK__ = reload(97, Vocab(VOCAB_PATH), hidden_size=32, in_channel=3, num_heads=4, en_num_layer=4, de_num_layer=8, dropout=0.5, num_class=2)

    x, valid_len = fusion_piny_letter(comments, __ARK__.vocab, 128)
    return __ARK__.analyse(x, classes, valid_len=valid_len)
