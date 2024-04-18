import os
from enum import Enum
from typing import Union, List, Optional, Sequence
from ark.nn.module import AttentionArk
from ark.nn.text_process import VOCAB_PATH, Vocab, fusion_piny_letter

vocab = Vocab(VOCAB_PATH)

text_layer = fusion_piny_letter

ark = AttentionArk(vocab, hidden_size=64, in_channel=3, num_steps=128, num_heads=4,
                   en_num_layer=2, de_num_layer=4, dropout=0.5, num_class=2)

ark.load(os.path.join(os.path.dirname(__file__), 'data/result-models/ark- 0.84-64-4-2-4.net'))


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

    x, valid_len = fusion_piny_letter(comments, vocab, 128)
    return ark.analyse(x, classes, valid_len=valid_len)
