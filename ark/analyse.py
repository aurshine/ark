import os
from enum import Enum
from typing import Union, List, Optional, Sequence

from ark.nn.module import Ark
from ark.nn.text_process import Vocab, fusion_piny_letter
from ark.setting import MODEL_LIB, VOCAB_PATH


text_layer = fusion_piny_letter
__ARK__ = None


def reload(vocab: Vocab, model_path: str):
    global __ARK__
    model_name = os.path.split(model_path)[1]
    model_name = os.path.splitext(model_name)[0]

    accuracy, hidden_size, steps, num_heads, num_layer, num_class = model_name.split('-')
    __ARK__ = Ark(vocab,
                  hidden_size=hidden_size,
                  in_channel=3,
                  num_heads=num_heads,
                  steps=steps,
                  num_layer=num_layer,
                  dropout=0,
                  num_class=num_class)
    __ARK__.eval()
    __ARK__.load(model_path)
    return __ARK__


class ByType(Enum):
    BY_TEXT = 0
    BY_LABEL = 1


def analyse(comments: Union[str, List[str]], classes: Optional[Sequence[str]] = None, by: Optional[ByType] = None, model_path: str = None) -> list:
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
        __ARK__ = reload(Vocab(VOCAB_PATH), model_path)

    x, valid_len = fusion_piny_letter(comments, __ARK__.vocab, 128)
    return __ARK__.analyse(x, classes, valid_len=valid_len)
