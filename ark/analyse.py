import os
from typing import Union, List
from ark.nn.module import AttentionArk
from ark.nn.text_process import VOCAB_PATH, Vocab, fusion_piny_letter

vocab = Vocab(VOCAB_PATH)

text_layer = fusion_piny_letter

ark = AttentionArk(vocab, hidden_size=64, in_channel=3, num_steps=128, num_heads=4,
                   en_num_layer=2, de_num_layer=4, dropout=0.5, num_class=2)

ark.load(os.path.join(os.path.dirname(__file__), 'data/result-models/ark- 0.84-64-4-2-4.net'))

BY_TEXT, BY_LABEL = 0, 1


def analyse(comments: Union[str, List[str]], by=BY_TEXT) -> list:
    """
    :param comments: 需要分析的文本, 可以是str 或 list[str]

    :param by: 每个文本对应的返回值型别, 可选 BY_TEXT, BY_LABEL

    :return:
    """
    if by not in [BY_TEXT, BY_LABEL]:
        raise RuntimeError(f'by value as {by} was not define')

    if isinstance(comments, str):
        comments = [comments]
    if by == BY_TEXT:
        classes = ['非恶意', '恶意']
    else:
        classes = [0, 1]

    x, valid_len = fusion_piny_letter(comments, vocab, 128)
    return ark.analyse(x, classes, valid_len=valid_len)
