import random
from typing import List, Union
from functools import lru_cache

from pypinyin import lazy_pinyin, Style, load_phrases_dict

from ..setting import COMMON_CHAR_PATH


EXTEND_PINY = {
    '0': [['ling']],
    '1': [['yi']],
    '2': [['er']],
    '3': [['san']],
    '4': [['si']],
    '5': [['wu']],
    '6': [['liu']],
    '7': [['qi']],
    '8': [['ba']],
    '9': [['jiu']]
}

load_phrases_dict(EXTEND_PINY)

@lru_cache(maxsize=10000)
def translate_piny(inputs: str, style=None) -> Union[List[str], List[List[str]]]:
    """文本翻译拼音

    :param inputs: 由列表存储的tokenize后的文本

    :param style: 拼音格式, 默认不带音标, Style.FIRST_LETTER 只得到首字母

    :return: 将文本转化为拼音返回

    >>> inputs = '你好吗'
    >>> translate_piny(inputs)
    ['ni', 'hao', 'ma']
    """
    if style is None:
        style = Style.NORMAL

    return [lazy_pinyin(s, style=style, errors='default', strict=False)[0] for s in inputs]


def translate_char(c: str, style=None):
    """
    单字符转化拼音
    """
    assert len(c) == 1, f'c = {c} c.len = {len(c)}'

    if style is None:
        style = Style.NORMAL

    return lazy_pinyin(c, style=style, v_to_u=False)[0]


piny_dict = None


def translate_into_other_piny(text: str, p: float = 0):
    """
    将一段文本的部分词元翻译为同拼音的其它词元

    :param text: 传入文本

    :param p: 每个词元被翻译的概率

    :return: 新的文本
    """
    global piny_dict
    if piny_dict is None:
        piny_dict = {}
        with open(COMMON_CHAR_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split()
                piny_dict[line[0]] = line[1:]

    ret = ''
    for s in text:
        u = random.uniform(0, 1)
        if u <= p:
            s_p = translate_char(s)
            s = random.choice(piny_dict.get(s_p, [s]))

        ret += s

    return ret