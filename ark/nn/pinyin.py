from pypinyin import lazy_pinyin, Style

EXTEND_PINY = {
    '0': 'ling',
    '1': 'yi',
    '2': 'er',
    '3': 'san',
    '4': 'si',
    '5': 'wu',
    '6': 'liu',
    '7': 'qi',
    '8': 'ba',
    '9': 'jiu'
}

EXTEND_LETTER = {
    '0': 'l',
    '1': 'y',
    '2': 'e',
    '3': 's',
    '4': 's',
    '5': 'w',
    '6': 'l',
    '7': 'q',
    '8': 'b',
    '9': 'j'
}


def translate_piny(inputs, style=None):
    """文本翻译拼音

    :param inputs: 由列表存储的tokenize后的文本 或 一串字符串
    :param style: 拼音格式, 默认不带音标, Style.FIRST_LETTER 只得到首字母
    :return: 将文本转化为拼音返回

    >>> inputs = [['你好吗'], ['还行']]
    >>> translate_piny(inputs)
    [['ni', 'hao', 'ma'], ['hai', 'xing']]
    """
    if style is None:
        style = Style.NORMAL
    else:
        assert style == Style.NORMAL or style == Style.FIRST_LETTER

    if isinstance(inputs, str):
        if style == Style.NORMAL:
            return [EXTEND_PINY.get(piny, piny) for piny in lazy_pinyin(inputs, style=style, v_to_u=False)]
        else:
            return [EXTEND_LETTER.get(piny, piny) for piny in lazy_pinyin(inputs, style=style, v_to_u=False)]
    else:
        return [translate_piny(_input, style) for _input in inputs]


def translate_char(c: str, style=None):
    """单字符转化拼音"""
    if style is None:
        style = Style.NORMAL
    else:
        assert style == Style.NORMAL or style == Style.FIRST_LETTER
    assert len(c) == 1, f'c = {c} c.len = {len(c)}'

    return lazy_pinyin(c, style=style, v_to_u=False)[0]