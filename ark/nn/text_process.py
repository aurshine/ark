from typing import Tuple, List, Union, Optional, Sequence

import numpy as np

from ark.nn.pinyin import translate_into_other_piny


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


def token_random_mask(token_list: Union[str, List[str]],
                      pred_position: Union[int, List[int], np.ndarray],
                      num_pred_position: int,
                      all_tokens: Sequence[str],
                      _mask_token: str = '<mask>'):
    """
    随机mask token, 并返回mask后的token_list, 以及对应的 mask_position

    :param token_list: 文本或token列表

    :param pred_position: 预测位置列表

    :param num_pred_position: 预测位置数量 或 预测位置范围

    :param all_tokens: 所有token列表

    :param _mask_token: mask token, 默认为'<mask>'

    :return: 返回mask后的token_list, mask_position, mask_position对应的原token
    """
    if isinstance(token_list, str):
        token_list = list(token_list)

    mask_position, source_tokens = [], []
    for pos in np.random.choice(pred_position, num_pred_position, replace=False):
        if np.random.rand() < 0.8:  # 80%的概率被mask
            mask_token = _mask_token
        elif np.random.rand() < 0.5:  # 10%的概率随机替换
            mask_token = np.random.choice(all_tokens)
        else:  # 10%的概率保持不变
            mask_token = token_list[pos]

        source_tokens.append(token_list[pos])
        token_list[pos] = mask_token
        mask_position.append(pos)

    return token_list, mask_position, source_tokens