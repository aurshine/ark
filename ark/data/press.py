import os
import time
from typing import Dict, Iterable

import pandas as pd
import keyboard

from .load import load
from ..setting import TRAIN_RESULT_PATH, DATASET_PATH

def get_text(file_path: str):
    df = pd.read_csv(file_path, sep='\t')
    return set(df['text'].tolist())

def maybe_cls_false(items: Iterable[int], cls='pos'):
    """
    对最后一次训练结果做分析，找出可能分错的样本

    :param items: 参考epoch

    :param cls: 类别，pos或neg
    
    :return: 可能分错的样本集合
    """
    last_train = os.path.join(os.listdir(TRAIN_RESULT_PATH)[-1], 'sample_score')
    last_train = os.path.join(TRAIN_RESULT_PATH, last_train)

    false_texts = None
    for i in items:
        file_path = os.path.join(last_train, f'sample_score_epoch{i}', f'false_{cls}_sample.csv')
        text_set = get_text(file_path)
        false_texts = text_set if false_texts is None else false_texts & text_set
    
    return false_texts

def reset_data(texts_labels: Dict[str, int]):
    """
    重置数据集，对可能分错的样本重新标记
    
    :param texts: 文本列表
    
    :param labels: 标签列表
    """
    all_data_path = os.path.join(DATASET_PATH, 'all_data')

    for file in os.listdir(all_data_path):
        file_path = os.path.join(all_data_path, file)
        if file.endswith('.csv'):
            df = pd.read_csv(file_path, sep='\t')
            # text在texts中，则重新标记，否则不变
            df['label'] = df.apply(lambda x: texts_labels.get(x['text'], x['label']), axis=1)
            df.to_csv(file_path, sep='\t', index=False)

def press_wait(maybes: Dict[str, int]):
    i, lens = 1, len(maybes)
    it = iter(maybes.items())
    text, label = next(it)
    print(f'({i}/{lens}) text: {text}\nmaybe label: {"neg" if label == 0 else "pos"}')
    changes = {}

    while True:
        event = keyboard.read_event()
        time.sleep(0.1)

        try:
            if event.event_type == keyboard.KEY_DOWN:
                if event.name == 'space':
                    changes[text] = label
                    print(f'已确认，将"{text}"的标签更改为 {label}\n')
                elif event.name == 'esc':
                    print('已退出分类')
                    return 
                text, label = next(it)
                i += 1
                print(f'({i}/{lens}) text: {text}\nmaybe label: {"neg" if label == 0 else "pos"}')
        except StopIteration:
            print('所有可能分错的样本已标记完成')
            reset_data(changes)
            return 

# 点击重新分类数据
def click_check(items: Iterable[int]):
    maybes = {}
    num_pos, num_neg = 0, 0
    for maybe_pos in maybe_cls_false(items, 'pos'):
        maybes[maybe_pos] = 1
    num_pos = len(maybes)
    for maybe_neg in maybe_cls_false(items, 'neg'):
        maybes[maybe_neg] = 0
    num_neg = len(maybes) - num_pos

    if len(maybes) == 0:
        print('没有发现可能分错的样本')
        return
    
    print(f'可能分错的正样本数：{num_pos}，可能分错的负样本数：{num_neg}')
    print('按esc退出分类，按空格键确认重新标记，按任意键继续\n')

    press_wait(maybes)
