import os
import random
from typing import List

import pandas as pd
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ark.data.dataloader import get_ark_loader, get_ark_pretrain_loader
from ark.setting import PRETRAIN_TOKENIZER_PATH, LOG_PATH, DATASET_PATH, PRETRAIN_DATASET_PATH
from ark.device import use_device
from ark.nn.module import Ark, ArkClassifier, ArkBertPretrain
from ark.nn.accuracy import Plot
from ark.nn.pretrain_loss import InitialFinalLoss


#################################################################################
# 模型参数
HIDDEN_SIZE = 256                                       # 隐藏层大小

NUM_HEADS = 8                                          # 多头注意力头数

NUM_LAYER = 8                                           # 解码器层数

STEPS = 128                                            # 每个文本的步长

DROPOUT = 0.5                                          # 随机失活率

NUM_CLASS = 2                                          # 分类数
#################################################################################
# 训练参数
BATCH_SIZE = 128                                        # 批量大小

TRAIN_EPOCHS = 200                                      # 最大训练轮数

STOP_MIN_EPOCH = 20                                     # 最小停止轮数

STOP_LOSS_VALUE = 0.1                                  # 最小停止损失值

OPTIMIZER_PARAMS = {'lr': 1e-4, 'weight_decay': 1e-2}  # 优化器参数(学习率、权重衰减)

TOKENIZER = BertTokenizer.from_pretrained(PRETRAIN_TOKENIZER_PATH)  # 预训练tokenizer
#################################################################################


def train(device=None):
    """
    由于导入数据使用多进程，请确保train在 if __name__ == '__main__': 代码块中运行

    训练模型
    """
    device = use_device(device)

    # loader参数
    loader_kwargs = {
        'sep': ',',
        'tokenizer': TOKENIZER,
        'max_length': STEPS,
        'batch_size': BATCH_SIZE,
        'device': device,
    }

    datas = pd.read_csv(os.path.join(DATASET_PATH, 'train.csv'), sep=',', encoding='utf-8')
    indices, num_datas = random.sample(range(len(datas)), len(datas)), len(datas)

    # 构造数据加载器
    train_loader = get_ark_loader(datas.iloc[indices[:num_datas * 9 // 10]], **loader_kwargs)
    valid_loader = get_ark_loader(datas.iloc[indices[num_datas * 9 // 10:]], **loader_kwargs)

    ark_classifier = ArkClassifier(hidden_size=HIDDEN_SIZE,
                                   num_classes=NUM_CLASS,
                                   num_heads=NUM_HEADS,
                                   dropout=DROPOUT,
                                   device=device)
    # 构造Ark模型
    ark = Ark(tokenizer=TOKENIZER,
              output_layer=ark_classifier,
              steps=STEPS,
              hidden_size=HIDDEN_SIZE,
              in_channel=3,
              num_heads=NUM_HEADS,
              num_layer=NUM_LAYER,
              dropout=DROPOUT,
              num_class=NUM_CLASS,
              device=device,
              prefix_name='ark')

    loss_list, valid_trues, valid_results = ark.fit(log_file='train.log',
                                                    train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    epochs=TRAIN_EPOCHS,
                                                    optim_params=OPTIMIZER_PARAMS,
                                                    stop_min_epoch=STOP_MIN_EPOCH,
                                                    stop_loss_value=STOP_LOSS_VALUE)

    plot = Plot(4)
    for valid_true, valid_result in zip(valid_trues, valid_results):
        acc = accuracy_score(valid_true, valid_result)
        f1 = f1_score(valid_true, valid_result)
        precision = precision_score(valid_true, valid_result)
        recall = recall_score(valid_true, valid_result)
        plot.add(acc, f1, precision, recall)

    plot.plot(labels=['accuracy', 'f1-score', 'precision', 'recall'], save_path=os.path.join(LOG_PATH, 'valid.png'))


def pre_train(device=None):
    """
    预训练模型
    """
    device = use_device(device)

    loader = get_ark_pretrain_loader(PRETRAIN_DATASET_PATH,
                                     tokenizer=TOKENIZER,
                                     num_pred_position=5,
                                     max_length=STEPS,
                                     batch_size=BATCH_SIZE,
                                     device=device)

    ark = Ark(tokenizer=TOKENIZER,
              output_layer=ArkBertPretrain(HIDDEN_SIZE, num_class=len(TOKENIZER), device=device),
              steps=STEPS,
              hidden_size=HIDDEN_SIZE,
              in_channel=3,
              num_heads=NUM_HEADS,
              num_layer=NUM_LAYER,
              dropout=DROPOUT,
              num_class=NUM_CLASS,
              device=device,
              prefix_name='ark_pretrain')

    ark.fit(train_loader=loader,
            log_file='pretrain.log',
            epochs=TRAIN_EPOCHS,
            optim_params=OPTIMIZER_PARAMS,
            stop_min_epoch=STOP_MIN_EPOCH,
            stop_loss_value=STOP_LOSS_VALUE,
            loss=InitialFinalLoss(tokenizer=TOKENIZER, reduction='mean'),
            )


def add_tokens(tokens: List[str]):
    TOKENIZER.add_tokens(tokens)
    TOKENIZER.save_pretrained(PRETRAIN_TOKENIZER_PATH)