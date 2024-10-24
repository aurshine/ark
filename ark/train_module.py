import os
import random

import pandas as pd

from ark.utils import use_device, date_prefix_filename, all_metrics
from ark.setting import PRETRAIN_TOKENIZER_PATH, LOG_PATH, DATASET_PATH, PRETRAIN_DATASET_PATH
from ark.data.dataloader import get_ark_loader, get_ark_pretrain_loader
from ark.nn.accuracy import Plot
from ark.nn.pretrain_loss import InitialFinalLoss
from ark.nn.tokenizer import Tokenizer
from ark.nn.module import Ark, ArkClassifier, ArkBertPretrain

#################################################################################
# 模型参数
HIDDEN_SIZE = 512                                       # 隐藏层大小

NUM_HEADS = 16                                          # 多头注意力头数

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

TOKENIZER = Tokenizer(PRETRAIN_TOKENIZER_PATH, STEPS)  # 预训练tokenizer
#################################################################################


def train(device=None):
    """
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
              num_heads=NUM_HEADS,
              num_layer=NUM_LAYER,
              dropout=DROPOUT,
              num_class=NUM_CLASS,
              device=device,
              prefix_name='ark')

    loss_list, valid_trues, valid_results = ark.fit(log_file='train.LOG',
                                                    train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    epochs=TRAIN_EPOCHS,
                                                    optim_params=OPTIMIZER_PARAMS,
                                                    stop_min_epoch=STOP_MIN_EPOCH,
                                                    stop_loss_value=STOP_LOSS_VALUE)

    plot = Plot(5)
    for valid_true, valid_result in zip(valid_trues, valid_results):
        plot.add(*all_metrics(valid_true, valid_result))

    plot.plot(labels=['accuracy', 'f1-score', 'precision', 'recall', 'fpr'],
              save_path=os.path.join(LOG_PATH, date_prefix_filename('valid_metrics.png'))
              )


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
              num_heads=NUM_HEADS,
              num_layer=NUM_LAYER,
              dropout=DROPOUT,
              num_class=NUM_CLASS,
              device=device,
              prefix_name='ark_pretrain')

    ark.fit(train_loader=loader,
            log_file='pretrain.LOG',
            epochs=20,
            optim_params=OPTIMIZER_PARAMS,
            stop_min_epoch=5,
            stop_loss_value=2,
            loss=InitialFinalLoss(tokenizer=TOKENIZER, reduction='mean'),
            )
