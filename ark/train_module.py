import os

import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ark.data.dataloader import get_ark_loader
from ark.setting import PRETRAIN_TOKENIZER_PATH, LOG_PATH, DATASET_PATH
from ark.device import use_device
from ark.nn.module import Ark, ArkClassifier
from ark.nn.accuracy import Plot

#################################################################################
# 模型参数
HIDDEN_SIZE = 64                                       # 隐藏层大小

NUM_HEADS = 4                                          # 多头注意力头数

NUM_LAYER = 8                                           # 解码器层数

STEPS = 128                                            # 每个文本的步长

DROPOUT = 0.5                                          # 随机失活率

NUM_CLASS = 2                                          # 分类数
#################################################################################
# 训练参数
K_FOLD = 15                                            # 交叉验证折数

NUM_VALID = 5                                          # 验证次数, -1表示全部验证

BATCH_SIZE = 64                                        # 批量大小

TRAIN_EPOCHS = 200                                     # 最大训练轮数

STOP_MIN_EPOCH = 0                                     # 最小停止轮数

STOP_LOSS_VALUE = 0.1                                  # 最小停止损失值

OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-2}  # 优化器参数(学习率、权重衰减)
#################################################################################


def train(device=None):
    """
    由于导入数据使用多进程，请确保train在 if __name__ == '__main__': 代码块中运行

    训练模型
    """
    device = use_device(device)

    # 加载预训练tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_TOKENIZER_PATH)

    # loader参数
    loader_kwargs = {
        'sep': ',',
        'tokenizer': tokenizer,
        'max_length': STEPS,
        'batch_size': BATCH_SIZE,
        'device': device,
    }

    datas = pd.read_csv(os.path.join(DATASET_PATH, 'train.csv'), sep=',', encoding='utf-8')
    indices = list(range(len(datas)))

    # 构造数据加载器
    train_loader = get_ark_loader(datas.iloc[:100], **loader_kwargs)
    valid_loader = get_ark_loader(datas.iloc[100:110], **loader_kwargs)

    ark_classifier = ArkClassifier(hidden_size=HIDDEN_SIZE,
                                   num_classes=NUM_CLASS,
                                   num_heads=NUM_HEADS,
                                   dropout=DROPOUT,
                                   device=device)
    # 构造Ark模型
    ark = Ark(tokenizer=tokenizer,
              output_layer=ark_classifier,
              steps=STEPS,
              hidden_size=HIDDEN_SIZE,
              in_channel=3,
              num_heads=NUM_HEADS,
              num_layer=NUM_LAYER,
              dropout=DROPOUT,
              num_class=NUM_CLASS,
              device=device)

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