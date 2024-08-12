import os

import torch
from sklearn.model_selection import train_test_split

from ark.data import load
from ark.data.dataloader import get_tensor_loader
from ark.setting import VOCAB_PATH, MODEL_LIB
from ark.nn.text_process import Vocab, fusion_piny_letter, data_augment_
from ark.nn.module import AttentionArk
from ark.nn.valid import k_fold_valid
from ark.nn.accuracy import save_fig

#################################################################################
# 模型参数
HIDDEN_SIZE = 64                                       # 隐藏层大小

NUM_HEADS = 4                                          # 多头注意力头数

NUM_LAYER = 8                                           # 解码器层数

STEPS = 128                                            # 每个文本的步长

DROPOUT = 0.5                                          # 随机失活率
#################################################################################
# 训练参数
K_FOLD = 15                                            # 交叉验证折数

NUM_VALID = 5                                          # 验证次数, -1表示全部验证

BATCH_SIZE = 128                                       # 批量大小

TRAIN_EPOCHS = 200                                     # 最大训练轮数

STOP_MIN_EPOCH = 0                                     # 最小停止轮数

STOP_LOSS_VALUE = 0.1                                  # 最小停止损失值

OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-2}  # 优化器参数(学习率、权重衰减)
#################################################################################


def train():
    """
    训练模型
    """
    # 读入数据
    texts, labels = load.load('tie-ba.csv')

    train_texts, _, train_labels, _ = train_test_split(texts, labels, train_size=0.99)
    train_labels = torch.tensor(train_labels)

    # 构建词典
    vocab = Vocab(VOCAB_PATH)

    # 文本处理层
    text_layer = fusion_piny_letter

    # 数据预处理
    train_x, valid_len = text_layer(train_texts, vocabs=vocab, steps=STEPS, front_pad=True)

    # 构建模型
    model = AttentionArk(vocab,
                         steps=STEPS,
                         hidden_size=HIDDEN_SIZE,
                         in_channel=3,
                         num_heads=NUM_HEADS,
                         num_layer=NUM_LAYER,
                         dropout=DROPOUT,
                         num_class=2)

    # 训练模型 k折交叉验证
    k_loss_list, k_train_acc, k_valid_acc, ark = k_fold_valid(K_FOLD, train_x, train_labels, valid_len, model=model,
                                                              num_class=2,
                                                              num_valid=NUM_VALID,
                                                              batch_size=BATCH_SIZE,
                                                              epochs=TRAIN_EPOCHS,
                                                              stop_loss_value=STOP_LOSS_VALUE,
                                                              stop_min_epoch=STOP_MIN_EPOCH,
                                                              optim_params=OPTIMIZER_PARAMS)

    max_cell = None
    # 平均准确率
    avg_acc = 0
    for i, valid_acc in enumerate(k_valid_acc):
        if max_cell is None or valid_acc[-1].score > max_cell.score:
            max_cell = valid_acc[-1]

        avg_acc += valid_acc[-1].score / len(k_valid_acc)
        valid_acc.plot('epochs', 'accuracy', [f'fold-{i}'], 'valid-k-fold-cross-valid', save=False)
    save_fig('valid.png')

    for i, train_acc in enumerate(k_train_acc):
        train_acc.plot('epochs', 'accuracy', [f'fold-{i}'], 'train-k-fold-cross-valid', save=False)
    save_fig('train.png')
    max_cell.confusion_matrix()

    print('avg acc:', avg_acc)

    for sub_ark, sub_acc in zip(ark, k_valid_acc):
        path = os.path.join(MODEL_LIB,
                            f'ark-{int(sub_acc[-1].score * 100)}-{HIDDEN_SIZE}-{STEPS}-{NUM_HEADS}-{NUM_LAYER}-{2}.net')
        sub_ark.save_state_dict(path)
