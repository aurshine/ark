import os
from ark.data import load
from ark.setting import VOCAB_PATH, MODEL_LIB
from ark.nn.text_process import Vocab, fusion_piny_letter
from ark.nn.module import AttentionArk
from ark.nn.valid import k_fold_valid
from ark.nn.accuracy import save_fig

# 模型参数
HIDDEN_SIZE = 64  # 隐藏层大小

NUM_HEADS = 4  # 多头注意力头数

EN_LAYER = 4  # 编码器层数

DE_LAYER = 8  # 解码器层数

STEPS = 128  # 每个文本的步长

DROPOUT = 0.5  # 随机失活率


# 训练参数
K_FOLD = 5  # 交叉验证折数

NUM_VALID = -1  # 验证次数, -1表示全部验证

BATCH_SIZE = 128  # 批量大小

TRAIN_EPOCHS = 200  # 最大训练轮数

STOP_MIN_EPOCH = 10  # 最小停止轮数

STOP_LOSS_VALUE = 0.1  # 最小停止损失值

OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-3}  # 优化器参数


def train():
    """
    训练模型
    """
    # 读入数据
    train_texts, train_labels, test_texts, test_labels = load.load_train_test_data(-1, drop_test=True)

    # 构建词典
    vocab = Vocab(VOCAB_PATH)

    text_layer = fusion_piny_letter

    # 数据预处理
    train_x, valid_len = text_layer(train_texts, vocabs=vocab, steps=STEPS, front_pad=True)

    # 构建模型
    ark = AttentionArk(vocab,
                       hidden_size=HIDDEN_SIZE,
                       in_channel=3,
                       num_steps=STEPS,
                       num_heads=NUM_HEADS,
                       en_num_layer=EN_LAYER,
                       de_num_layer=DE_LAYER,
                       dropout=DROPOUT,
                       num_class=2)

    # 训练模型 k折交叉验证
    k_loss_list, k_train_acc, k_valid_acc, ark = k_fold_valid(K_FOLD, train_x, train_labels, valid_len, model=ark,
                                                              num_class=2,
                                                              num_valid=NUM_VALID,
                                                              batch_size=BATCH_SIZE,
                                                              epochs=TRAIN_EPOCHS,
                                                              stop_loss_value=STOP_LOSS_VALUE,
                                                              stop_min_epoch=STOP_MIN_EPOCH,
                                                              optim_params=OPTIMIZER_PARAMS)

    avg_acc = 0
    for i, valid_acc in enumerate(k_valid_acc):
        avg_acc += valid_acc.max().score
        valid_acc.plot('epochs', 'accuracy', [f'fold-{i}'], 'valid-k-fold-cross-valid', save=False)
    save_fig('valid.png')

    for i, train_acc in enumerate(k_train_acc):
        train_acc.plot('epochs', 'accuracy', [f'fold-{i}'], 'train-k-fold-cross-valid', save=False)
    save_fig('train.png')

    print('avg acc:', avg_acc / len(k_valid_acc))
    for sub_ark, sub_acc in zip(ark, k_valid_acc):
        path = os.path.join(MODEL_LIB,
                            f'ark-{sub_acc.max().score: .2f}-{HIDDEN_SIZE}-{NUM_HEADS}-{EN_LAYER}-{DE_LAYER}.net')
        sub_ark.save_state_dict(path)


if __name__ == '__main__':
    train()