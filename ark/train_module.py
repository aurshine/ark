import os

from sklearn.model_selection import train_test_split

from .data.load import load
from .data.dataloader import get_ark_loader, get_ark_pretrain_loader
from .nn.accuracy import Plot
from .nn.pretrain_loss import InitialFinalLoss
from .nn.tokenizer import Tokenizer
from .nn.module import Ark, ArkClassifier, ArkBertPretrain
from .setting import PRETRAIN_TOKENIZER_PATH, DATASET_PATH, PRETRAIN_DATASET_PATH
from .utils import use_device, all_metrics


#################################################################################
# 模型参数
HIDDEN_SIZE = 128                                       # 隐藏层大小

NUM_HEADS = 8                                          # 多头注意力头数

NUM_LAYER = 8                                           # 解码器层数

STEPS = 128                                            # 每个文本的步长

DROPOUT = 0.5                                          # 随机失活率

NUM_CLASS = 2                                          # 分类数
#################################################################################
# 训练参数
BATCH_SIZE = 48                                        # 批量大小

TRAIN_EPOCHS = 200                                      # 最大训练轮数

STOP_MIN_EPOCH = 20                                     # 最小停止轮数

STOP_LOSS_VALUE = 0.1                                  # 最小停止损失值

OPTIMIZER_PARAMS = {'lr': 1e-4, 'weight_decay': 1e-5}  # 优化器参数(学习率、权重衰减)

RANDOM_SEED = 42                                       # 随机种子

TEST_SIZE = 0.1                                        # 训练集验证集划分比例

TOKENIZER = Tokenizer(PRETRAIN_TOKENIZER_PATH, STEPS)  # 预训练tokenizer
#################################################################################


def train(model_path=None, pretrain_model_path=None, device=None):
    """
    训练模型

    :param model_path: 模型导入路径, 默认为None无需导入

    :param pretrain_model_path: 预训练模型导入路径, 默认为None无需导入

                                设置后model_path导入的部分参数将被忽略, 仅导入预训练模型参数

    :param device: 训练设备, 默认为None, 自动选择
    """
    device = use_device(device)

    # loader参数
    loader_kwargs = {
        'tokenizer': TOKENIZER,
        'max_length': STEPS,
        'batch_size': BATCH_SIZE,
        'device': device,
    }

    texts, labels = load(os.path.join(DATASET_PATH, 'all_data'))
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(texts, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # 构造数据加载器
    train_loader = get_ark_loader(train_texts, train_labels, **loader_kwargs)
    valid_loader = get_ark_loader(valid_texts, valid_labels, **loader_kwargs)

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
              num_channels=3,
              dropout=DROPOUT,
              num_class=NUM_CLASS,
              device=device,
              prefix_name='ark')

    if model_path is not None:
        ark.load(model_path)
    if pretrain_model_path is not None:
        ark.load_pretrain(pretrain_model_path)

    loss_list, valid_trues, valid_results = ark.fit(train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    epochs=TRAIN_EPOCHS,
                                                    optim_params=OPTIMIZER_PARAMS,
                                                    stop_min_epoch=STOP_MIN_EPOCH,
                                                    stop_loss_value=STOP_LOSS_VALUE)

    plot = Plot(5)
    for valid_true, valid_result in zip(valid_trues, valid_results.argmax(dim=-1)):
        plot.add(*all_metrics(valid_true, valid_result))

    plot.plot(labels=['accuracy', 'f1-score', 'precision', 'recall', 'fpr'],
              save_path=os.path.join(ark.log_path, 'valid_metrics.png')
              )


def pre_train(model_path=None, device=None):
    """
    预训练模型

    :param model_path: 预训练模型导入路径, 默认为None无需导入

    :param device: 训练设备, 默认为None, 自动选择
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
              num_channels=3,
              dropout=DROPOUT,
              num_class=NUM_CLASS,
              device=device,
              prefix_name='ark_pretrain')

    if model_path is not None:
        ark.load(model_path)

    ark.fit_pretrain(train_loader=loader,
                     epochs=5,
                     optim_params=OPTIMIZER_PARAMS,
                     loss=InitialFinalLoss(tokenizer=TOKENIZER, reduction='mean'),
                     )