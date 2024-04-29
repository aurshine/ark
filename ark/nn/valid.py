import sys
from copy import deepcopy
from typing import Tuple, List
from tqdm import tqdm
import torch
from torch import Tensor
from ark.nn.accuracy import Accuracy, AccuracyCell
from ark.nn.bagging import Bagging
from ark.data.dataloader import get_tensor_loader


def get_k_fold(k: int, valid_index: int, *datas: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
    assert len(datas) != 0

    fold_size = len(datas[0]) // k
    train_fold, valid_fold = [[] for _ in enumerate(datas)], []

    for i in range(k):
        fold_datas = [data[i * fold_size: (i + 1) * fold_size] for data in datas]
        if i == valid_index:
            valid_fold = fold_datas
        else:
            for j, data in enumerate(fold_datas):
                train_fold[j].append(data)

    return [torch.cat(data) for data in train_fold], valid_fold


def k_fold_valid(k: int, *datas: Tensor, model, num_class, num_valid=-1, batch_size=64, **kwargs) \
        -> Tuple[List, List[Accuracy], List[Accuracy], Bagging]:
    """
    得到 k 折交叉验证的训练集和测试集

    :param k: k折数

    :param datas: datas[0]需要为输入X, datas[1]需要是label Y

    :param model: 训练模型

    :param num_class: 分类类别数

    :param num_valid: 验证的次数, 为 -1 时验证 k 次

    :param batch_size: 批量大小

    :param kwargs: 模型训练需要的超参数

    :return: k_loss_list, k_train_acc, k_valid_acc, Bagging(models, num_class)
    """
    assert k > 1
    if num_valid < 0:
        num_valid = k

    k_loss_list, k_train_acc, k_valid_acc, models = [], [], [], []
    for i in tqdm(range(num_valid)):
        train_data, valid_data = get_k_fold(k, i, *datas)

        models.append(deepcopy(model))
        train_loader = get_tensor_loader(*train_data, batch_size=batch_size, drop_last=True)
        valid_loader = get_tensor_loader(*valid_data, batch_size=batch_size, drop_last=True)

        loss_list, train_acc, valid_acc = models[-1].fit(train_loader, valid_loader=valid_loader, **kwargs)

        k_loss_list.append(loss_list)
        k_train_acc.append(train_acc)
        k_valid_acc.append(valid_acc)
        print(valid_acc.max())

    return k_loss_list, k_train_acc, k_valid_acc, Bagging(models, num_class)