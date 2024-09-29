import os
import sys
import datetime
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import init
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ark.device import use_device
from ark.running import Timer
from ark.setting import LOG_PATH, MODEL_LIB


def get_metrics(epoch: int, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    计算模型在指定 epoch 的指标, 并返回字符串格式的指标信息
    """
    return (f'MetricsAtEpoch: {epoch}\t'
            f'Accuracy: {accuracy_score(y_true, y_pred): 4f}\t'
            f'Precision: {precision_score(y_true, y_pred, average="weighted"): 4f}\t'
            f'Recall: {recall_score(y_true, y_pred, average="weighted"): 4f}\t'
            f'F1-score: {f1_score(y_true, y_pred, average="weighted"): 4f}\n')


def log(message: str, file_path: str = None):
    """
    打印日志信息到控制台, 并可选择保存到文件

    :param message: 日志信息

    :param file_path: 文件路径, 默认为 None, 即打印到控制台
    """
    message = f'[{datetime.datetime.now(): %Y-%m-%d %H:%M:%S}]\t{message}'
    print(message)
    if file_path is not None:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(message)


class Trainer(nn.Module):
    def __init__(self, num_class, device=None):
        super(Trainer, self).__init__()
        self.device = use_device(device)
        self.num_class = num_class

    def forward(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def fit(self,
            train_loader,
            log_file: str = None,
            epochs=500,
            stop_loss_value=-1,
            stop_min_epoch=0,
            optimizer=None,
            optim_params: Optional[Dict] = None,
            loss=None,
            valid_loader=None):
        """
        训练函数

        :param train_loader: 训练集导入器

        :param log_file: 日志文件地址, 默认为 None, 即不保存日志文件

        :param epochs: 设置的最大训练轮数

        :param stop_loss_value: 最大停止训练的损失值

        :param stop_min_epoch: 最小停止训练的训练轮数

        :param optimizer: 优化器, 默认为 None 时使用 AdamW

        :param loss: 损失函数, 默认为 None 时使用 CrossEntropyLoss

        :param optim_params: 当 optimizer 为 None 时, 对默认优化器初始化

        :param valid_loader: 验证集

        :return: 每轮训练的loss构成的列表, 验证集的真实标签, 每轮训练验证集的预测结果
        """
        self.train()

        if log_file is not None and not os.path.exists(log_file):
            log_file = os.path.join(LOG_PATH, log_file)
            open(log_file, 'w').close()

        if optimizer is None:
            if optim_params is None:
                optim_params = {'lr': 2e-3, 'weight_decay': 0.1}
            optimizer = torch.optim.AdamW(self.parameters(), **optim_params)

        if loss is None:
            loss = nn.CrossEntropyLoss()

        # 打印训练信息
        log(f'fit on {self.device}\n', log_file)
        log(f'model architecture: {self}\n', log_file)
        log(f'optimizer: {optimizer}\n', log_file)
        log(f'loss_fn: {loss}\n', log_file)
        log(f'num_class: {self.num_class}\n', log_file)
        log(f'epochs: {epochs}\n', log_file)
        log(f'stop_loss_value: {stop_loss_value}\n', log_file)
        log(f'stop_min_epoch: {stop_min_epoch}\n\n', log_file)
        # 记录 每个 epoch 的 loss, 训练集准确率，验证集准确率
        loss_list, valid_results, valid_trues = [], [], []
        for epoch in range(epochs):
            epoch_loss, valid_true, valid_result = self.fit_epoch(epoch=epoch,
                                                                  train_loader=train_loader,
                                                                  optimizer=optimizer,
                                                                  loss_fn=loss,
                                                                  valid_loader=valid_loader)
            loss_list.append(epoch_loss)
            valid_results.append(valid_result)
            valid_trues.append(valid_true)

            is_stop = self._achieve_stop_condition(epoch + 1, stop_min_epoch, epoch_loss, stop_loss_value)
            if (epoch + 1) % 10 == 0 or is_stop:
                log(f'Epoch {epoch + 1}, valid loss: {epoch_loss}', log_file)
                log(get_metrics(epoch + 1, valid_result, valid_true), log_file)
                self.save_state_dict(os.path.join(MODEL_LIB, f'epoch_{epoch + 1}.pth'))
                if is_stop:
                    break

        return loss_list, valid_trues, valid_results

    def _loader_forward(self, loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        传入一个loader, 计算loader的预测结果

        每次yield一个batch的数据, 计算batch的预测结果, 并返回预测结果

        :param loader: 导入器

        :return: 预测结果 和 真实标签
        """
        for data in loader:
            multi_channel_tokens: List[torch.Tensor] = []
            multi_channel_masks:  List[torch.Tensor] = []

            for key, value in data.items():
                if 'tokens' in key:
                    multi_channel_tokens.append(value['input_ids'])
                    multi_channel_masks.append(value['attention_mask'])

            y = data['label']

            y_hat = self.forward(multi_channel_tokens, multi_channel_masks)
            yield y_hat, y

    @staticmethod
    def _achieve_stop_condition(epoch: int, stop_min_epoch: int, loss: float, stop_max_loss: float) -> bool:
        """
        达到停止条件时返回 True, 否则返回 False
        """
        return epoch >= stop_min_epoch and loss <= stop_max_loss

    def fit_epoch(self,
                  epoch: int,
                  train_loader,
                  optimizer,
                  loss_fn,
                  valid_loader=None
                  ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        训练一个 epoch 的操作

        :param epoch: 当前训练轮数

        :param train_loader: 训练集导入器,

        :param optimizer: 优化器

        :param loss_fn: 计算损失的函数

        :param valid_loader: 验证集导入器

        :return: 训练集的平均损失, 验证集的真实标签, 验证集的预测结果
        """
        self.train()
        epoch_loss = 0
        train_result, train_true, valid_result, valid_true = [], [], [], []

        for y_hat, y in self._loader_forward(train_loader):
            batch_loss = loss_fn(y_hat, y)
            epoch_loss += batch_loss.item() / len(train_loader)

            # 梯度计算
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 记录训练集的预测结果
            train_result.append(y_hat.argmax(dim=-1).cpu())
            train_true.append(y.cpu())

        # 记录训练集的预测结果
        train_result = torch.cat(train_result).numpy()
        train_true = torch.cat(train_true).numpy()

        log(f'Epoch {epoch + 1}, train_loss: {epoch_loss:.4f}')
        log(get_metrics(epoch + 1, train_true, train_result))

        # 训练结束验证
        if valid_loader is not None:
            valid_true, valid_result = self.validate(valid_loader)

        return epoch_loss, valid_true, valid_result

    def validate(self, valid_loader) -> Tuple[np.ndarray, np.ndarray]:
        """
        验证函数

        :param valid_loader: 验证集导入器

        :return: true_y, valid_y
        """
        self.eval()
        true_y, valid_y = [], []
        for y_hat, y in self._loader_forward(valid_loader):
            valid_y.append(y_hat.argmax(dim=-1).cpu())
            true_y.append(y.cpu())

        return torch.cat(true_y).numpy(), torch.cat(valid_y).numpy()

    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        进入eval模式, 预测输入数据 x 的标签

        :param x: 形状与 forward 函数的输入相同

        :param args: 其他参数

        :param kwargs: 其他参数

        :returns: 返回长度为batch_size的一维tensor
        """
        self.eval()

        y = self.forward(x, *args, **kwargs)

        if 0 < self.num_class < y.shape[-1]:
            y = y[:, :self.num_class]
        return torch.argmax(y, dim=-1)

    @Timer()
    def analyse(self, x: torch.Tensor, classes: list, **kwargs) -> list:
        """
        :param x: 形状与forward函数的输入相同

        :param classes:  列表组成的分类名, ['非恶意', '恶意']

        :return: 长度为 batch_size 的 list
        """
        return [classes[y] for y in self.predict(x, **kwargs)]

    def save_state_dict(self, path: str, cover=True):
        """
        保存模型

        :param path: 保存文件地址

        :param cover: 是否覆盖源文件地址的文件, 默认覆盖
        """
        if not cover and os.path.exists(path):
            point_idx = path.find('.')
            path_name, path_suffix = path[: point_idx], path[point_idx:]

            idx = 0
            while os.path.exists(path):
                idx += 1
                path = f'{path_name}({idx}){path_suffix}'

        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """读取本地模型"""
        self.load_state_dict(torch.load(path, map_location=use_device()))

    def init_params(self):
        """
        初始化模型参数
        """
        for name, param in self.named_parameters():
            try:
                if isinstance(param, nn.parameter.UninitializedParameter):
                    continue

                if 'weight' in name:
                    if param.dim() == 1:
                        param = param.unsqueeze(0)
                    init.xavier_uniform_(param.data)

                elif 'bias' in name:
                    init.zeros_(param.data)
            except ValueError as e:
                print(name, param)
                print(e)
                sys.exit()