import os
import sys
import datetime
import logging
from typing import Dict, Union, Optional, List, Tuple, Generator

import numpy as np
import torch
from torch import nn
from torch.nn import init
from sklearn.metrics import confusion_matrix

from ark.device import use_device
from ark.running import Timer
from ark.setting import LOG_PATH, MODEL_LIB


def get_metrics(epoch: int, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    计算模型在指定 epoch 的指标, 并返回字符串格式的指标信息
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * precision * recall / (precision + recall)

    return (f'Epoch: {epoch}\t'
            f'Accuracy: {accuracy: 4f}\t'
            f'Precision: {precision: 4f}\t'
            f'Recall: {recall: 4f}\t'
            f'FPR: {fpr: 4f}\t'
            f'F1-score: {f1: 4f}\n')


class Trainer(nn.Module):
    def __init__(self, num_class, device=None):
        super(Trainer, self).__init__()
        self.device = use_device(device)
        self.num_class = num_class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

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
        # 初始化日志文件
        self._init_logger(log_file)

        if optimizer is None:
            if optim_params is None:
                optim_params = {'lr': 2e-3, 'weight_decay': 0.1}
            optimizer = torch.optim.AdamW(self.parameters(), **optim_params)

        if loss is None:
            loss = nn.CrossEntropyLoss()

        # 打印训练信息
        self.logger.info(f'num_params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
        self.logger.info(f'fit on {self.device}')
        self.logger.info(f'model architecture: {self}')
        self.logger.info(f'optimizer: {optimizer}')
        self.logger.info(f'loss_fn: {loss}')
        self.logger.info(f'num_class: {self.num_class}')
        self.logger.info(f'epochs: {epochs}')
        self.logger.info(f'stop_loss_value: {stop_loss_value}')
        self.logger.info(f'stop_min_epoch: {stop_min_epoch}\n')
        # 记录 每个 epoch 的 loss, 训练集准确率，验证集准确率
        loss_list, valid_results, valid_trues = [], [], []

        self.train()
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
                self.logger.warning(f'Epoch: {epoch + 1}, valid loss: {epoch_loss}')
                self.logger.warning(get_metrics(epoch + 1, valid_result, valid_true))
                self.save_state_dict(os.path.join(MODEL_LIB, f'epoch_{epoch + 1}.pth'))
                if is_stop:
                    break

        return loss_list, valid_trues, valid_results

    def _loader_forward(self, loader) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
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

        self.logger.info(f'Epoch {epoch + 1}, train_loss: {epoch_loss:.4f}')
        self.logger.info(get_metrics(epoch + 1, train_true, train_result))

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

    def _init_logger(self, log_file: str = None):
        """
        初始化日志文件

        :param log_file: log文件地址, 默认为 None, 即不保存日志文件
        """
        if log_file is not None:
            if not os.path.exists(log_file):
                log_file = os.path.join(LOG_PATH, log_file)
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(stream_handler)

    def _to_device(self, ts: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(ts, list):
            return [t.to(self.device) for t in ts]
        else:
            return ts.to(self.device)