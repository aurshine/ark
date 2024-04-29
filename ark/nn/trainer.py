import os
import sys
import math
from typing import Dict, Optional, List, Tuple, Union
import torch
from torch import nn
from torch.nn import init
from ark.device import use_device
from ark.nn.accuracy import Accuracy, AccuracyCell
from ark.running import Timer


class Trainer(nn.Module):
    def __init__(self, num_class, mini_batch_size=128, device=None):
        super(Trainer, self).__init__()
        self.mini_batch_size = mini_batch_size if mini_batch_size is not None and mini_batch_size > 0 else 128
        self.device = use_device(device)
        self.num_class = num_class

    def forward(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def print_fit_info(self, epoch: Union[int, str], loss: float, train_acc: AccuracyCell, valid_acc: AccuracyCell):
        """
        打印训练信息
        """
        print(f'epoch {epoch}:\n'
              f'exp_loss = {math.exp(loss)}\n'
              f'train_accuracy: {train_acc}\n'
              f'valid_accuracy: {valid_acc}\n')

    def fit(self, train_loader,
            epochs=500,
            stop_loss_value=-1,
            stop_min_epoch=0,
            optimizer=None,
            optim_params: Optional[Dict] = None,
            loss=None,
            max_norm=-1,
            valid_loader=None) -> Tuple[List, Accuracy, Accuracy]:
        """
        训练函数
        :param train_loader: 训练集导入器

        :param epochs: 设置的最大训练轮数

        :param stop_loss_value: 最大停止训练的损失值

        :param stop_min_epoch: 最小停止训练的训练轮数

        :param optimizer: 优化器, 默认为 None 时使用 AdamW

        :param loss: 损失函数, 默认为 None 时使用 CrossEntropyLoss

        :param max_norm: 梯度剪裁, 若传入值不大于 0 则不裁剪

        :param optim_params: 当 optimizer 为 None 时, 对默认优化器初始化

        :param valid_loader: 验证集

        :return: 每轮训练的loss构成的列表, 每轮训练训练集的准确率, 每轮训练验证集的准确率
        """
        self.train()
        print(f'fit on {self.device}')

        if optimizer is None:
            if optim_params is None:
                optim_params = {'lr': 2e-3, 'weight_decay': 0.1}
            optimizer = torch.optim.AdamW(self.parameters(), **optim_params)

        if loss is None:
            loss = nn.CrossEntropyLoss()

        # 记录 每个 epoch 的 loss, 训练集准确率，验证集准确率
        loss_list, train_acc, valid_acc = [], Accuracy(self.num_class), Accuracy(self.num_class)
        for epoch in range(epochs):
            epoch_loss, train_cell, valid_cell = self.fit_epoch(train_loader, optimizer, loss, max_norm, valid_loader)
            train_acc.add_cell(train_cell)
            valid_acc.add_cell(valid_cell)

            if epoch >= stop_min_epoch and epoch_loss < stop_loss_value:
                self.print_fit_info(epoch, epoch_loss, train_cell, valid_cell)
                break

            if (epoch + 1) % 10 == 0:
                self.print_fit_info(epoch, epoch_loss, train_cell, valid_cell)

        return loss_list, train_acc, valid_acc

    def fit_epoch(self, loader, optimizer, loss, max_norm=0, valid_loader: Union[List, Tuple, None] = None) -> Tuple[float, AccuracyCell, AccuracyCell]:
        """
        训练一个 epoch 的操作

        :param loader: 训练集导入器 (x, y, *args)

        :param optimizer: 优化器

        :param loss: 计算损失的函数

        :param max_norm: 梯度剪裁

        :param valid_loader: 验证集导入器 (valid_x, valid_y, *args)

        :return: 训练集的平均损失, 训练集的准确率, 验证集的准确率
        """
        epoch_loss = 0
        train_accuracy, valid_accuracy = AccuracyCell(self.num_class), AccuracyCell(self.num_class)

        for x, y, *args in loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.forward(x, *args)
            batch_loss = loss(y_hat, y)
            epoch_loss += batch_loss.item() / len(loader)

            # 梯度计算
            batch_loss.backward()
            if max_norm > 0:
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()

            # 准确率计算
            train_accuracy += AccuracyCell(self.num_class, torch.argmax(y_hat, dim=-1), y)

        # 训练结束验证
        if valid_loader is not None:
            self.eval()
            for valid_x, valid_y, *valid_args in valid_loader:
                valid_x, valid_y = valid_x.to(self.device), valid_y.to(self.device)
                y_hat = self.forward(valid_x, *valid_args)
                valid_accuracy += AccuracyCell(self.num_class, torch.argmax(y_hat, dim=-1), valid_y)
            self.train()

        return epoch_loss, train_accuracy, valid_accuracy

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