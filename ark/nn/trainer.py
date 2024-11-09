import os
import sys
import logging
from typing import Dict, Union, Optional, List, Tuple, Generator

import torch
from torch import nn
from torch.nn import init

from ark.utils import date_prefix_filename, use_device, get_metrics_str, cpu_ts
from ark.setting import TRAIN_RESULT_PATH


class Trainer(nn.Module):
    def __init__(self, num_class, device=None, prefix_name='trainer'):
        super(Trainer, self).__init__()
        self.device = use_device(device)
        self.num_class = num_class
        self.train_result_path = os.path.join(TRAIN_RESULT_PATH, date_prefix_filename(prefix_name))
        self.checkpoint_path = os.path.join(self.train_result_path, 'checkpoint')
        self.log_path = os.path.join(self.train_result_path, 'log')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        for path in [self.train_result_path, self.checkpoint_path, self.log_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    def forward(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def fit(self,
            train_loader,
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
        self._init_logger()
        # 初始化优化器
        optimizer = self._init_optimizer(optimizer, optim_params)
        # 初始化损失函数
        loss = self._init_loss(loss)
        # 记录训练前的配置信息
        self._log_train_config(optimizer, loss, epochs, stop_loss_value, stop_min_epoch)

        # 记录 每个 epoch 的 loss, 训练集真实标签，验证集预测结果
        loss_list, valid_trues, valid_results = [], [], []

        self.train()
        num_batches = len(train_loader)
        for epoch in range(epochs):
            epoch_loss, valid_true, valid_result = self.fit_epoch(epoch=epoch,
                                                                  train_loader=train_loader,
                                                                  num_batches=num_batches,
                                                                  optimizer=optimizer,
                                                                  loss_fn=loss,
                                                                  valid_loader=valid_loader)
            loss_list.append(epoch_loss)
            valid_results.append(valid_result)
            valid_trues.append(valid_true)

            is_stop = self._achieve_stop_condition(epoch + 1, stop_min_epoch, epoch_loss, stop_loss_value)
            if (epoch + 1) % 10 == 0 or is_stop:
                self.logger.warning(f'Epoch: {epoch + 1}, ValidMetrics:')
                self.logger.warning(get_metrics_str(epoch + 1, valid_result, valid_true))
                self.save_state_dict(os.path.join(self.checkpoint_path, f'epoch{epoch + 1}.pth'))
                if is_stop:
                    break

        return loss_list, valid_trues, valid_results

    def fit_pretrain(self,
                     train_loader,
                     epochs=20,
                     stop_loss_value=1,
                     stop_min_epoch=5,
                     optimizer=None,
                     optim_params: Optional[Dict] = None,
                     loss=None,
                     ):
        # 初始化日志文件
        self._init_logger()
        # 初始化优化器
        optimizer = self._init_optimizer(optimizer, optim_params)
        # 初始化损失函数
        loss = self._init_loss(loss)
        # 记录训练前的配置信息
        self._log_train_config(optimizer, loss, epochs, stop_loss_value, stop_min_epoch)

        self.train()
        num_batches = len(train_loader)
        for epoch in range(epochs):
            epoch_loss, _, _ = self.fit_epoch(epoch=epoch,
                                              train_loader=train_loader,
                                              num_batches=num_batches,
                                              optimizer=optimizer,
                                              loss_fn=loss)
            self.save_state_dict(os.path.join(self.checkpoint_path, f'pretrain_epoch{epoch + 1}.pth'))
            if self._achieve_stop_condition(epoch + 1, stop_min_epoch, epoch_loss, stop_loss_value):
                break

    def fit_epoch(self,
                  epoch: int,
                  train_loader,
                  num_batches,
                  optimizer,
                  loss_fn,
                  valid_loader=None
                  ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        训练一个 epoch 的操作

        :param epoch: 当前训练轮数

        :param train_loader: 训练集导入器,

        :param num_batches: 训练集的batch数

        :param optimizer: 优化器

        :param loss_fn: 计算损失的函数

        :param valid_loader: 验证集导入器

        :return: 训练集的平均损失, 验证集的真实标签(batch_size, ), 验证集的预测结果(batch_size, )
        """
        self.train()
        epoch_loss, y_trues, y_predicts = 0, [], []

        for i, (y_hat, y) in enumerate(self._loader_forward(train_loader)):
            batch_loss = loss_fn(y_hat, y)
            epoch_loss += batch_loss.item()  # / len(train_loader)

            # 梯度计算
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 记录训练集的预测结果
            y_predicts.append(cpu_ts(y_hat.argmax(dim=-1)))
            y_trues.append(cpu_ts(y))
            self.logger.info(f'Epoch {epoch + 1}, Batch ({i + 1}/{num_batches}), Loss: {batch_loss.item():.4f}')

        # 记录训练集的预测结果
        y_predicts = torch.cat(y_predicts)
        y_trues = torch.cat(y_trues)

        self.logger.info(f'Epoch {epoch + 1}, Train A Epoch Average Loss: {epoch_loss:.4f}')
        self.logger.info(get_metrics_str(epoch + 1, y_trues, y_predicts))

        # 训练结束验证
        if valid_loader is not None:
            valid_true, valid_predict = self.validate(valid_loader)
        else:
            valid_true, valid_predict = None, None

        return epoch_loss, valid_true, valid_predict

    def _loader_forward(self, loader) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        传入一个loader, 计算loader的预测结果

        loader 迭代器返回的是一个字典, 除了包含 'tokens' 字段和 'label' 字段外, 其他字段会被传入到forward函数中

        每次yield一个batch的数据, 计算batch的预测结果, 并返回预测结果

        :param loader: 导入器

        :return: 预测结果 和 真实标签
        """
        for data in loader:
            multi_channel_tokens: List[torch.Tensor] = []
            multi_channel_masks: List[torch.Tensor] = []

            y, kwargs = None, {}
            for key, value in data.items():
                if key == 'label':
                    y = value
                elif 'tokens' in key:
                    multi_channel_tokens.append(value['input_ids'])
                    multi_channel_masks = value['attention_mask']
                else:
                    kwargs[key] = value

            if y is None:
                raise ValueError('label not found in data')

            multi_channel_tokens = self._to_device(multi_channel_tokens)
            multi_channel_masks = self._to_device(multi_channel_masks)
            y = self._to_device(y)

            y_hat = self.forward(multi_channel_tokens, multi_channel_masks, **kwargs)
            yield y_hat, y

    @staticmethod
    def _achieve_stop_condition(epoch: int, stop_min_epoch: int, loss: float, stop_max_loss: float) -> bool:
        """
        达到停止条件时返回 True, 否则返回 False
        """
        return epoch >= stop_min_epoch and loss <= stop_max_loss

    def validate(self, valid_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        验证模式, 计算真实标签和预测结果

        :param valid_loader: 验证集导入器

        :return: y_true, y_predicts
        """
        self.eval()
        y_trues, y_predicts = [], []
        with torch.no_grad():
            for y_hat, y in self._loader_forward(valid_loader):
                y_predicts.append(cpu_ts(y_hat.argmax(dim=-1)))
                y_trues.append(cpu_ts(y))

        return torch.cat(y_trues), torch.cat(y_predicts)

    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        进入eval模式, 预测输入数据 x 的标签

        :param x: 形状与 forward 函数的输入相同

        :param args: 其他参数

        :param kwargs: 其他参数

        :returns: 返回长度为batch_size的一维tensor
        """
        self.eval()
        with torch.no_grad():
            y = self.forward(x, *args, **kwargs)

        if 0 < self.num_class < y.shape[-1]:
            y = y[:, :self.num_class]
        return cpu_ts(torch.argmax(y, dim=-1))

    def analyse(self, x: torch.Tensor, classes: list, **kwargs) -> list:
        """
        :param x: 形状与forward函数的输入相同

        :param classes:  列表组成的分类名, ['非恶意', '恶意']

        :return: 长度为 batch_size 的 list
        """
        return [classes[y] for y in self.predict(x, **kwargs)]

    def save_state_dict(self, path: str):
        """
        保存模型

        :param path: 保存文件地址
        """
        path = date_prefix_filename(path)
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

    def _init_logger(self):
        """
        初始化日志文件
        """
        self.logger.handlers.clear()

        log_file = os.path.join(self.log_path, 'train.log')
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
            return [self._to_device(t) for t in ts]
        elif isinstance(ts, dict):
            return {k: self._to_device(v) for k, v in ts.items()}
        elif isinstance(ts, torch.Tensor):
            return ts.to(self.device)
        else:
            raise ValueError(f'Unsupported type: {type(ts)} for _to_device')

    def _init_optimizer(self, optimizer, optim_params):
        if optimizer is None:
            if optim_params is None:
                optim_params = {'lr': 2e-3, 'weight_decay': 0.1}
            optimizer = torch.optim.AdamW(self.parameters(), **optim_params)
        return optimizer

    def _init_loss(self, loss):
        if loss is None:
            loss = nn.CrossEntropyLoss()
        return loss

    def _log_train_config(self, optimizer, loss, epochs, stop_loss_value, stop_min_epoch):
        # 打印训练信息
        self.logger.info(f'model architecture: {self}')
        self.logger.info(f'optimizer: {optimizer}')
        self.logger.info(f'loss_fn: {loss}')
        self.logger.info(f'fit on {self.device}')
        self.logger.info(f'num_params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
        self.logger.info(f'num_class: {self.num_class}')
        self.logger.info(f'epochs: {epochs}')
        self.logger.info(f'stop_loss_value: {stop_loss_value}')
        self.logger.info(f'stop_min_epoch: {stop_min_epoch}\n')