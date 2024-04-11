import torch
from torch import nn


class Scaler(nn.Module):
    def __init__(self, rate):
        super(Scaler, self).__init__()
        self.rate = rate

    def forward(self, X):
        return X * self.rate


class StanderScaler(Scaler):
    def __init__(self, mean=0, std=1):
        """
        均值为 mean, 标准差为 std 的标准放缩

        :param mean: 均值

        :param std: 标准差
        """
        super(StanderScaler, self).__init__()
        self.mean, self.std = mean, std

    def forward(self, x: torch.Tensor, dim=-1):
        """
        对 x 进行标准化放缩
        """
        return self.std * (x - torch.mean(x, dim).reshape(-1, 1) + self.mean) / torch.std(x, dim).reshape(-1, 1)