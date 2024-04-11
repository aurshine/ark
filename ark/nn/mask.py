from typing import Union
import torch
from torch import nn


def sequence_back_mask_(X: torch.Tensor, valid_len: torch.Tensor, masked_value: Union[int, float, torch.Tensor] = 0):
    """
     保留 sequence[i] 的前 valid_len[i] 个特征, 剩下特征用 masked_value 掩盖

     直接对输入的 X 修改

    X 的维度需要大于 valid_len 的维度
    """
    assert X.dim() > valid_len.dim()
    max_len = X.size(valid_len.dim())

    mask = valid_len.unsqueeze(-1) <= torch.arange(max_len).repeat(list(valid_len.shape) + [1])

    X[mask] = masked_value

    return X


def sequence_back_mask(X: torch.Tensor, valid_len: torch.Tensor, masked_value=0):
    """
    保留 sequence[i] 的前 valid_len[i] 个特征, 剩下特征用 maked_value 掩盖

    复制新的 X 修改

    X 的维度需要大于 valid_len 的维度
    """
    X = X.detach().clone()

    return sequence_back_mask_(X, valid_len, masked_value)


def sequence_front_mask_(X: torch.Tensor, valid_len: torch.Tensor, masked_value: Union[int, float, torch.Tensor] = 0):
    """
    用masked_value 遮掩 sequence[i] 的前 valid_len[i] 个特征

    直接对输入的 X 修改

    X 的维度需要大于 valid_len 的维度
    """
    assert X.dim() > valid_len.dim()
    max_len = X.size(valid_len.dim())

    mask = valid_len.unsqueeze(-1) > torch.arange(max_len).repeat(list(valid_len.shape) + [1])

    X[mask] = masked_value

    return X


def sequence_front_mask(X: torch.Tensor, valid_len: torch.Tensor, masked_value: Union[int, float, torch.Tensor] = 0):
    """
    用masked_value 遮掩 sequence[i] 的前 valid_len[i] 个特征

    复制新的 X 修改

    X 的维度需要大于 valid_len 的维度
    """
    X = X.detach().clone()
    return sequence_front_mask_(X, valid_len, masked_value)


def sequence_mask_(X: torch.Tensor, valid_len: torch.Tensor, masked_value: Union[int, float, torch.Tensor] = 0, front_mask: bool = True):
    if front_mask:
        return sequence_front_mask_(X, valid_len, masked_value)
    else:
        return sequence_back_mask_(X,  valid_len, masked_value)


def sequence_mask(X: torch.Tensor, valid_len: torch.Tensor, masked_value: Union[int, float, torch.Tensor] = 0, front_mask: bool = True):
    if front_mask:
        return sequence_front_mask(X, valid_len, masked_value)
    else:
        return sequence_back_mask(X,  valid_len, masked_value)


def masked_softmax(X: torch.Tensor, valid_len: torch.Tensor, front_mask, dim=-1):
    if valid_len is not None:
        X = sequence_mask_(X, valid_len, -1e6, front_mask)
    return nn.functional.softmax(X, dim)


class MaskedSoftmax(nn.Module):
    def __init__(self, valid_len, front_mask: bool):
        super(MaskedSoftmax, self).__init__()
        self.valid_len = valid_len
        self.front_mask = front_mask

    def forward(self, X):
        return masked_softmax(X, self.valid_len, self.front_mask)