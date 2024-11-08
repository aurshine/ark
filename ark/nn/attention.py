import math

import torch
from torch import nn

from ark.utils import use_device


def separate_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    将输入的tensor分割成多个头，并转置

    :param x: (batch_size, ..., feature_size)

    :param num_heads: 头数

    :return: (batch_size * num_heads, ..., feature_size // num_heads)
    """
    if x.shape[-1] % num_heads != 0:
        raise ValueError("feature size should be divisible by num_heads")

    # [...]
    batch_size, other_dim = x.shape[0], x.shape[1: -1]

    # (batch_size, 1, ..., num_heads, feature_size // num_heads)
    y = x.reshape(batch_size, 1, *other_dim, num_heads, -1)

    # (batch_size, num_heads, ...., feature_size // num_heads)
    y = y.transpose(1, -2)

    # (batch_size * num_heads, ..., feature_size // num_heads)
    y = y.reshape(batch_size * num_heads, *other_dim, -1)

    return y


def concat_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    将输入的tensor合并成多个头

    :param x: (batch_size * num_heads, ..., feature_size // num_heads)

    :param num_heads: 头数

    :return: (batch_size, ..., feature_size)
    """
    if x.shape[0] % num_heads != 0:
        raise ValueError("batch size should be divisible by num_heads")

    feature_size, other_dim = x.shape[-1], x.shape[1:-1]

    # (batch_size, num_heads, ..., 1, feature_size // num_heads)
    x = x.reshape(-1, num_heads, *other_dim, 1, feature_size)
    # (batch_size, 1, ... num_heads, feature_size // num_heads)
    x = x.transpose(1, -2)

    # (batch_size, ..., feature_size)
    x = x.reshape(x.shape[0], *other_dim, -1)
    return x


def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """
    缩放点积注意力

    :param queries: (batch_size, query_steps, hidden_size)

    :param keys: (batch_size, kv_steps, hidden_size)

    :return: (batch_size, query_steps, kv_steps)
    """
    # (batch_size, hidden_size, kv_steps)
    keys = torch.transpose(keys, 1, 2)
    d = queries.shape[-1]

    # (batch_size, query_steps, kv_steps)
    scores = torch.bmm(queries, keys) / (math.sqrt(d))

    # (batch_size, query_steps, kv_steps)
    scores = torch.softmax(scores, dim=-1)

    return scores


def cosine_similarity_attention(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """
    余弦相似度注意力

    :param queries: (batch_size, query_steps, hidden_size)

    :param keys: (batch_size, kv_steps, hidden_size)

    :return: (batch_size, query_steps, kv_steps)
    """
    d = queries.shape[-1]

    # (batch_size, query_steps, kv_steps)
    products = torch.bmm(queries, torch.transpose(keys, 1, 2))
    # (batch_size, query_steps, 1, hidden_size)
    queries = queries.unsqueeze(dim=2)
    # (batch_size, 1, kv_steps, hidden_size)
    keys = keys.unsqueeze(dim=1)

    qk_mod = torch.norm(queries) * torch.norm(keys) / d

    # (batch_size, query_steps, kv_steps)
    scores = torch.softmax(products / (torch.sqrt(qk_mod) + 1e-8), dim=-1)

    return scores


class LearningPosition(nn.Module):
    def __init__(self, steps, hidden_size, device=None):
        super(LearningPosition, self).__init__()
        self.device = use_device(device)
        self.P = nn.Parameter(torch.zeros((1, steps, hidden_size), device=self.device), requires_grad=True)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        return _input + self.P
