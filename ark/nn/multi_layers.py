from collections import deque
from typing import Union, List, Tuple, Optional

import torch
from torch import nn

from ark.device import use_device
from ark.nn.addnorm import AddNorm


class MultiLinear(nn.Module):
    def __init__(self,
                 num_outputs: List[int],
                 active,
                 dropout: Union[List[float], float] = 0,
                 norm=None,
                 num_input=None,
                 save_last_active=False,
                 device=None):
        """可以实现多个全连接层的网络

        :param num_outputs: 每层的输出节点数

        :param active: 每层输出后的激活函数, 最后一层没有

        :param dropout: 抛弃层, 在每层的输出之前, 如果为 list, 长度需要不大于 num_outputs

        :param norm: 归一化层, 可选 'batch_norm' 'layer_norm' 在每层的输出之后, 激活函数之前, 不与 dropout 同时使用

        :param num_input: 输入节点数, 选填, 默认为 None 时自动计算输入节点

        :param save_last_active: 是否保留最后一层网络后的激活函数
        """
        super(MultiLinear, self).__init__()

        self.device = use_device(device)
        layers = []
        num_layer = len(num_outputs)

        assert num_layer > 0
        if isinstance(dropout, (int, float)):
            dropout = [dropout] * num_layer
        if len(dropout) < num_layer:
            dropout += [0] * (num_layer - len(dropout))

        for num_output, drop in zip(num_outputs, dropout):
            layers.append(nn.LazyLinear(num_output, device=self.device)
                              if num_input is None
                              else nn.Linear(num_input, num_output, device=self.device)
                              )
            if drop > 0:
                layers.append(nn.Dropout(drop))
            elif norm == 'batch_norm':
                layers.append(nn.BatchNorm1d(num_output, device=self.device))
            elif norm == 'layer_norm':
                layers.append(nn.LayerNorm(num_output, device=self.device))

            layers.append(active)
            num_input = num_output

        if not save_last_active:
            layers.pop()
        self.dense = nn.Sequential(*layers)

    def forward(self, X):
        return self.dense(X.to(self.device))


class MultiConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 num_layer=1,
                 batch_norm=False,
                 active=None,
                 device=None):
        """多层同样操作的卷积

        :param in_channels: 输入通道数

        :param out_channels: 输出通道数

        :param kernel_size: 卷积核大小

        :param stride: 步长, 默认步长为 1

        :param padding: 填充大小, 默认填充为 0

        :param num_layer: 网络层数, 默认 3 层

        :param batch_norm: 是否启用批量归一化

        :param active: 激活函数, 默认为 None 时使用 ReLU
        """
        super(MultiConv2d, self).__init__()
        self.device = use_device(device)

        def trans(x):
            if not isinstance(x, (list, tuple)):
                x = (x, x)
            return x

        kernel_size, stride, padding = trans(kernel_size), trans(stride), trans(padding)
        if active is None:
            active = nn.ReLU()

        layers = []
        for i in range(num_layer):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device=self.device))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels, device=self.device))
            layers.append(active)
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

    def forward(self, X):
        return self.conv(X.to(self.device))


class MultiEmbedding(nn.Module):
    def __init__(self, num_embeddings: List[int], embedding_size: int, paddings_idx: Optional[List[int]] = None, **kwargs):
        """
        :param num_embeddings: 每个词表里词元的数量

        :param embedding_size: 向量化的长度

        :param padding_idx: 填充词元的下标

        :param kwargs: nn.Embedding 的其它参数
        """
        super(MultiEmbedding, self).__init__()

        if paddings_idx is None:
            paddings_idx = [None] * len(num_embeddings)

        self.embedding_layers = [nn.Embedding(num_embedding, embedding_size, padding_idx, **kwargs)
                                 for num_embedding, padding_idx in zip(num_embeddings, paddings_idx)
                                 ]

    def forward(self, X: torch.Tensor):
        """将数字向量化, 每个通道对应不同的 Embedding 层

        :param X: 形状为 (batch_size, num_channels, steps)

        :return: (batch_size, num_channels, steps, embedding_size)
        """
        assert X.shape[1] == len(self.embedding_layers)

        X = X.permute(1, 0, 2)

        Y = [embedding(x) for x, embedding in zip(X, self.embedding_layers)]

        return torch.stack(Y).permute(1, 0, 2, 3)


class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, device=None):
        super(PositionWiseFFN, self).__init__()
        self.device = use_device(device)
        self.linear1 = MultiLinear([hidden_size, output_size], active=nn.LeakyReLU(), dropout=dropout, num_input=input_size, device=self.device)
        self.linear2 = nn.Linear(input_size, output_size, device=self.device) if input_size != output_size else None
        self.add_norm = AddNorm(output_size, dropout=dropout, device=self.device)

    def forward(self, X):
        if self.linear2 is None:
            return self.add_norm(X, self.linear1(X))
        else:
            return self.add_norm(self.linear2(X), self.linear1(X))


class TransformerLayer(nn.Module):
    """
    Transformer 块

    由 MultiheadAttention -> Addnorm -> PositionWiseFFN -> Addnorm 组成
    """
    def __init__(self, hidden_size, num_heads, dropout=0, device=None):
        super(TransformerLayer, self).__init__()
        self.device = use_device(device)

        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout, batch_first=True, device=self.device)
        self.add_norm = AddNorm(hidden_size, dropout, device=self.device)
        self.ffn = PositionWiseFFN(hidden_size, hidden_size, hidden_size, dropout, device=self.device)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """
        :param query: 形状为 (batch_size, query_steps, hidden_size)

        :param key: 形状为 (batch_size, key_steps, hidden_size)

        :param value: 形状为 (batch_size, key_steps, hidden_size)

        :param key_padding_mask: BoolTensor类型 形状为 (batch_size, key_steps)

        :param kwargs: 可选参数, 用于 nn.MultiheadAttention 的其它参数

        :return: 形状为 (batch_size, query_steps, hidden_size)
        """
        return self.ffn(self.add_norm(query, self.attention(query, key, value, key_padding_mask, **kwargs)[0]))


class TransformerLayers(nn.Module):
    """
    多层的 Transformer 块

    由多层的 TransformerLayer 组成
    """
    def __init__(self, hidden_size, num_heads, num_layer=1, dropout=0, device=None):
        super(TransformerLayers, self).__init__()
        self.device = use_device(device)
        self.transformer_blocks = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout, device=self.device)
                                                 for _ in range(num_layer)])

    def forward(self, x, **kwargs):
        """
        :param x: 形状为(batch_size, steps, num_hidden)

        :return: 形状为(batch_size, steps, num_hidden)
        """
        x = x.to(self.device)

        for block in self.transformer_blocks:
            x = block(x, x, x, **kwargs)

        return x


class HistoryTransformerLayers(nn.Module):
    """
    记录历史的 Transformer 层

    由多层的TransformerLayer组成

    每一层的 key-value 由前 max_history_len 层的输出在steps维度拼接组成
    """
    def __init__(self, hidden_size, num_heads, num_layers, max_history_len=3, dropout=0, device=None):
        super(HistoryTransformerLayers, self).__init__()
        self.device = use_device(device)
        self.attentions = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout, device=self.device) for _ in range(num_layers)])
        self.max_history_len = max_history_len

    def forward(self, x, key_padding_mask=None, **kwargs):
        """
        :param x: 形状为(batch_size, steps, num_hidden)


        :param key_padding_mask: BoolTensor类型 形状为 (batch_size, key_steps)

        :param kwargs: 可选参数, 用于 nn.MultiheadAttention 的其它参数

        :return: 形状为(batch_size, steps, num_hidden)
        """
        deque_key_values = deque([x], maxlen=self.max_history_len)

        for attention in self.attentions:
            key_values = torch.cat(list(deque_key_values), dim=1)
            x = attention(x, key_values, key_values, key_padding_mask=key_padding_mask, **kwargs)
            deque_key_values.append(x)
            key_padding_mask = None

        return x


class FusionChannel(nn.Module):
    def __init__(self, hidden_size, num_channel, steps=None, dropout=0.5, device=None):
        super(FusionChannel, self).__init__()
        self.device = use_device(device)
        if steps is None:
            steps = 1

        self.c_weight = nn.Parameter(torch.randn(size=(steps, num_channel, 1), device=self.device), requires_grad=True)
        self.lin = MultiLinear([hidden_size, hidden_size],
                               dropout=dropout,
                               active=nn.ReLU(),
                               num_input=hidden_size,
                               norm='layer_norm',
                               save_last_active=True,
                               device=self.device
                               )

    def forward(self, x: torch.Tensor):
        """
        :param x: 3D-(batch_size, channels, hidden_size), 4D-(batch_size, steps, channels, hidden_size)

        :return: 2D-(batch_size, hidden_size) if x is 3D else 3D-(batch_size, steps, hidden_size)
        """
        if x.dim() == 3:
            x = torch.unsqueeze(x, dim=1)

        y = torch.sum(self.c_weight * x, dim=-2)
        y = self.lin(y)
        return y