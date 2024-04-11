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
        self.dense = []
        num_layer = len(num_outputs)

        assert num_layer > 0
        if isinstance(dropout, (int, float)):
            dropout = [dropout] * num_layer
        if len(dropout) < num_layer:
            dropout += [0] * (num_layer - len(dropout))

        for num_output, drop in zip(num_outputs, dropout):
            self.dense.append(nn.LazyLinear(num_output, device=self.device)
                              if num_input is None
                              else nn.Linear(num_input, num_output, device=self.device)
                              )
            if drop > 0:
                self.dense.append(nn.Dropout(drop))
            elif norm == 'batch_norm':
                self.dense.append(nn.BatchNorm1d(num_output, device=self.device))
            elif norm == 'layer_norm':
                self.dense.append(nn.LayerNorm(num_output, device=self.device))

            self.dense.append(active)
            num_input = num_output

        if not save_last_active:
            self.dense.pop()
        self.dense = nn.Sequential(*self.dense)

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
        self.linear1 = MultiLinear([hidden_size, output_size], active=nn.ReLU(), dropout=dropout, num_input=input_size, device=self.device)
        self.linear2 = nn.Linear(input_size, output_size, device=self.device) if input_size != output_size else None
        self.add_norm = AddNorm(output_size, dropout=dropout, device=self.device)

    def forward(self, X):
        if self.linear2 is None:
            return self.add_norm(X, self.linear1(X))
        else:
            return self.add_norm(self.linear2(X), self.linear1(X))


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0, device=None):
        super(TransformerLayer, self).__init__()
        self.device = use_device(device)

        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout, batch_first=True, device=self.device)
        self.add_norm = AddNorm(hidden_size, dropout, device=self.device)
        self.ffn = PositionWiseFFN(hidden_size, hidden_size, hidden_size, dropout, device=self.device)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        return self.ffn(self.add_norm(query, self.attention.forward(query, key, value, key_padding_mask, **kwargs)[0]))