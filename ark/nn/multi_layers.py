from typing import Union, List, Tuple, Optional, Callable

import torch
from torch import nn

from ark.device import use_device
from ark.nn.addnorm import AddNorm
from ark.nn.attention import cosine_similarity_attention


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


class MultiLinear(nn.Module):
    def __init__(self,
                 num_outputs: List[int],
                 active,
                 dropout: float = 0,
                 norm=None,
                 num_input=None,
                 save_last_active=False,
                 device=None):
        """可以实现多个全连接层的网络

        :param num_outputs: 每层的输出节点数

        :param active: 每层输出后的激活函数

        :param dropout: 抛弃层, 在每层的输出之前

        :param norm: 归一化层, 可选 'batch_norm' 'layer_norm' 在每层的输出之后, 激活函数之前

        :param num_input: 输入节点数, 选填, 默认为 None 时自动计算输入节点

        :param save_last_active: 是否保留最后一层网络后的激活函数
        """
        super(MultiLinear, self).__init__()

        self.device = use_device(device)
        layers = []
        num_layer = len(num_outputs)

        assert num_layer > 0
        for num_output in num_outputs:
            layers.append(nn.LazyLinear(num_output, device=self.device)
                          if num_input is None
                          else nn.Linear(num_input, num_output, device=self.device)
                          )
            if norm == 'batch_norm':
                layers.append(nn.BatchNorm1d(num_output, device=self.device))
            elif norm == 'layer_norm':
                layers.append(nn.LayerNorm(num_output, device=self.device))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            layers.append(active)
            num_input = num_output

        if not save_last_active:
            layers.pop()
        self.dense = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense(x)


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


class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, device=None):
        super(PositionWiseFFN, self).__init__()
        self.device = use_device(device)
        self.linear = MultiLinear([hidden_size, output_size],
                                  num_input=input_size,
                                  active=nn.LeakyReLU(),
                                  norm='layer_norm',
                                  dropout=dropout,
                                  save_last_active=True,
                                  device=self.device)

        if input_size == output_size:
            self.add_norm = AddNorm(output_size, dropout=dropout, device=self.device)

    def forward(self, x):
        y = self.linear(x)

        if hasattr(self, 'add_norm'):
            y = self.add_norm(x, y)

        return y


class Attention(nn.Module):
    def __init__(self,
                 query_size: int,
                 key_size: int,
                 value_size: int = None,
                 hidden_size: Optional[int] = None,
                 device=None):
        """
        注意力机制

        :param query_size: query_size

        :param key_size: key_size

        :param value_size: 如果为None, 则不会改变value_size, 否则 hidden_size 也不为None时，会改变value_size

        :param hidden_size: 如果不为None，则将query, key, value先通过线性变换映射到hidden_size维度，再进行注意力计算

        :param device: device
        """
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.device = use_device(device)

        value_size = value_size if value_size is not None else key_size
        if self.hidden_size is not None:
            self.query2hidden = nn.Linear(query_size, hidden_size, device=self.device)
            self.key2hidden = nn.Linear(key_size, hidden_size, device=self.device)
            if value_size is not None:
                self.value2hidden = nn.Linear(value_size, hidden_size, device=self.device)
            else:
                self.value2hidden = None

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                get_qk_weight: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                masks: Optional[torch.Tensor] = None):
        """
        :param queries: (batch_size, query_steps, query_size)

        :param keys: (batch_size, kv_steps, key_size)

        :param values: (batch_size, kv_steps, value_size)

        :param get_qk_weight: 传入query key， 得到每个key的权重，用于计算注意力

        :param masks: (batch_size, query_steps)

        :return: (batch_size, query_steps, value_size)
        """
        if self.hidden_size is not None:
            # (batch_size, query_steps, hidden_size)
            queries = self.query2hidden(queries)
            keys = self.key2hidden(keys)
            if values is not None:
                values = self.value2hidden(values)

        # (batch_size, query_steps, kv_steps)
        weight = get_qk_weight(queries, keys)
        if masks is not None:
            masks = masks.unsqueeze(2).expand_as(weight)

            weight = weight.masked_fill(masks == 0, -1e9)

        # (batch_size, query_steps, value_size)
        output = torch.bmm(weight, values)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_size: int,
                 key_size: int,
                 num_heads: int,
                 head_hidden_size: Optional[int] = None,
                 device=None):
        """
        多头注意力机制

        :param query_size: query_size

        :param key_size: key_size

        :param num_heads: 注意力头数

        :param head_hidden_size: 每个头的隐藏层大小，如果不为None，则将query, key, value先通过线性变换映射到head_hidden_size维度，再进行注意力计算

        :param device: device
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.device = use_device(device)
        assert query_size % num_heads == 0, "query size should be divisible by num_heads"
        assert key_size % num_heads == 0, "key size should be divisible by num_heads"

        self.attention = Attention(query_size // num_heads,
                                   key_size // num_heads,
                                   head_hidden_size,
                                   device=self.device)

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                get_qk_weight: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                masks: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        :param queries: (batch_size, query_steps, query_size)

        :param keys: (batch_size, kv_steps, key_size)

        :param values: (batch_size, kv_steps, value_size)

        :param get_qk_weight: 传入query key， 得到每个key的权重，用于计算注意力

        :param masks: (batch_size, query_steps)

        :return: (batch_size, query_steps, value_size)
        """
        queries = separate_heads(queries, self.num_heads)
        keys = separate_heads(keys, self.num_heads)
        values = separate_heads(values, self.num_heads)

        if masks is not None:
            masks = masks.repeat_interleave(self.num_heads, dim=0)

        # (batch_size * num_heads, query_steps, value_size // num_heads)
        heads = self.attention(queries, keys, values, get_qk_weight, masks=masks)
        # (batch_size, num_heads, query_steps, value_size // num_heads)
        heads = concat_heads(heads, self.num_heads)

        return heads


class TransformerLayer(nn.Module):
    """
    Transformer 块

    由 MultiheadAttention -> Addnorm -> PositionWiseFFN -> Addnorm 组成
    """

    def __init__(self,
                 query_size: int,
                 num_heads: int,
                 key_size: int = None,
                 value_size: int = None,
                 hidden_size: Optional[int] = None,
                 dropout=0.5,
                 device=None):
        """
        transformer 块

        :param query_size: query_size

        :param num_heads: 注意力头数

        :param key_size: key_size, 如果为None则默认为 query_size

        :param value_size: value_size, 如果为None则默认为 key_size

        :param hidden_size: 如果不为None，则将query, key, value先通过线性变换映射到hidden_size维度，再进行注意力计算

        :param dropout: dropout

        :param device: device
        """
        super(TransformerLayer, self).__init__()
        self.device = use_device(device)
        key_size = query_size if key_size is None else key_size
        value_size = key_size if value_size is None else value_size

        self.attention = MultiHeadAttention(query_size, key_size, num_heads, hidden_size, device=self.device)
        self.ffn = PositionWiseFFN(value_size, value_size, value_size, device=self.device)
        self.add_norm1 = AddNorm(value_size, dropout=dropout, device=self.device)
        self.add_norm2 = AddNorm(value_size, dropout=dropout, device=self.device)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor = None,
                v: torch.Tensor = None,
                masks: torch.Tensor = None,
                get_qk_weight: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None):
        """
        计算 transformer 块的输出

        :param q: query

        :param k: key

        :param v: value

        :param masks: 掩码，形状为(batch_size, query_steps)

        :param get_qk_weight: 获取query key的权重，用于计算注意力，默认为cosine_similarity_attention

        :return: 与value形状相同的输出
        """
        if get_qk_weight is None:
            get_qk_weight = cosine_similarity_attention

        k = q if k is None else k
        v = k if v is None else v

        y1 = self.attention(q, k, v, get_qk_weight, masks=masks)
        y1 = self.add_norm1(q, y1)
        y2 = self.ffn(y1)
        y2 = self.add_norm2(y1, y2)

        return y2


class TransformerLayers(nn.Module):
    """
    多层的 Transformer 块

    由多层的 TransformerLayer 组成
    """

    def __init__(self,
                 query_size: int,
                 num_heads: int,
                 num_layer: int,
                 key_size: int = None,
                 value_size: int = None,
                 hidden_size: int = None,
                 dropout: float = 0.5,
                 device=None):
        super(TransformerLayers, self).__init__()
        self.device = use_device(device)
        self.transformer_blocks = nn.ModuleList([TransformerLayer(query_size, num_heads, key_size, value_size,
                                                                  hidden_size, dropout=dropout, device=self.device)
                                                 for _ in range(num_layer)])

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor = None,
                v: torch.Tensor = None,
                masks: torch.Tensor = None):
        """
        :return: 形状为(batch_size, steps, hidden_size)
        """
        y = q
        for block in self.transformer_blocks:
            y = block(q, k, v, masks=masks)

        return y


class FusionChannel(nn.Module):
    def __init__(self, hidden_size, num_channel, steps=None, dropout=0.5, device=None):
        super(FusionChannel, self).__init__()
        self.device = use_device(device)
        if steps is None:
            steps = 1

        self.c_weight = nn.Parameter(torch.randn(size=(steps, num_channel, 1), device=self.device), requires_grad=True)
        self.lin = MultiLinear([hidden_size, hidden_size],
                               dropout=dropout,
                               active=nn.LeakyReLU(),
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
        assert x.dim() in [3, 4], "x should be 3D or 4D"

        expect_dim = x.dim() - 1
        if x.dim() == 3:
            x = torch.unsqueeze(x, dim=1)

        y = torch.sum(self.c_weight * x, dim=-2)
        y = self.lin(y)

        if expect_dim == 2:
            y = torch.squeeze(y, dim=-2)

        return y
