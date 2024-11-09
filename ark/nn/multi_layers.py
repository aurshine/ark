from typing import List, Optional, Callable

import torch
from torch import nn

from ark.utils import use_device
from ark.nn.addnorm import AddNorm
from ark.nn.attention import cosine_similarity_attention, separate_heads, concat_heads


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


class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, device=None):
        super(PositionWiseFFN, self).__init__()
        self.device = use_device(device)
        self.linear = MultiLinear([hidden_size, output_size],
                                  num_input=input_size,
                                  active=nn.GELU(),
                                  dropout=dropout,
                                  save_last_active=True,
                                  device=self.device)

    def forward(self, x):
        return self.linear(x)


class Attention(nn.Module):
    def __init__(self,
                 query_size: int,
                 key_size: int,
                 value_size: Optional[int] = None,
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
        self.dropout = nn.Dropout(0.1)

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                get_qk_weight: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                masks: Optional[torch.Tensor] = None):
        """
        :param queries: (batch_size, query_steps, query_size)

        :param keys: (batch_size, key_steps, key_size)

        :param values: (batch_size, value_steps, value_size)

        :param get_qk_weight: 传入query key， 得到每个key的权重，用于计算注意力

        :param masks: 与queries[:-1]形状相同的掩码

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
            masks = masks.unsqueeze(-1).expand_as(weight)

            weight = weight.masked_fill(masks == 0, -1e9)

        # (batch_size, query_steps, value_size)
        output = self.dropout(torch.bmm(weight, values))

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

        :param keys: (batch_size, key_steps, key_size)

        :param values: (batch_size, value_steps, value_size)

        :param get_qk_weight: 传入query key， 得到每个key的权重，用于计算注意力

        :param masks: 与queries[:-1]形状相同的掩码

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
                 get_qk_weight: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine_similarity_attention,
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
        self.get_qk_weight = get_qk_weight

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor = None,
                v: torch.Tensor = None,
                masks: torch.Tensor = None):
        """
        计算 transformer 块的输出

        :param q: (batch_size, query_steps, query_size)

        :param k: (batch_size, key_steps, key_size)

        :param v: (batch_size, value_steps, value_size)

        :param masks: 与q[:-1]形状相同的掩码

        :return: (batch_size, query_steps, value_size)
        """
        k = q if k is None else k
        v = k if v is None else v

        y1 = self.attention(q, k, v, self.get_qk_weight, masks=masks)
        y1 = self.add_norm1(q, y1)
        y2 = self.ffn(y1)
        y2 = self.add_norm2(y1, y2)

        return y2