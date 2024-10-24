from typing import List, Union

import torch
from torch import nn, Tensor

from ark.utils import use_device
from ark.nn.multi_layers import FusionChannel


class Encoder(nn.Module):
    def __init__(self, device=None):
        super(Encoder, self).__init__()
        self.device = use_device(device)

    def forward(self, x, **kwargs):
        raise NotImplementedError


class ArkEncoder(Encoder):
    """
    词嵌入并融合通道信息
    """
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 steps: int,
                 dropout: float = 0,
                 device=None):
        """
        :param vocab_size: 词典大小

        :param hidden_size: 隐藏层大小

        :param dropout: dropout值

        :param device: 模型训练的环境 (cpu/gpu)
        """
        super(ArkEncoder, self).__init__(device)
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, device=self.device)
        self.position_embedding = nn.Embedding(steps, hidden_size, device=self.device)
        self.channel_embedding = nn.Embedding(3, hidden_size, device=self.device)

        self.ln = nn.LayerNorm(hidden_size, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def _word_embedding(self, x):
        """
        对词嵌入进行编码

        :param x: 形状为 (num_channels, batch_size, steps)

        :returns: (num_channels, batch_size, steps, hidden_size)
        """
        return self.word_embedding(x)

    def _position_embedding(self, x):
        """
        对位置信息进行编码

        :param x: 形状为 (num_channels, batch_size, steps)

        :returns: (1, 1, steps, hidden_size)
        """
        steps = x.shape[-1]
        # (1, 1, steps)
        position = torch.arange(steps, dtype=torch.long, device=self.device).reshape(1, 1, steps)
        # (1, 1, steps, hidden_size)
        position_embedding = self.position_embedding(position)

        return position_embedding

    def _channel_embedding(self, x):
        """
        对输入的通道信息进行编码，并融合到词嵌入中

        :param x: 形状为 (num_channels, batch_size, steps)

        :returns: num_channels, 1, 1, hidden_size)
        """
        num_channels = x.shape[0]
        # (num_channels, 1, 1)
        channel = torch.arange(num_channels, dtype=torch.long, device=self.device).reshape(num_channels, 1, 1)
        # (num_channels, 1, 1, hidden_size)
        channel_embedding = self.channel_embedding(channel)
        return channel_embedding

    def forward(self, x: Union[Tensor, List[Tensor]], masks: Tensor = None, **kwargs):
        """
        :param x: 每个tensor的形状为 (batch_size, steps), 如果x为list表示多通道输入

        :param masks: mask形状为 (batch_size, steps)

        :return: 形状为 (batch_size, steps, hidden_size)
        """
        # (num_channels, batch_size, steps)
        if isinstance(x, (list, tuple)):
            if len(x) != self.channel_embedding.num_embeddings:
                raise ValueError(f"{len(x)} != {self.channel_embedding.embedding_dim}, "
                                 f"The number of input channels should be equal to the number of channel embeddings.")
            x = torch.stack(x, dim=0)

        # (num_channels, batch_size, steps, hidden_size)
        x_embedding = self._word_embedding(x) + self._position_embedding(x) + self._channel_embedding(x)
        x_embedding = self.dropout(self.ln(x_embedding))

        return x_embedding, masks
