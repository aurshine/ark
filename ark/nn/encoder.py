import math

import torch
from torch import nn

from ark.device import use_device
from ark.nn.multi_layers import FusionChannel
from ark.nn.attention import LearningPosition


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
    def __init__(self, vocab, hidden_size, num_channel, steps, dropout=0, device=None):
        """
        :param vocab: 词典

        :param hidden_size: 隐藏层大小

        :param num_channel: 输入通道数

        :param dropout: dropout值

        :param device: 模型训练的环境 (cpu/gpu)
        """
        super(ArkEncoder, self).__init__(device)
        self.embedding = nn.Embedding(len(vocab), hidden_size, padding_idx=vocab.unk_index, device=self.device)
        self.position_encoding = LearningPosition(steps, hidden_size)

        self.fusion_ch = FusionChannel(hidden_size=hidden_size,
                                       num_channel=num_channel,
                                       steps=steps,
                                       dropout=dropout,
                                       device=self.device
                                       )

    def forward(self, x: torch.Tensor, valid_len=None):
        """
        :param x: 形状为 (batch_size, num_channels, steps)

        :param valid_len: 形状为 (batch_size, )

        :return: 形状为 (batch_size, steps, hidden_size)
        """
        # (batch_size, steps, num_channels)
        x = x.transpose(1, 2)
        # (batch_size, steps, num_channels, hidden_size)
        x_embedding = self.embedding(x)
        # (batch_size, steps, hidden_size)
        x = self.fusion_ch(x_embedding)
        x = self.position_encoding(x)

        return x
