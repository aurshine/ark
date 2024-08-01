import math
import torch
from torch import nn
from ark.device import use_device
from ark.nn.multi_layers import TransformerLayer, TransformerLayers, FusionChannel


class Encoder(nn.Module):
    def __init__(self, device=None):
        super(Encoder, self).__init__()
        self.device = use_device(device)

    def forward(self, x, **kwargs):
        raise NotImplementedError


class LstmEncoder(Encoder):
    def __init__(self, input_size, hidden_size, num_layer=1, dropout=0, bidirectional=False, device=None):
        super(LstmEncoder, self).__init__(device)
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layer, bidirectional=bidirectional, dropout=dropout,
                            device=self.device)

    def forward(self, x, **kwargs):
        if x is None:
            return None, None

        return self.lstm(x.to(self.device), **kwargs)


class PositionEncoder(Encoder):
    def __init__(self, hidden_size, max_length=300, dropout=0.0):
        """
        :param hidden_size: 输入的最后一维大小

        :param max_length: 最长的编码长度

        :param dropout: dropout值

        对于长为steps, 每个词元由长为 hidden_size 的向量组成

        第 i 个词元的第 2 * j     个特征的位置编码为 p[i][2 * j] = sin(i / 10000^(2 * j / hidden_size))
        第 i 个词元的第 2 * j + 1 个特征的位置编码为 p[i][2 * j + 1] = cos(i / 10000^(2 * j / hidden_size))
        """
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros(size=(1, max_length, hidden_size,))

        # i 的形状为 (steps, 1)
        i = torch.arange(max_length, dtype=torch.float32).reshape(-1, 1)
        # j 的形状为 (hidden_size / 2,)
        j = torch.pow(10000, torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size)

        self.P[:, :, 0::2] = torch.sin(i / j)
        self.P[:, :, 1::2] = torch.cos(i / j)

    def forward(self, x: torch.Tensor, **kwargs):
        """x的形状需要是 (batch_size, steps, hidden_size)"""
        if x.dim() == 4:
            x = x + self.P.unsqueeze(1)[:, :, : x.shape[2], :].to(x.device)
        else:
            x = x + self.P[:, : x.shape[1], :].to(x.device)
        return self.dropout(x)


class ArkEncoder(Encoder):
    """
    词嵌入并融合通道信息
    """
    def __init__(self, vocab, hidden_size, num_channel, steps, dropout=0, device=None):
        """
        :param vocab: 词典

        :param hidden_size: 隐藏层大小

        :param num_channel: 输入通道数

        :param num_heads: 多头注意力层的个数

        :param num_layer: 编码器层数

        :param dropout: dropout值

        :param device: 模型训练的环境 (cpu/gpu)
        """
        super(ArkEncoder, self).__init__(device)
        self.embedding = nn.Embedding(len(vocab), hidden_size, padding_idx=vocab.unk_index, device=self.device)
        self.position_encoding = PositionEncoder(hidden_size)
        self.sqrt_hidden = math.sqrt(hidden_size)
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
        x_embedding = self.embedding(x) * self.sqrt_hidden
        # (batch_size, steps, hidden_size)
        x = self.fusion_ch(x_embedding)
        x = self.position_encoding(x)

        return x
