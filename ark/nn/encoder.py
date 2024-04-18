import math
import torch
from torch import nn
from ark.device import use_device
from ark.nn.multi_layers import TransformerLayer


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


class ArkEncoderBlock(Encoder):
    def __init__(self, hidden_size, num_heads, num_layer=1, dropout=0, device=None):
        super(ArkEncoderBlock, self).__init__(device)
        self.transformer_blocks = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout, device=self.device)
                                                 for _ in range(num_layer)])

    def forward(self, X, **kwargs):
        """
        :param X: 形状为(batch_size, steps, num_hidden)

        :return: 形状为(batch_size, steps, num_hidden)
        """
        X = X.to(self.device)

        for block in self.transformer_blocks:
            X = block(X, X, X, **kwargs)

        return X


class ArkEncoder(Encoder):
    def __init__(self, vocab, hidden_size, num_channel, num_heads, num_layer=1, dropout=0, device=None):
        super(ArkEncoder, self).__init__(device)
        self.embedding = nn.Embedding(len(vocab), hidden_size, padding_idx=vocab.unk_index, device=self.device)
        self.position_encoding = PositionEncoder(hidden_size)
        self.sqrt_hidden = math.sqrt(hidden_size)
        self.encoder_blocks = nn.ModuleList([ArkEncoderBlock(hidden_size, num_heads, num_layer, dropout, device=self.device)
                                             for _ in range(num_channel)])
        self.fusion = TransformerLayer(hidden_size, num_heads, dropout, device=self.device)

    def forward(self, X, valid_len=None):
        """
        :param X: 形状为 (batch_size, num_channels, steps)

        :param valid_len: 形状为 (batch_size, )

        :return: 形状为 (batch_size, steps, hidden_size)
        """
        X = X.to(self.device).permute(1, 0, 2)

        # [原字通道, piny通道, 首字母通道], 形状均为 (batch_size, steps, hidden_size)
        Xs = [self.position_encoding(self.embedding(x) * self.sqrt_hidden) for x in X]
        steps = X.shape[-1]

        valid_len = steps - valid_len
        mask = torch.arange(steps).repeat(valid_len.shape[0], 1) < valid_len.repeat_interleave(steps).reshape(-1, steps)
        mask = mask.to(self.device)

        Ys = [block(x, key_padding_mask=mask) for x, block in zip(Xs, self.encoder_blocks)]
        Y = torch.cat(Ys, dim=1)

        return self.fusion(Xs[0], Y, Y)
