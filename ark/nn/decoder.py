from collections import deque
import torch
from torch import nn
from ark.device import use_device
from ark.nn.multi_layers import PositionWiseFFN, TransformerLayer, HistoryTransformerLayers


class Decoder(nn.Module):
    def __init__(self, device=None):
        super(Decoder, self).__init__()
        self.device = use_device(device)

    def init_state(self, enc_output, *args):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        """:return: hidden, states"""
        raise NotImplementedError


class ArkDecoder(Decoder):
    def __init__(self, hidden_size, num_heads, num_layer, num_steps, dropout=0, device=None):
        super(ArkDecoder, self).__init__(device)
        self.history_layers = HistoryTransformerLayers(hidden_size, num_heads, num_layer, dropout=dropout, device=self.device)
        self.flatten = nn.Flatten()
        self.ffn = PositionWiseFFN(hidden_size * num_steps, hidden_size, hidden_size, dropout=dropout, device=self.device)
        self.fusion = TransformerLayer(hidden_size, num_heads, dropout=dropout, device=self.device)

    def init_state(self, enc_output, *args):
        return None

    def forward(self, X, **kwargs):
        """
        :param X:  形状为 (batch_size, steps, hidden_size)

        :param kwargs: MultiHeadAttention 的其它参数
        """
        X = X.to(self.device)
        X = self.history_layers(X, **kwargs)

        # 形状为 (batch_size, 1, hidden_size)
        query = self.ffn(self.flatten(X)).unsqueeze(1)

        return self.fusion(query, X, X, **kwargs).squeeze(1)
