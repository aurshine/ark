import torch
from torch import nn
from ark.device import use_device
from ark.nn.multi_layers import TransformerLayers, TransformerLayer


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
    def __init__(self, hidden_size, num_heads, num_layer, dropout=0, device=None):
        super(ArkDecoder, self).__init__(device)
        self.transformer_layers = TransformerLayers(hidden_size, num_heads, num_layer, hidden_size=hidden_size, dropout=dropout, device=self.device)
        self.query = nn.Parameter(torch.empty(size=(1, 1, hidden_size), device=self.device))
        nn.init.xavier_normal_(self.query)
        self.fusion = TransformerLayer(hidden_size, num_heads, dropout=dropout, device=self.device)

    def init_state(self, enc_output, *args):
        return None

    def forward(self, x, **kwargs):
        """
        :param x:  形状为 (batch_size, steps, hidden_size)

        :param kwargs: MultiHeadAttention 的其它参数
        """
        x = self.transformer_layers(x, **kwargs)

        # 形状为 (batch_size, 1, hidden_size)
        query = self.query.repeat(x.shape[0], 1, 1)

        return self.fusion(query, x, x, **kwargs).squeeze(1)
