import torch
from torch import nn
from ark.device import use_device
from ark.nn.multi_layers import TransformerLayers, TransformerLayer


class ArkDecoder(nn.Module):
    """
    解码器，由多个 TransformerLayer 组成

    解码结果
    """
    def __init__(self, hidden_size, num_heads, num_layer, dropout=0, device=None):
        super(ArkDecoder, self).__init__()
        self.device = use_device(device)
        self.transformer_layers = TransformerLayers(hidden_size, num_heads, num_layer, hidden_size=hidden_size, dropout=dropout, device=self.device)

    def forward(self, x, **kwargs):
        """
        :param x:  形状为 (batch_size, steps, hidden_size)

        :param kwargs: MultiHeadAttention 的其它参数
        """
        return self.transformer_layers(x, **kwargs)


