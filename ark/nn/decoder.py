import torch
from torch import nn
from ark.utils import use_device
from ark.nn.multi_layers import TransformerLayer


class ArkDecoder(nn.Module):
    """
    解码器，由多个 TransformerLayer 组成

    解码结果
    """
    def __init__(self, hidden_size: int, num_heads: int, num_layer: int, dropout: float = 0.5, device=None):
        super(ArkDecoder, self).__init__()
        self.device = use_device(device)
        self.transformer123 = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout=dropout, device=self.device)
                                            for _ in range(num_layer)])
        self.transformer132 = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout=dropout, device=self.device)
                                            for _ in range(num_layer)])
        self.transformer231 = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout=dropout, device=self.device)
                                            for _ in range(num_layer)])
        self.concat = nn.Linear(hidden_size * 3, hidden_size, device=self.device)

    def forward(self, x, **kwargs):
        """
        :param x: 形状为 (num_channels, batch_size, steps, hidden_size)

        :param kwargs: TransformerLayers 的其它参数
        """
        x1, x2, x3 = x
        for layer123, layer132, layer231 in zip(self.transformer123, self.transformer132, self.transformer231):
            y3 = layer123(x1, x2, x3, **kwargs)
            y2 = layer132(x1, x3, x2, **kwargs)
            y1 = layer231(x2, x3, x1, **kwargs)

            x1, x2, x3 = y1, y2, y3

        return self.concat(torch.cat([x1, x2, x3], dim=-1))