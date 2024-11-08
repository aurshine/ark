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

        def transformer_layer():
            return TransformerLayer(hidden_size, num_heads, dropout=dropout, device=self.device)

        self.transformer_layers1 = nn.ModuleList([transformer_layer() for _ in range(num_layer)])
        self.transformer_layers2 = nn.ModuleList([transformer_layer() for _ in range(num_layer)])
        self.transformer_layers3 = nn.ModuleList([transformer_layer() for _ in range(num_layer)])

        self.dim_reduce = nn.Linear(hidden_size * 3, hidden_size, device=self.device)

    def forward(self, x, **kwargs):
        """
        :param x: 形状为 (num_channels, batch_size, steps, hidden_size)

        :param kwargs: TransformerLayers 的其它参数
        """
        x1, x2, x3 = x
        for layer1, layer2, layer3 in zip(self.transformer_layers1, self.transformer_layers2, self.transformer_layers3):
            flatten_x = torch.cat([x1, x2, x3], dim=-2)
            y1 = layer1(x1, flatten_x, **kwargs)
            y2 = layer2(x2, flatten_x, **kwargs)
            y3 = layer3(x3, flatten_x, **kwargs)
            x1, x2, x3 = y1, y2, y3

        return self.dim_reduce(torch.cat([x1, x2, x3], dim=-1))