import torch
from torch import nn
from ark.utils import use_device
from ark.nn.multi_layers import ChannelWiseTransformerLayer


class ArkDecoder(nn.Module):
    """
    解码器，由多个 TransformerLayer 组成

    解码结果
    """
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_layer: int,
                 num_channels: int,
                 dropout: float = 0.5,
                 device=None):
        super(ArkDecoder, self).__init__()
        self.device = use_device(device)

        def cwt_layer():
            return ChannelWiseTransformerLayer(hidden_size,
                                               num_heads,
                                               num_channels,
                                               dropout=dropout,
                                               device=self.device)

        self.channel_wise_transformer_layers = nn.ModuleList([cwt_layer() for _ in range(num_layer)])

        self.dim_reduce = nn.Linear(hidden_size * num_channels, hidden_size, device=self.device)

    def forward(self, x, **kwargs):
        """
        :param x: 形状为 (num_channels, batch_size, steps, hidden_size)

        :param kwargs: TransformerLayers 的其它参数

        :return: 解码结果，形状为 (batch_size, steps, hidden_size)
        """
        for cwt_layer in self.channel_wise_transformer_layers:
            # (num_channels, batch_size, steps, hidden_size)
            x = cwt_layer(x, **kwargs)
        
        # (batch_size, steps, num_channels * hidden_size)
        x = torch.reshape(
            # (batch_size, steps, num_channels, hiddem_size)
            torch.permute(x, (1, 2, 0, 3)),
            (x.size(1), x.size(2), -1)
        )

        # (batch_size, steps, hidden_size)
        y = self.dim_reduce(x)
        return y