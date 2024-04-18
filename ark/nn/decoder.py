import torch
from torch import nn
from ark.device import use_device
from ark.nn.addnorm import AddNorm
from ark.nn.multi_layers import PositionWiseFFN, TransformerLayer


class Decoder(nn.Module):
    def __init__(self, device=None):
        super(Decoder, self).__init__()
        self.device = use_device(device)

    def init_state(self, enc_output, *args):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        """:return: hidden, states"""
        raise NotImplementedError


class ArkDecoderBlock(Decoder):
    def __init__(self, hidden_size, num_heads, num_layer=1, dropout=0, device=None):
        super(ArkDecoderBlock, self).__init__(device=device)
        self.transformer_blocks = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout, device=self.device)
                                                 for _ in range(num_layer)])

    def init_state(self, enc_output, *args):
        return None

    def forward(self, X, **kwargs):
        """
        :param X: 形状为 (batch_size, steps, num_hidden)

        :param kwargs: MultiHeadAttention 的其它参数

        :return: 形状为 (batch_size, steps, num_hidden)
        """
        key_value = X
        for block in self.transformer_blocks:
            X = block(X, key_value, key_value, **kwargs)
            key_value = torch.cat([key_value, X], dim=1)

        return X


class ArkDecoder(Decoder):
    def __init__(self, hidden_size, num_heads, num_layer, num_steps, dropout=0, device=None):
        super(ArkDecoder, self).__init__(device)
        self.decoder_block = ArkDecoderBlock(hidden_size, num_heads, num_layer, dropout, device=self.device)
        self.flatten = nn.Flatten()
        self.ffn = PositionWiseFFN(hidden_size * num_steps, hidden_size, hidden_size, dropout, device=self.device)
        self.fusion = TransformerLayer(hidden_size, num_heads, dropout, device=self.device)

    def init_state(self, enc_output, *args):
        return None

    def forward(self, X, **kwargs):
        """
        :param X:  形状为 (batch_size, steps, hidden_size)

        :param kwargs: MultiHeadAttention 的其它参数
        """
        X = X.to(self.device)
        X = self.decoder_block(X, **kwargs)

        # 形状为 (batch_size, 1, hidden_size)
        query = self.ffn(self.flatten(X)).unsqueeze(1)

        return self.fusion(query, X, X, **kwargs).squeeze(1)
