import torch
from torch import nn
from ark.device import use_device
from ark.nn.multi_layers import TransformerLayers, TransformerLayer


class ArkDecoder(nn.Module):
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


class ArkClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, num_heads, dropout=0, device=None):
        super(ArkClassifier, self).__init__()
        self.device = use_device(device)

        self.query = nn.Parameter(torch.empty(size=(1, 1, hidden_size), device=self.device))
        nn.init.xavier_normal_(self.query)
        self.fusion = TransformerLayer(hidden_size, num_heads, dropout=dropout, device=self.device)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, **kwargs):
        """

        :param x: x 形状为 (batch_size, steps, hidden_size)

        :param kwargs: MultiHeadAttention 的其它参数

        :return: (batch_size, num_classes)
        """
        query = self.query.repeat(x.shape[0], 1, 1)
        x = self.fusion(query, x, x, **kwargs).squeeze(1)
        return self.classifier(x)