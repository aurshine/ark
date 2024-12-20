from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import use_device
from .trainer import Trainer
from .encoder import ArkEncoder
from .decoder import ArkDecoder
from .tokenizer import Tokenizer


class Ark(Trainer):
    """
    ark 注意力模型
    """
    def __init__(self,
                 tokenizer: Tokenizer,
                 steps: int,
                 hidden_size: int,
                 num_heads: int,
                 num_layer: int,
                 num_class: int,
                 num_channels: int = 3,
                 dropout: float = 0.5,
                 output_layer=None,
                 device: Union[str, torch.device, None] = None,
                 **kwargs):
        """
        初始化

        :param tokenizer: 预训练的 tokenizer

        :param steps: 最大时间步

        :param hidden_size: 隐藏层大小

        :param num_heads: 注意力头数

        :param num_layer: 层数

        :param num_class: 类别数

        :param num_channels: 输入通道数

        :param dropout: dropout

        :param output_layer: 输出层, 需要接受形状为 (batch_size, steps, hidden_size) 的输入, 默认为None

        :param device: 设备
        """
        super(Ark, self).__init__(num_class, device=device, **kwargs)
        self.tokenizer = tokenizer

        self.encoder = ArkEncoder(vocab_size=len(self.tokenizer),
                                  hidden_size=hidden_size,
                                  steps=steps,
                                  dropout=dropout,
                                  device=self.device
                                  )

        self.decoder = ArkDecoder(hidden_size=hidden_size,
                                  num_heads=num_heads,
                                  num_layer=num_layer,
                                  num_channels=num_channels,
                                  dropout=dropout,
                                  device=self.device
                                  )

        self.output_layer = output_layer

    def forward(self,
                x: Union[torch.Tensor, List[torch.Tensor]],
                masks: Union[torch.Tensor, List[torch.Tensor]] = None,
                **kwargs):
        """
        :param x: 每个tensor的形状为 (batch_size, steps), 如果x为list表示多通道输入

        :param masks: 每个tensor的mask形状为 (batch_size, steps), 如果masks为list表示多通道输入
        """
        x = self._to_device(x)

        if masks is not None:
            masks = self._to_device(masks)
        # y     (batch_size, steps, hidden_size)
        # masks (batch_size, steps)
        y, masks = self.encoder(x, masks, **kwargs)
        # y     (batch_size, steps, hidden_size)
        y = self.decoder(y, memory_key_padding_mask=masks)
        if self.output_layer is not None:
            y = self.output_layer(y, **kwargs)
        return y


class ArkClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, num_heads, dropout=0, device=None):
        super(ArkClassifier, self).__init__()
        self.device = use_device(device)

        self.query = nn.Parameter(torch.empty(size=(1, 1, hidden_size), device=self.device))
        nn.init.xavier_normal_(self.query)
        self.fusion = nn.TransformerDecoderLayer(d_model=hidden_size, 
                                                 nhead=num_heads, 
                                                 dim_feedforward=hidden_size*4, 
                                                 dropout=dropout, 
                                                 batch_first=True, 
                                                 device=self.device)
        self.classifier = nn.Linear(hidden_size, num_classes, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        """
        :param x: x 形状为 (batch_size, steps, hidden_size)

        :param kwargs: TransformerLayer 的其它参数

        :return: (batch_size, num_classes)
        """
        query = self.query.repeat(x.shape[0], 1, 1)
        x = self.fusion(self.dropout(query), x, **kwargs).squeeze(1)
        return self.classifier(x)


class ArkBertPretrain(nn.Module):
    def __init__(self, hidden_size, num_class, device=None):
        super(ArkBertPretrain, self).__init__()
        device = use_device(device)
        self.cls = nn.Linear(hidden_size, num_class, device=device)

    def forward(self, x, masked_position: torch.Tensor, **kwargs):
        """
        ArkBERT 预训练模型

        截取 masked_position 位置的 token 进行预训练

        :param x: 形状为 (batch_size, steps, hidden_size)

        :param masked_position: 形状为(batch_size, num_masked_position)

        :return: 形状为 (batch_size, num_masked_position, num_class)
        """
        y = self.cls(x)
        index = masked_position.unsqueeze(-1).expand(-1, -1, y.shape[-1])
        return torch.gather(y, dim=1, index=index)