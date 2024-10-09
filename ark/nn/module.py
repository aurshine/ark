from typing import List, Union

import torch
from torch import nn

from ark.nn.trainer import Trainer
from ark.nn.encoder import ArkEncoder
from ark.nn.decoder import ArkDecoder
from ark.nn.multi_layers import TransformerLayer
from ark.device import use_device


def analyse(model: Trainer, inputs, classes: List[str]):
    """
    :param model: 用于分析的模型

    :param inputs: 输入

    :param classes: 类别

    :return: 每个输入的预测结果
    """
    return [classes[index] for index in model.predict(inputs)]


class Ark(Trainer):
    """
    ark 注意力模型
    """
    def __init__(self,
                 tokenizer,
                 steps: int,
                 hidden_size: int,
                 in_channel: int,
                 num_heads: int,
                 num_layer: int,
                 num_class: int,
                 dropout: float = 0.5,
                 output_layer=None,
                 device: Union[str, torch.device, None] = None):
        """
        初始化

        :param tokenizer: 预训练的 tokenizer

        :param steps: 最大时间步

        :param hidden_size: 隐藏层大小

        :param in_channel: 输入通道数

        :param num_heads: 注意力头数

        :param num_layer: 层数

        :param num_class: 类别数

        :param dropout: dropout

        :param output_layer: 输出层, 需要接受形状为 (batch_size, steps, hidden_size) 的输入, 默认为None

        :param device: 设备
        """
        super(Ark, self).__init__(num_class, device=device)
        self.tokenizer = tokenizer

        self.encoder = ArkEncoder(vocab_size=len(self.tokenizer.get_vocab()),
                                  hidden_size=hidden_size,
                                  num_channel=in_channel,
                                  steps=steps,
                                  dropout=dropout,
                                  device=self.device
                                  )

        self.decoder = ArkDecoder(hidden_size=hidden_size,
                                  num_heads=num_heads,
                                  num_layer=num_layer,
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
        y = self.decoder(y, masks=masks, **kwargs)

        if self.output_layer is not None:
            y = self.output_layer(y, **kwargs)
        return y

    def decode_ids(self, y: Union[List[int], List[List[int]], torch.Tensor]) -> List[str]:
        """
        将id序列解码为字符串序列

        :param y: 输出的tensor

        :return: 每个输入的预测结果
        """
        return self.tokenizer.decode(y, clean_up_tokenization_spaces=True)


class ArkClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, num_heads, dropout=0, device=None):
        super(ArkClassifier, self).__init__()
        self.device = use_device(device)

        self.query = nn.Parameter(torch.empty(size=(1, 1, hidden_size), device=self.device))
        nn.init.xavier_normal_(self.query)
        self.fusion = TransformerLayer(hidden_size, num_heads, dropout=dropout, device=self.device)
        self.classifier = nn.Linear(hidden_size, num_classes, device=self.device)

    def forward(self, x, **kwargs):
        """

        :param x: x 形状为 (batch_size, steps, hidden_size)

        :param kwargs: TransformerLayer 的其它参数

        :return: (batch_size, num_classes)
        """
        query = self.query.repeat(x.shape[0], 1, 1)
        x = self.fusion(query, x, x, **kwargs).squeeze(1)
        return self.classifier(x)


class ArkBertPretrain(nn.Module):
    def __init__(self):
        super(ArkBertPretrain, self).__init__()

    def forward(self, x, masked_position: torch.Tensor, **kwargs):
        """
        ArkBERT 预训练模型

        截取 masked_position 位置的 token 进行预训练

        :param x: 形状为 (batch_size, steps, hidden_size)

        :param masked_position: 形状为(batch_size, num_masked_position)

        :return: 形状为 (batch_size, num_masked_position, hidden_size)
        """
        index = masked_position.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.gather(x, dim=1, index=index)