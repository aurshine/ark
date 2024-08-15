import sys
import torch
from torch import nn
from typing import List
from ark.nn.trainer import Trainer
from ark.nn.encoder import ArkEncoder
from ark.nn.decoder import ArkDecoder


def analyse(model: Trainer, inputs, classes: List[str]):
    """
    :param model: 用于分析的模型

    :param inputs: 输入

    :param classes: 类别

    :return: 每个输入的预测结果
    """
    return [classes[index] for index in model.predict(inputs)]


class AttentionArk(Trainer):
    """
    ark 注意力模型
    """
    def __init__(self, vocab, steps, hidden_size, in_channel, num_heads, num_layer, num_class, dropout=0.5, mini_batch_size=None, device=None):
        super(AttentionArk, self).__init__(num_class, mini_batch_size=mini_batch_size, device=device)
        self.vocab = vocab
        self.encoder = ArkEncoder(vocab,
                                  hidden_size,
                                  in_channel,
                                  steps=steps,
                                  dropout=dropout,
                                  device=self.device
                                  )

        self.decoder = ArkDecoder(hidden_size,
                                  num_heads,
                                  num_layer=num_layer,
                                  dropout=dropout,
                                  device=self.device
                                  )

        self.linear = nn.Linear(hidden_size, num_class, device=self.device)

    def forward(self, x, valid_len=None, **kwargs):
        """
        :param x: 形状为 (batch_size, num_channels, steps)

        :param valid_len: 形状为 (batch_size, )

        :return: (batch_size, num_class)
        """
        return self.linear(self.decoder(self.encoder(x, valid_len)))
