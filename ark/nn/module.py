import sys
import torch
from torch import nn
from typing import List
from ark.nn.trainer import Trainer
from ark.nn.encoder import ArkEncoder
from ark.nn.decoder import ArkDecoder
from ark.running import Timer


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
    def __init__(self, vocab, hidden_size, in_channel, num_steps, num_heads, en_num_layer, de_num_layer, dropout, num_class, mini_batch_size=None, device=None):
        super(AttentionArk, self).__init__(num_class, mini_batch_size=mini_batch_size, device=device)
        self.vocab = vocab
        self.encoder = ArkEncoder(vocab, hidden_size, in_channel, num_heads, num_layer=en_num_layer, dropout=dropout,
                                  device=self.device)

        self.decoder = ArkDecoder(hidden_size, num_heads, num_layer=de_num_layer, num_steps=num_steps, dropout=dropout,
                                  device=self.device)

        self.linear = nn.Linear(hidden_size, num_class, device=self.device)

    def forward(self, X, valid_len=None, **kwargs):
        """
        :param X: 形状为 (batch_size, num_channels, steps)

        :param valid_len: 形状为 (batch_size, )

        :return: (batch_size, num_class)
        """
        batch_size = X.shape[0]
        if batch_size <= self.mini_batch_size:
            X = X.to(self.device)

            return self.linear(self.decoder(self.encoder(X, valid_len)))
        else:
            results = []
            for i in range(0, batch_size, self.mini_batch_size):
                x_batch = X[i: min(batch_size, i + self.mini_batch_size)].to(self.device)
                valid_len_batch = valid_len[i: min(batch_size, i + self.mini_batch_size)] if valid_len is not None else None
                results.append(self.linear(self.decoder(self.encoder(x_batch, valid_len_batch))))

            return torch.cat(results, dim=0)

