import sys
from torch import nn
from typing import List
from ark.device import use_device
from ark.nn.trainer import Trainer
from ark.nn.attention import AdditiveAttention
from ark.nn.encoder import LstmEncoder, PositionEncoder, ArkEncoder
from ark.nn.decoder import LstmDecoder, ArkDecoder
from ark.nn.encoder_decoder import EncoderDecoder
from ark.nn.addnorm import AddNorm


def analyse(model: Trainer, inputs, classes: List[str]):
    """

    :param model: 用于分析的模型

    :param inputs: 输入

    :param classes: 类别

    :return: 每个输入的预测结果
    """
    return [classes[index] for index in model.predict(inputs)]


class SentimentRNN(Trainer):
    """情感分析循环神经网络"""
    def __init__(self, input_layer, rnn_layer, output_layer, num_class=-1, device=None, **kwargs):
        """
        :param input_layer: 输入层, 传入 inputs

        :param rnn_layer: 工作层, 输入为 inputs_layer的输出 和 state, 在此类为 rnn 层, rnn层的输出需要是 Tuple[outputs, state]
                          rnn_layer每层的输入需要接收 inputs 和 state 两个参数, 输出需要返回 hiddens 和 state 两个参数

        :param output_layer: 输出层, 输入为 rnn_layer的输出

        :param num_class: 分类数量, 如果最后 输出的数量 > 分类数量 则在预测的时候只取 前num_class作为预测

        :param device: 在哪个设备上运行, 默认为None, 优先选择 cuda0 和 cpu
        """
        super(SentimentRNN, self).__init__(num_class, device)
        self.input_layer, self.rnn_layer, self.output_layer = input_layer, rnn_layer, output_layer
        self.num_class = num_class
        self.device = use_device() if device is None else device
        self.to(self.device)

    def forward(self, inputs, state=None, **kwargs):
        """前向传播

        :param inputs: 输入值

        :param state: 当前状态, 默认使用类内的初始化方法

        :returns: 返回输出结果 y 和 rnn_layer返回的 state
        """
        inputs = inputs.to(self.device)

        inputs = self.input_layer(inputs)

        # hiddens shape = (step, batch_size, num_hidden)
        hiddens, state = self.rnn_layer(inputs, state)

        hidden_size = hiddens.shape[-1]
        attention_weight = AdditiveAttention(hidden_size, hidden_size, hidden_size, dropout=0.5)
        queries, keys_values = hiddens[-1].unsqueeze(1), hiddens[: -1].permute(1, 0, 2)
        hiddens = attention_weight(queries, keys_values, keys_values).squeeze(1)

        # y shape = (batch_size, num_output)
        y = self.output_layer(hiddens)

        return y


class SentimentEncoderDecoder(Trainer):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_class,
                 input_layer=None,
                 output_layer=None,
                 num_layer=1,
                 lstm_dropout=0,
                 dropout=0,
                 bidirectional=False,
                 device=None):
        super(SentimentEncoderDecoder, self).__init__(num_class, device)
        self.input_layer = input_layer
        self.encoder_decoder = EncoderDecoder(encoder=LstmEncoder(embedding_size, hidden_size, num_layer,
                                                                  lstm_dropout, bidirectional, device=self.device),
                                              decoder=LstmDecoder(embedding_size, hidden_size, num_layer,
                                                                  lstm_dropout, bidirectional, device=self.device),)
        self.output_layer = output_layer
        self.direction = 2 if bidirectional else 1

        self.attention = AdditiveAttention(hidden_size * self.direction,
                                           hidden_size * self.direction,
                                           hidden_size,
                                           dropout,
                                           device=self.device)
        self.position_encoding = PositionEncoder(hidden_size * self.direction, dropout=dropout)
        self.an = AddNorm(hidden_size * self.direction, dropout, device=self.device)

    def forward(self, inputs, valid_len=None, *args, **kwargs):
        """
        输入为两个文本通道

        第一个通道作为encoder输入

        第二个通道作为decoder输入

        :param inputs: 形状为 (batch_size, channels=2, steps)

        :param valid_len: 形状为 (batch_size, )
        """
        inputs = inputs.to(self.device)
        if self.input_layer is not None:
            inputs = self.input_layer(inputs)
            # inputs 形状为 (channels=2, steps, batch_size, embedding_size)
            inputs = inputs.permute(1, 2, 0, 3)

        # hidden 形状为 (steps, batch_size, hidden_size * bidirectional)
        # state 形状为 (num_state, num_layer * bidirectional, batch_size, hidden_size)
        hidden, _ = self.encoder_decoder(inputs[0], inputs[-1])

        # hidden 形状为 (batch_size, steps, hidden_size * bidirectional)
        hidden = self.position_encoding(hidden.permute(1, 0, 2))

        y = self.an(hidden[:, -1:, :], self.attention(hidden[:, -1:, :], hidden, hidden, valid_len=valid_len)).squeeze(1)

        if self.output_layer is not None:
            y = self.output_layer(y)

        return y


class AttentionArk(Trainer):
    def __init__(self, vocab, hidden_size, in_channel, num_steps, num_heads, en_num_layer, de_num_layer, dropout, num_class, device=None):
        super(AttentionArk, self).__init__(num_class, device)

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
        X = X.to(self.device)

        return self.linear(self.decoder(self.encoder(X, valid_len)))