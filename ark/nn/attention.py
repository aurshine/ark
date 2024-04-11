import sys

import torch
from torch import nn
from ark.device import use_device
from ark.nn.mask import masked_softmax
from ark.nn.encoder import PositionEncoder


class AdditiveAttention(nn.Module):
    def __init__(self, queries_size, keys_size, hidden_size, dropout=0, device=None):
        super(AdditiveAttention, self).__init__()
        self.device = use_device(device)
        self.linear_queries = nn.Linear(queries_size, hidden_size, bias=False, device=self.device)
        self.linear_keys = nn.Linear(keys_size, hidden_size, bias=False, device=self.device)
        self.linear_score = nn.Linear(hidden_size, 1, bias=False, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_len=None, front_mask=True):
        """
        :param queries: 形状为 (batch_size, num_queries, queries_size)

        :param keys: 形状为 (batch_size, num_keys_values, keys_size)

        :param values: 形状为 (batch_size, num_keys_values, values_size)

        :param valid_len: 形状为(batch_size, ) 掩盖对应 batch 的 key value 对

        :param front_mask: 对于一个 mask_len , 若 front_mask 为 True, 表示掩盖前 mask_len 个值

        :return: 返回形状为 (batch_size, num_queries, values_size)
        """
        queries, keys, values = queries.to(self.device), keys.to(self.device), values.to(self.device)
        queries, keys = self.linear_queries(queries).unsqueeze(2), self.linear_keys(keys).unsqueeze(1)

        # 形状为 (batch_size, num_queries, num_keys_values, hidden_size)
        features = torch.tanh(queries + keys)

        if valid_len is not None:
            assert valid_len.dim() == 1
            valid_len = valid_len.repeat_interleave(features.size(1)).reshape(features.size(0), -1)

        # 形状为 (batch_size, num_queries, num_keys_values)
        attention_score = masked_softmax(self.linear_score(features).squeeze(-1), valid_len, front_mask)

        return torch.bmm(self.dropout(attention_score), values)


class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=1, bidirectional=False, device=None):
        super(LSTMAttention, self).__init__()
        self.device = use_device(device)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer, bidirectional=bidirectional, device=self.device)

        bi = 2 if bidirectional else 1
        self.attention = AdditiveAttention(hidden_size * bi, hidden_size * bi, hidden_size * bi)
        self.position = PositionEncoder(hidden_size * bi)

    def forward(self, X, state=None):
        hidden, state = self.lstm(X, state)
        hidden = self.position(hidden.permute(1, 0, 2))

        y = self.attention(hidden[:, -1, :].squeeze(1), hidden, hidden).unsqueeze(1)

        return y