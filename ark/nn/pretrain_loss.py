import torch
from torch import nn
from transformers import BertTokenizer


class InitialFinalLoss(nn.CrossEntropyLoss):
    def __init__(self, tokenizer: BertTokenizer):
        super(InitialFinalLoss, self).__init__()
        self.tokenizer = tokenizer

    def forward(self, y_hat, y):
        """
        :param y_hat: (batch_size, num_prediction, num_classes)

        :param y: (batch_size, num_prediction)
        """
        loss1 = super(InitialFinalLoss, self).forward(y_hat, y)
        pred = y_hat.argmax(dim=-1)

        y_hat_decode = self.tokenizer.decode(pred, skip_special_tokens=True)
        y_decode = self.tokenizer.decode(y, skip_special_tokens=True)

        same_like = torch.zeros_like(y)