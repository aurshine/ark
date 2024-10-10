import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer

from ark.nn.pinyin import translate_piny, Style


class InitialFinalLoss(nn.CrossEntropyLoss):
    """
    声母韵母相似度损失的计算方法为:

    1. 相似度 = 1 if token相同 else (self.initial_weight * 声母相同 + self.final_weight * 韵母相同)

    2. pred_score = max(softmax(y))

    3. 声母韵母相似度损失 = cross_entropy(y) - log(pred_score) * (1 - 相似度)
    """
    def __init__(self, tokenizer: BertTokenizer, initial_weight=0.35, final_weight=0.35, reduction='mean', **kwargs):
        super(InitialFinalLoss, self).__init__(reduction=reduction, **kwargs)
        self.tokenizer = tokenizer
        self.initial_weight = initial_weight
        self.final_weight = final_weight

    def forward(self, y_hat, y):
        """
        :param y_hat: (batch_size, num_prediction, num_classes)

        :param y: (batch_size, num_prediction)
        """
        loss1 = super(InitialFinalLoss, self).forward(y_hat, y)
        # (batch_size, num_prediction)
        pred = y_hat.argmax(dim=-1)

        y_hat_decodes = [self.tokenizer.convert_ids_to_tokens(pred_, skip_special_tokens=True) for pred_ in pred]
        y_decodes = [self.tokenizer.convert_ids_to_tokens(y_, skip_special_tokens=True) for y_ in y]

        # pred 哪些位置的 token 与 y 相似的
        token_similarities = torch.zeros_like(y, dtype=torch.float64)
        for i, (y_hat_str, y_str) in enumerate(zip(y_hat_decodes, y_decodes)):
            for j, (y_hat_token, y_token) in enumerate(zip(y_hat_str, y_str)):
                if y_hat_token == y_token:
                    token_similarities[i, j] = 1

                if len(y_hat_token) == len(y_token):
                    # 声母相同
                    if translate_piny(y_hat_token, Style.INITIALS) == translate_piny(y_token, Style.INITIALS):
                        token_similarities[i, j] += self.initial_weight
                    # 韵母相同
                    if translate_piny(y_hat_token, Style.FINALS) == translate_piny(y_token, Style.FINALS):
                        token_similarities[i, j] += self.final_weight

        pred_score, indices = torch.max(F.softmax(y_hat, dim=-1), dim=-1)
        loss2 = torch.log(pred_score) * (1 - token_similarities)

        if self.reduction == 'mean':
            loss2 = loss2.mean()
        elif self.reduction == 'sum':
            loss2 = loss2.sum()

        return loss1 - loss2