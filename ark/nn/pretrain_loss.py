import torch
from torch import nn
from torch.nn import functional as F

from .pinyin import translate_piny, Style
from .tokenizer import Tokenizer


class InitialFinalLoss(nn.CrossEntropyLoss):
    """
    声母韵母相似度损失的计算方法为:

    1. 相似度 = 1 if token相同 else (self.initial_weight * 声母相同 + self.final_weight * 韵母相同)

    2. pred_score = max(softmax(y))

    3. 声母韵母相似度损失 = cross_entropy(y) + (-LOG(true_score) * 相似度 - LOG(pred_score) * (1 - 相似度))
    """
    def __init__(self, tokenizer: Tokenizer, initial_weight=0.35, final_weight=0.35, reduction='mean', **kwargs):
        super(InitialFinalLoss, self).__init__(reduction=reduction, **kwargs)
        self.tokenizer = tokenizer
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.epsilon = 1e-8

    def forward(self, y_hat, y):
        """
        :param y_hat: (batch_size, num_prediction, num_classes)

        :param y: (batch_size, num_prediction)
        """
        # (batch_size * num_prediction, num_classes), (batch_size * num_prediction)
        y_hat, y = y_hat.view(-1, y_hat.size(-1)), y.view(-1)

        loss1 = super(InitialFinalLoss, self).forward(y_hat, y)
        # (batch_size * num_prediction)
        pred = y_hat.argmax(dim=-1)

        y_hat_decodes = self.tokenizer.ids_to_tokens(pred.tolist())
        y_decodes = self.tokenizer.ids_to_tokens(y.tolist())

        # pred 哪些位置的 token 与 y 相似的 (batch_size * num_prediction)
        token_similarities = torch.zeros_like(y, dtype=torch.float64, device=y.device)
        for i, (y_hat_token, y_token) in enumerate(zip(y_hat_decodes, y_decodes)):
            if y_hat_token == y_token:
                token_similarities[i] = 1

            if len(y_hat_token) == len(y_token):
                # 声母相同
                if translate_piny(y_hat_token, Style.INITIALS) == translate_piny(y_token, Style.INITIALS):
                    token_similarities[i] += self.initial_weight
                # 韵母相同
                if translate_piny(y_hat_token, Style.FINALS) == translate_piny(y_token, Style.FINALS):
                    token_similarities[i] += self.final_weight

        # (batch_size * num_prediction, num_classes)
        y_hat_softmax = F.softmax(y_hat, dim=-1)
        # y_hat每个样本在预测标签上的得分 (batch_size * num_prediction)
        pred_score = torch.gather(y_hat_softmax, dim=-1, index=pred.unsqueeze(-1)).unsqueezed(-1)
        # y_hat每个样本在真实标签上的得分 (batch_size * num_prediction)
        true_score = torch.gather(y_hat_softmax, dim=-1, index=y.unsqueeze(-1)).unsqueezed(-1)

        # 相似度越大, loss越小, 所以1 - token_similarities
        loss2 = -torch.log(pred_score + self.epsilon) * token_similarities - torch.log(true_score + self.epsilon) * (1 - token_similarities)

        if self.reduction == 'mean':
            loss2 = loss2.mean()
        elif self.reduction == 'sum':
            loss2 = loss2.sum()

        return loss1 + loss2