from torch import nn
from ark.device import use_device


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout, device=None):
        super(AddNorm, self).__init__()
        self.device = use_device(device)
        self.ln = nn.LayerNorm(norm_shape, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        ret = self.ln(self.dropout(y) + x)
        return ret