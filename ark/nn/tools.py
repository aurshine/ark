import numpy as np
import pandas as pd
import torch
from torch import nn


class Permute(nn.Module):
    def __init__(self, *permutes, **kwargs):
        super(Permute, self).__init__(**kwargs)
        self.permutes = permutes

    def forward(self, inputs: torch.Tensor):
        return inputs.permute(*self.permutes)


class Squeeze(nn.Module):
    def __init__(self, dim, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.dim = dim

    def forward(self, inputs):
        return torch.squeeze(inputs, self.dim)


class UnSqueeze(nn.Module):
    def __init__(self, dim, **kwargs):
        super(UnSqueeze, self).__init__(**kwargs)
        self.dim = dim

    def forward(self, inputs):
        return torch.unsqueeze(inputs, self.dim)


class ToTensor(nn.Module):
    def __init__(self, **kwargs):
        super(ToTensor, self).__init__(**kwargs)

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return torch.clone(inputs).detach_()
        elif isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs)
        elif isinstance(inputs, (pd.Series, pd.DataFrame)):
            return torch.from_numpy(inputs.to_numpy())
        else:

            return torch.tensor(inputs)
