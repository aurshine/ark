from copy import deepcopy
from typing import Union, List
import torch
from ark.nn.trainer import Trainer


class Bagging(Trainer):
    def __init__(self, model: Union[Trainer, List[Trainer]], num_class, num_model=1, device=None):
        super(Bagging, self).__init__(num_class, device=device)
        if isinstance(model, Trainer):
            self.models: List[Trainer] = [deepcopy(model).to(self.device) for _ in range(num_model)]
        elif isinstance(model, list):
            self.models: List[Trainer] = [md.to(self.device) for md in model]

    def fit(self, *args, **kwargs):
        for model in self.models:
            model.fit(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return [model.forward(x, *args, **kwargs) for model in self.models]

    def predict(self, x, **kwargs):
        # outputs 的形状为 (batch_size, num_models)
        outputs = torch.stack([model.predict(x, **kwargs) for model in self.models]).permute(1, 0)
        # pred shape = (batch_size, num_models, num_class)
        pred = torch.zeros(size=(*list(outputs.shape), self.num_class))

        # pred[i][j][outputs[i][j]] = 1
        pred.scatter_(2, outputs.unsqueeze(2).to(torch.int64), 1)

        return torch.argmax(pred.sum(dim=1), dim=-1)

    def __getitem__(self, index):
        return self.models[index]

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)