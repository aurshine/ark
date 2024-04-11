import os
import numpy as np
import torch
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def clac_accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    传入两个一维tensor, 计算准确率
    """
    assert len(y_hat.shape) == 1 and len(y.shape) == 1

    return torch.sum(y_hat == y).item() / y_hat.shape[-1]


def save_fig(path=None):
    if path is None:
        idx = 0
        while os.path.exists(f'{idx}.png'):
            idx += 1
        path = f'{idx}.png'

    plt.savefig(path)
    plt.clf()


class AccuracyCell:
    """
    对单个预测值与实际值做准确率计算
    """

    def __init__(self, num_class, y_hat: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None):
        self.num_class = num_class
        self._score = None
        self.class_matrix = np.zeros(shape=(num_class, num_class), dtype=np.int32)
        if y_hat is not None and y is not None:
            self.__call__(y_hat, y)

    def clear(self):
        self._score = None
        self.class_matrix.fill(0)

    def __getitem__(self, index=None) -> Tuple[float, float, float]:
        """
        取得某类别的 (分类正确数, 该类的总数, 准确率)

        :param index: 第index类别

        :return: (分类正确数, 该类的总数, 准确率)
        """
        r, t = self.right(index), self.total(index)
        rate = (r / t) if t != 0 else 0
        return r, t, rate

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor):
        """y 和 y_hat 必须是一维的tensor

        清空之前保存的值, 计算当前传入的准确率

        :return: 返回自身
        """
        assert len(y_hat.shape) == 1 and len(y.shape) == 1
        self.clear()

        y_hat, y = y_hat.detach(), y.detach()
        for y_pred, y_true in zip(y_hat, y):
            self.class_matrix[y_true, y_pred] += 1

        return self

    @property
    def score(self):
        if self._score is None:
            self._score = self.__getitem__()[2]

        return self._score

    def right(self, index=None) -> float:
        """某一类的正确数, index=None 表示所有类"""
        if index is None:
            indices = list(range(self.num_class))
            return float(np.sum(self.class_matrix[indices, indices]))
        else:
            return float(self.class_matrix[index, index])

    def total(self, index=None) -> float:
        """某一类真实的数量, index=None 表示所有类"""
        if index is None:
            return float(np.sum(self.class_matrix))
        else:
            return float(np.sum(self.class_matrix[index]))

    def confusion_matrix(self, label):
        cmd = ConfusionMatrixDisplay(self.class_matrix, display_labels=label)
        cmd.plot()
        plt.show()
        plt.savefig('confusion_matrix.png')

    def __len__(self):
        return self.num_class

    def __repr__(self):
        ret = 'AccuracyCell: \n'
        for i in range(self.num_class):
            r, t, rate = self.__getitem__(i)
            ret += f'ACCURACY SCORE({i}): {r} / {t} = {r / t if t != 0 else 0.0: .4}\n'
        ret += f'ACCURACY TOTAL SCORE: {self.score}\n'
        return ret

    def __add__(self, other: "AccuracyCell"):
        assert self.num_class == other.num_class

        ret = AccuracyCell(self.num_class)
        ret.class_matrix = self.class_matrix + other.class_matrix

        return ret

    def __iadd__(self, other: "AccuracyCell"):
        assert self.num_class == other.num_class

        self.class_matrix += other.class_matrix
        self._score = None
        return self


class Accuracy:
    """记录准确率的对象

    通过add方法添加预测值和真实值
    """

    def __init__(self, num_class):
        """
        :param num_class: 分类数量
        """
        self.num_class = num_class
        self.accuracy_cells: List[AccuracyCell] = []
        self.max_score_index = None

    def add(self, y_hat: torch.Tensor, y: torch.Tensor):
        """增加 y_hat 和 y, 需要是一维tensor

        :param y_hat: 预测值

        :param y: 真实值
        """
        self.add_cell(AccuracyCell(self.num_class, y_hat, y))

    def add_cell(self, cell: AccuracyCell):
        self.accuracy_cells.append(cell)
        if self.max() is None or cell.score > self.max().score:
            self.max_score_index = len(self.accuracy_cells) - 1

    def max(self) -> Optional[AccuracyCell]:
        """
        :return: 返回准确率最大的 AccuracyCell 或 None
        """
        if self.max_score_index is None:
            return None
        return self.accuracy_cells[self.max_score_index]

    def max_index(self) -> int:
        """
        :return: 返回准确率最高的 AccuracyCell 对象的下标
        """
        return self.max_score_index

    def avg_score(self) -> float:
        score_sum = 0
        for cell in self.accuracy_cells:
            score_sum += cell.score

        return score_sum / len(self.accuracy_cells)

    def plot(self, x_label='', y_label='', labels=None, title=None, save=False):
        """绘制每个accuracy"""
        _plot = Plot(1)
        for cell in self.accuracy_cells:
            _plot.add(cell.score)
        _plot.plot(x_label, y_label, labels, title, save_path=None, save=False)

        if save:
            save_fig()

    def __getitem__(self, index):
        return self.accuracy_cells[index]

    def __len__(self):
        return len(self.accuracy_cells)

    def __repr__(self):
        ret = 'Accuracy: \n'

        for i, cell in enumerate(self.accuracy_cells):
            ret += f'cell {i}: score = {cell.score: .4}\n'

        ret += f'max score = {self.max().score: .4} in cell {self.max_score_index}\n'
        ret += f'avg score = {self.avg_score(): .4}\n'
        return ret


class Plot:
    def __init__(self, n_linear):
        self.points = [[] for _ in range(n_linear)]
        self.num = 0

    def add(self, *point):
        self.num += 1
        for i, p in enumerate(point):
            if isinstance(p, (int, float)):
                self.points[i].append(p)
            elif isinstance(p, list):
                self.points[i] += p
            else:
                print(f'p type {type(p)}')
                raise TypeError

    def plot(self, x_label='', y_label='', labels=None, title=None, save_path=None, save=True):
        # plt.rcParams["font.sans-serif"] = ["SimHei"]
        if labels is None:
            labels = [None] * self.num

        for axis_y, label in zip(self.points, labels):
            plt.plot(axis_y, label=label)

            plt.title(title)
            plt.legend()
            plt.grid()
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            if save:
                save_fig(save_path)