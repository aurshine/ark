import os
from typing import List, Union

import matplotlib.pyplot as plt


def save_fig(path=None):
    if path is None:
        idx = 0
        while os.path.exists(f'{idx}.png'):
            idx += 1
        path = f'{idx}.png'

    plt.savefig(path)
    plt.clf()


class Plot:
    def __init__(self, n_linear):
        self.points = [[] for _ in range(n_linear)]
        self.num = 0

    def add(self, *point: Union[int, float, List[Union[int, float]]]):
        self.num += 1
        for i, p in enumerate(point):
            if isinstance(p, (int, float)):
                self.points[i].append(p)
            elif isinstance(p, list):
                self.points[i] += p
            else:
                print(f'p type {type(p)}')
                raise TypeError

    def plot(self, x_label='', y_label='', labels: Union[List[str], str] = None, title=None, save_path=None):
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

        if save_path is not None:
            save_fig(save_path)
