import os
from threading import Lock

from typing import List, Tuple, Union, Generator

from ark.spider.classify import write_lines, get_lines, sort_unique


class PermuteString:
    """
    记录一个base_string 以及其替换组
    """
    def __init__(self, base_string: str = None, replaces: List[Tuple[str, str]] = None):
        """
        初始化一个PermuteString对象

        :param base_string: 原始字符串

        :param replaces: 字符串替换组
        """
        if base_string is None:
            base_string = ''

        if replaces is None:
            replaces = []

        self.base_string = base_string
        self.replaces = replaces

    def __repr__(self):
        return self.base_string

    def __str__(self):
        return self.base_string

    def __getitem__(self, permute_indices: int) -> str:
        """
        得到指定排列组合的字符串

        :param permute_indices: 排列组合的索引 可以为负数，表示倒序排列

        :return: 指定排列组合的字符串
        """
        assert isinstance(permute_indices, int), "permute_indices must be an integer"

        n = len(self.replaces)
        s = self.base_string

        # 索引映射到 [0, 2^n-1]
        permute_indices = (permute_indices % (1 << n) + (1 << n)) % (1 << n)
        for i in range(n):
            if permute_indices >> i & 1 == 1:
                s = s.replace(self.replaces[i][0], self.replaces[i][1])
        return s

    def append(self, _old: str, _new: str):
        """
        添加一个替换组
        """
        self.replaces.append((_old, _new))

    def transform(self) -> List[str]:
        """
        返回所有组合情况
        """
        n = len(self.replaces)

        # n 个替换组有 2^n 种排列组合方法
        return [self.__getitem__(i) for i in range(0, 1 << n)]


class Comment:
    """
    用于存储PermuteString的容器, 本质是一个 list 结构
    """

    # Comment 对象做download操作时的锁
    __comment_lock__ = Lock()

    def __init__(self, string_list: Union[List[PermuteString], List[str]] = None):
        if string_list is None:
            string_list = []

        assert isinstance(string_list, list), "permute_string_list must be a list"

        # list[str] -> list[PermuteString]
        if len(string_list) > 0 and isinstance(string_list[0], str):
            string_list = [PermuteString(permute_string) for permute_string in string_list]

        self.comments = string_list

    def append(self, comment: Union[PermuteString, str]):
        if isinstance(comment, PermuteString):
            self.comments.append(comment)
        elif isinstance(comment, str):
            self.comments.append(PermuteString(comment))

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        return self.comments[item]

    def __add__(self, other):
        if isinstance(other, Comment):
            return Comment(self.comments + other.comments)
        else:
            raise TypeError(f'类型Comment 无法与{type(other)}做 + 运算')

    def __iadd__(self, other):
        if isinstance(other, Comment):
            self.comments += other.comments
        else:
            raise TypeError(f'类型Comment 无法与{type(other)}做 += 运算')

        return self

    def __iter__(self):
        return iter(self.comments)

    def tolist(self) -> List[PermuteString]:
        """
        返回所有PermuteString
        """
        return self.comments

    def download(self, path, mode, encoding=None):
        """
        将所有PermuteString的排列结果写入文件

        写入时会排序去重

        :param path: 文件路径

        :param mode: 文件打开模式,'w' or 'a'

        :param encoding: 文件编码，默认utf-8
        """
        assert mode in ['w', 'a'], "mode must be 'w' or 'a'"
        if self.__len__() == 0:
            return

        if encoding is None:
            encoding = 'utf-8'

        comments = list(permutes(self.comments))
        with Comment.__comment_lock__:
            if mode == 'a' and os.path.exists(path):
                comments += get_lines(path=path, encoding=encoding)

            write_lines(sort_unique(comments), path=path, encoding=encoding, mode='w')


def permutes(permute_strings: List[PermuteString]) -> Generator[str, None, None]:
    """
    将所有 PermuteString 的排列结果用一维的 list 记录
    """
    for permute_string in permute_strings:
        assert isinstance(permute_string, PermuteString), "permute_strings must be a list of PermuteString"
        for tran_s in permute_string.transform():
            yield tran_s