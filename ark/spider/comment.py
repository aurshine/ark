import os.path

from typing import List
from ark.setting import UN_CLASSIFY_PATH
from ark.spider.classify import writeLines, getLines, sortUnique


class PermuteString:
    """
    记录一个base_string 以及其替换组
    """
    def __init__(self, base_string=None, replaces=None):
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

    def __getitem__(self, permute_indices):
        n = len(self.replaces)
        s = self.base_string

        permute_indices = (permute_indices % (1 << n) + (1 << n)) % (1 << n)
        for i in range(n):
            if permute_indices >> i & 1:
                s = s.replace(self.replaces[i][0], self.replaces[i][1])
        return s

    def append(self, _old, _new):
        """
        添加一个替换组
        """
        self.replaces.append((_old, _new))

    def transform(self):
        """
        返回所有组合情况
        """
        n = len(self.replaces)

        # n 个替换组有 2^n 种排列组合方法
        return [self.__getitem__(i) for i in range(0, 1 << n)]


class Comment:
    """
    用于存储PermuteString的容器
    """
    def __init__(self, permute_string_list=None):
        if permute_string_list is None:
            permute_string_list = []

        # list[str] -> list[PermuteString]
        if len(permute_string_list) > 0 and isinstance(permute_string_list[0], str):
            permute_string_list = [PermuteString(permute_string) for permute_string in permute_string_list]

        self.comments = permute_string_list

    def append(self, comment):
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

    def tolist(self):
        return self.comments

    def download(self, path=UN_CLASSIFY_PATH, encoding='utf-8', mode='a'):
        comments = permutes(self.comments)
        if mode == 'a' and os.path.exists(path):
            comments += getLines(path=path, encoding=encoding)

        writeLines(sortUnique(comments), path=path, encoding=encoding, mode='w', info=True)


def permutes(permute_strings: List[PermuteString]):
    """将所有PermuteString的排列结果用一维的list记录"""
    return [s for permute_string in permute_strings for s in permute_string.transform()]