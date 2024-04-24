import os
from threading import Lock
from typing import Iterable

from typing import List
from ark.setting import URLS, HAS_URLS_PATH


__io_lock__ = Lock()


def get_lines(path, encoding='utf-8') -> List[str]:
    """
    传入一个文件地址， 返回文件的每一行组成的列表, 会自动去除首尾空白符

    与 write_lines 共享锁

    :param path: 文件路径

    :param encoding: 编码格式
    """
    with __io_lock__:
        lines = []
        with open(path, encoding=encoding, mode='r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line.isspace():
                    lines.append(line)

    return lines


def write_lines(lines: Iterable, path, mode, encoding=None):
    """
     线程安全的多行写入, 自动去除空白符

     与 get_lines 共享锁

    :param lines: 待写入的可迭代对象

    :param path: 写入文件的路径

    :param mode: 写入模式, 当模式为追加写入且文件不存在时, 则自动切换为覆盖写入

    :param encoding: 编码格式
    """
    if encoding is None:
        encoding = 'utf-8'

    if mode == 'a' and not os.path.exists(path):
        mode = 'w'

    with __io_lock__:
        cnt = 0
        with open(path, encoding=encoding, mode=mode) as f:
            for line in lines:
                cnt += 1
                f.write(line + '\n')

        print(f'写入 {cnt} 行文本到 {path}\n'
              f'编码模式: {encoding}\n'
              f'写入模式: {mode}\n')


def sort_unique(lines: Iterable, key=None, reverse=False) -> list:
    """
    排序去重, 返回排序去重后的列表

    :param lines: 待排序去重地可迭代对象

    :param key: 排序的键

    :param reverse: 排序的顺序
    """
    return list(set(sorted(lines, key=key, reverse=reverse)))


def clear(path, encoding='utf-8'):
    """
    清空文件内容
    """
    write_lines([], path=path, mode='w', encoding=encoding)


def add_url(url: str):
    """
    添加新的已经爬取的url，并写入到文件中
    """
    url = str(url).strip()
    if url not in URLS:
        URLS.add(url)
        write_lines([url], HAS_URLS_PATH, mode='a', encoding='utf-8')


def is_exist(url: str):
    """
    判断url是否已经存在
    """
    url = url.strip()
    return url in URLS