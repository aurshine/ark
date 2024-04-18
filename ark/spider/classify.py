from typing import List
from tqdm import tqdm
from ark.setting import *


def getLines(path=UN_CLASSIFY_PATH, encoding='utf-8') -> List[str]:
    """
    传入一个文件地址， 返回文件的每一行组成的列表, 会自动去除首尾空白符
    """
    lines = []
    with open(path, encoding=encoding, mode='r') as f:
        for line in f.readlines():
            if line.isspace():
                continue

            lines.append(line.strip())

    return lines


def writeLines(lines: iter, path=UN_CLASSIFY_PATH, encoding='utf-8', mode='a', info=False):
    """
    多行写入,自动去除空白符

    :param lines: 待写入的可迭代对象

    :param path: 写入文件的路径

    :param encoding: 编码格式

    :param mode: 写入模式

    :param info: 是否打印信息
    """
    with open(path, encoding=encoding, mode=mode) as f:
        for line in tqdm(lines):
            f.write(line.strip() + '\n')

    if info:
        print(f'写入 {len(lines)} 行文本到 {path}\n编码模式: {encoding} 写入模式: {mode}\n')


def sortUnique(lines: iter, key=None, reverse=False) -> list:
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
    with open(path, mode='w', encoding=encoding) as f:
        f.write('')


def add_url(url):
    """
    添加新的已经爬取的url，并写入到文件中
    """
    if url not in URLS:
        URLS.add(url)
        writeLines([url], HAS_URLS_PATH)


def is_exist(url):
    """
    判断url是否已经存在
    """
    return str(url) in URLS