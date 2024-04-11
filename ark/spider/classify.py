from ark.setting import *
from tqdm import tqdm


def getLines(path=UN_CLASSIFY_PATH, encoding='utf-8'):
    """传入一个文件地址， 返回文件的每一行组成的列表, 会自动去除首尾空白符
    """
    lines = []
    with open(path, encoding=encoding, mode='r') as f:
        for line in f.readlines():
            if line.isspace():
                continue

            lines.append(line.strip())

    return lines


def writeLines(lines: iter, path=UN_CLASSIFY_PATH, encoding='utf-8', mode='a', info=False):
    """多行写入,自动去除空白符
    """
    if info:
        print(f'写入 {len(lines)} 行文本到 {path}\n编码模式: {encoding} 写入模式: {mode}\n')
    with open(path, encoding=encoding, mode=mode) as f:
        for line in tqdm(lines):
            f.write(line.strip() + '\n')


def sortUnique(lines, key=None, reverse=False):
    return list(set(sorted(lines, key=key, reverse=reverse)))


def clear(path, encoding='utf-8'):
    with open(path, mode='w', encoding=encoding) as f:
        f.write('')


def add_url(url):
    """添加新的url"""

    if not DEBUG:
        if url not in URLS:
            URLS.add(url)
            writeLines([url], HAS_URLS_PATH)


def is_exist(url):
    return str(url) in URLS