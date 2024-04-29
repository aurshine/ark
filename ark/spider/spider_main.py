import random
import time
import json
import traceback
from typing import Generator, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor as Pool

import requests
from bs4 import BeautifulSoup

from ark.spider.classify import is_exist, add_url
from ark.spider.wash import wash_comments, Comment
from ark.setting import HEADERS, DELAY_SECONDS


__session__ = requests.Session()


def get_text(url: str, **kwargs) -> Optional[str]:
    """
    get 请求得到 url 的响应文本内容

    :param url: 请求的 url

    :param kwargs: requests.get 的其他参数

    :return: 响应文本内容
    """
    # 随机延时
    time.sleep(random.uniform(DELAY_SECONDS[0], DELAY_SECONDS[1]))
    if 'headers' not in kwargs:
        kwargs['headers'] = HEADERS

    response = __session__.get(url=url, **kwargs)
    response.raise_for_status()
    return response.text


def get_tie_ba_html(tie_ba_name: str, page=0, **kwargs) -> str:
    """
    获取贴吧的 html 内容

    :param tie_ba_name: 贴吧名

    :param page: 页码, 默认为 0

    :param kwargs: requests.get() 的参数

    :return: 贴吧 html 内容
    """
    url = f'https://tieba.baidu.com/f'
    params = {'kw': tie_ba_name, 'pn': 50 * page, 'ie': 'utf-8'}

    html = get_text(url, params=params, **kwargs)
    if html is None:
        return ''
    begin = html.find('<ul id="thread_list" class="threadlist_bright j_threadlist_bright">')
    end = html.find('<div class="thread_list_bottom clearfix">')

    if begin == -1 or end == -1:
        raise IndexError(f'未找到{tie_ba_name}贴吧 {page} 页的帖子内容')
    return html[begin: end]


def parse_tie_ba_html(html: str) -> Generator[str, None, None]:
    """
    解析贴吧 html 内容

    :param html: 贴吧 xx吧 html 内容

    :return: 帖子 id 的生成器
    """
    soup = BeautifulSoup(html, 'lxml')
    for tie_zi in soup.select('.thread_item_box'):
        data_field = tie_zi.get('data-field', None)

        if data_field is not None:
            data_field = json.loads(data_field)
            tie_zi_id = data_field.get('id', None)
            if tie_zi_id is not None:
                yield str(tie_zi_id)


def into_tie_ba_content_page(p_id: str, **kwargs) -> Comment:
    """
    进入贴吧帖子内容页

    :param p_id: 帖子 id

    :param kwargs: requests.get() 的参数
    """
    if is_exist(p_id):
        return Comment()

    url = f'https://tieba.baidu.com/p/{p_id}'
    html = get_text(url, **kwargs)
    if html is None:
        return Comment()
    soup = BeautifulSoup(html, 'lxml')
    main_comments = soup.select('.d_post_content')
    replay_comments = soup.select('.lzl_content_main')

    res = [comment.text for comment in main_comments + replay_comments]
    add_url(p_id)
    return wash_comments(res)


def spider_main(tie_ba_names: Union[str, List[str]], save_path: str, num_work=None, encoding=None):
    """
    贴吧爬虫主函数

    :param tie_ba_names: 贴吧名列表或单个贴吧名

    :param save_path: 保存路径

    :param num_work: 线程池线程数, 默认为 None, 表示不使用线程池

    :param encoding: 编码
    """
    if isinstance(tie_ba_names, str):
        tie_ba_names = [tie_ba_names]

    tids = []
    for tie_ba_name in tie_ba_names:
        html = get_tie_ba_html(tie_ba_name)
        tids.extend(parse_tie_ba_html(html))

    if num_work is None:
        for tid in tids:
            into_tie_ba_content_page(tid).download(path=save_path, encoding=encoding, mode='a')
    else:
        with Pool(max_workers=num_work) as pool:
            def work(t_id):
                into_tie_ba_content_page(t_id).download(path=save_path, encoding=encoding, mode='a')
            for tid in tids:
                pool.submit(work, tid)