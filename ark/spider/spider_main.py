import os
import json
from typing import Generator
import requests
from bs4 import BeautifulSoup

session = requests.Session()


def get_text(url: str, **kwargs):
    """
    get 请求得到 url 的响应文本内容

    :param url: 请求的 url

    :param kwargs: requests.get 的其他参数

    :return: 响应文本内容
    """
    response = session.get(url=url, **kwargs)
    response.raise_for_status()
    return response.text


def get_tie_ba_html(tie_ba_name: str) -> str:
    """
    获取贴吧的 html 内容

    :param tie_ba_name: 贴吧名

    :return: 贴吧 html 内容
    """
    url = f'https://tieba.baidu.com/f'
    headers = {
        'Host': 'tieba.baidu.com',
        'User-Agent': 'User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;',
        'Referer': 'https://wappass.baidu.com/'
    }

    html = get_text(url, headers=headers, params={"kw": tie_ba_name})
    begin = html.find('<!--<div class="aside_region forum_info j_forum_info" id="">')
    if begin == -1:
        raise IndexError('未找到帖子内容')
    return html[begin:]


def parse_tie_ba_html(html: str) -> Generator:
    """
    解析贴吧 html 内容

    :param html: 贴吧 html 内容

    :return: 帖子标题列表
    """
    soup = BeautifulSoup(html, 'lxml')
    for tie_zi in soup.select('.thread_item_box'):
        data_field = tie_zi.get('data-field', None)

        if data_field is not None:
            data_field = json.loads(data_field)
            tie_zi_id = data_field.get('id', None)
            if tie_zi_id is not None:
                yield tie_zi_id


html = get_tie_ba_html('孙笑川')

for i in parse_tie_ba_html(html):
    print(i)