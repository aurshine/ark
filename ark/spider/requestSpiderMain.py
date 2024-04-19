import os
import random
import time
from typing import Union, List
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from ark.setting import *
from ark.spider.wash import wash_comments
from ark.spider.classify import add_url, is_exist
from ark.spider.comment import Comment

__session = requests.Session()
retries = HTTPAdapter(max_retries=3)
__session.mount('http://', retries)
__session.mount('https://', retries)


def delay_time():
    return random.uniform(DELAY_SECONDS[0], DELAY_SECONDS[1])


def is_baidu_valid(soup):
    valid = soup.select_one('title')
    if valid and valid.text == '百度安全验证':
        print('需要百度安全验证')
        return True

    if valid:
        print(valid.text)
    else:
        print(valid)
    return False


def update_headers(headers):
    HEADERS.update(headers)


def update_proxies(proxies):
    PROXIES.update(proxies)


def session(url, mode: str, referer=None, **kwargs):
    """ 发送请求返回响应

    :param url: 请求url
    :param referer: 请求url来自的页面, 用于配置referer, 默认为None
    :param mode: get 或 post
    :return: 返回响应
    """

    print(url)
    update_headers({'Referer': referer})
    response = None
    if mode.lower() == 'get':
        response = __session.get(url=url, headers=HEADERS, proxies=PROXIES, **kwargs)
    elif mode.lower() == 'post':
        response = __session.post(url=url, headers=HEADERS, proxies=PROXIES, **kwargs)

    response.raise_for_status()
    update_headers({'Referer': url})
    time.sleep(delay_time())
    return response


def slice_html(html: str, begin_tag: str, end_tag: str):
    """对html切片"""
    begin = html.find(begin_tag)
    end = html.find(end_tag)

    return html[begin: end]


def max_prefix(match_str: str, stop_flag):
    """寻找最长前缀

    :param match_str: 匹配字符串
    :param stop_flag: 可调用参数, 传入一个字符返回 True 或 False, False时结束匹配
    :return: str
    """
    length = 0
    for s in match_str:
        if not stop_flag(s):
            break
        length += 1

    return match_str[: length]


def max_suffix(match_str: str, stop_flag):
    """寻找最长后缀

    :param match_str: 匹配字符串
    :param stop_flag: 可调用参数, 传入一个字符返回 True 或 False, False时结束匹配
    :return: str
    """
    length = len(match_str)
    for s in match_str[::-1]:
        if not stop_flag(s):
            break
        length -= 1

    return match_str[length:]


def filter_a_tag(html):
    """返回a标签组成的list
    """
    begin_tag = '<ul id="thread_list" class="threadlist_bright j_threadlist_bright">'
    end_tag = '<div class="thread_list_bottom clearfix">'
    soup = BeautifulSoup(slice_html(html, begin_tag, end_tag), 'lxml')

    if is_baidu_valid(soup):
        return []

    return soup.select('.threadlist_title a')


def join_with_tie_ba(path):
    """连接地址

    https://tieba.baidu.com + path
    """
    return urljoin('https://tieba.baidu.com', path)


def get_ba_response(tbs):
    """
    得到某些吧的响应

    :param tbs: 想要请求的吧名, 可以是一个str 或 iter[str] 类型

    :return: 返回所有响应的迭代器

    for res in get_ba_response(['孙笑川', '原神'])
    """
    if isinstance(tbs, str):
        tbs = [tbs]

    for tb in tbs:
        response = session(mode='get', url=join_with_tie_ba('/f'), params={'kw': tb},
                           referer='https://tieba.baidu.com')
        yield response


def get_total_comment_param(html, pn=1):
    """
    获取贴吧 totalComment请求需要的负载
    """
    ret = {
        'pn': pn,
        'see_lz': '0',
    }

    def check(s: str):
        return s.isdigit()

    for param in ['tid', 'fid']:
        key_str = f"{param}:'"
        idx = html.find(key_str)
        ret[param] = max_prefix(html[idx + len(key_str):], check)

    ret['t'] = int(time.time() * 1000)
    return ret


def get_total_comment(html, pn=1, referer=None):
    """
    获取一个页面帖子下的回复评论
    """

    response = session(mode='get', url=f'{join_with_tie_ba("p/totalComment")}',
                       params=get_total_comment_param(html, pn=pn),
                       referer=referer)

    comments = Comment()
    json_obj = dict(response.json())
    if json_obj['errmsg'] != 'success':
        print(json_obj['errmsg'])
    else:
        comment_list = json_obj['data']['comment_list']

        if isinstance(comment_list, dict):
            for tid, cm_list in comment_list.items():
                for comment_message in cm_list['comment_info']:
                    comments.append(comment_message['content'])
        else:
            print('[]')

    return comments


def get_comment(html=Union[None, str], soup=Union[None, BeautifulSoup], pn=1, referer=None):
    """获取一个页面的评论

    :param html: None 或 字符串, 在soup为None的时候使用

    :param soup: None 或 BeautifulSoup, 不为 None的时候优先使用

    :param pn: page_num 某个帖子的页面页数, 下标从1开始

    :return: list[str]
    """
    if soup is None:
        soup = BeautifulSoup(html, 'lxml')

    comments = Comment()
    if not is_baidu_valid(soup):
        for div_tag in soup.select('.j_d_post_content'):
            comments.append(div_tag.text)

        comments += get_total_comment(soup.prettify(), pn=pn, referer=referer)

    return comments


def page_spider(url, num_page, comments=None, referer=None):
    """爬取某个帖子的前num_page页

    :param url: 帖子的url

    :param num_page: 需要爬取的页数

    :param comments: 爬取结果存储的地方, list

    :return: comments
    """
    if comments is None:
        comments = Comment()

    for pn in range(num_page):
        page_resp = session(mode='get', url=url, params={'pn': pn}, referer=referer)

        soup = BeautifulSoup(page_resp.text, 'lxml')
        if not is_baidu_valid(soup):
            comments += get_comment(soup=soup, pn=pn + 1, referer=page_resp.url)

            current_pn = int(soup.select_one('.l_reply_num > :nth-child(2)').text)
            if not current_pn or pn >= current_pn:
                break

    return comments


def spider(tbs: Union[None, str, List[str]] = "孙笑川", num_pages=1, download=True):
    """
    :param tbs: 需要爬的吧名, 默认孙吧

    :param num_pages: 每个吧贴爬取的页数, 可以是int 或 list

    :param download: 是否自动下载

    :return: 爬取的评论清洗后组成的Comment对象
    """
    if isinstance(tbs, str):
        tbs = [tbs]

    if isinstance(num_pages, int):
        num_pages = [num_pages] * len(tbs)

    comments = Comment()
    try:
        for ba_resp, num_page in zip(get_ba_response(tbs), num_pages):
            for a_tag in filter_a_tag(ba_resp.text):
                url = join_with_tie_ba(a_tag.get('href', None))
                if url is not None:
                    tid = max_suffix(url, lambda x: x.isdigit())
                    if is_exist(tid):
                        print(f'{tid} exist')
                        continue
                    page_spider(url, num_page, comments, referer=ba_resp.url)
                    add_url(tid)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(e)
    finally:
        print('finally')
        ret = wash_comments(comments)
        if download:
            ret.download()
        return ret
