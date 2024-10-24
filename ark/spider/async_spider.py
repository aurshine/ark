import os
import asyncio
from typing import List, Dict

import aiohttp
from bs4 import BeautifulSoup

from ark.setting import HEADERS


def clear_reply(relpy_text: str):
    """清洗回复文本"""
    text = relpy_text.split('：', 1)[-1].strip()
    print(text)
    exit()
    return text


def catch_replies(reply_filed: BeautifulSoup) -> List[str]:
    """返回某个回复楼层的回复列表"""
    reply_list = []
    for reply_tag in reply_filed.select('.'):
        reply_list.append(clear_reply(reply_tag.text))

    return reply_list


def catch_comments_from_comment_field(comment_field: BeautifulSoup) -> Dict[str, str]:
    """返回某个评论的主楼和所有回复"""
    master_field = comment_field.select_one('.p_content')
    reply_field = comment_field.select_one('.j_lzl_m_w')

    result = {
        'master_comment': master_field.text.strip(),
        'reply_comments': catch_replies(reply_field)
    }

    return result


async def catch_comments_from_tie_ba_pn(session: aiohttp.ClientSession, main_url: str, pn: int) -> List[Dict[str, str]]:
    """返回某个帖子的某一页的评论列表"""
    response = await session.get(main_url, params={'pn': pn})
    response.raise_for_status()
    html = await response.text()
    soup = BeautifulSoup(html, 'lxml')

    comments = []
    for comment_field in soup.select('.l_post'):
        comments.append(catch_comments_from_comment_field(comment_field))
    return comments


async def catch_comments_from_tie_ba(session: aiohttp.ClientSession, pid: str) -> List[Dict[str, str]]:
    """返回某个帖子的所有评论列表"""
    content_main_url = f'https://tieba.baidu.com/p/{pid}'
    response = await session.get(content_main_url)

    response.raise_for_status()
    html = await response.text()
    soup = BeautifulSoup(html, 'lxml')

    comments = []
    pn = 1
    for i in range(pn):
        comments.extend(await catch_comments_from_tie_ba_pn(session, content_main_url, pn + 1))

    return comments


async def catch_content_urls(session: aiohttp.ClientSession, tie_ba_name: str) -> List[Dict[str, str]]:
    """返回某个吧的帖子链接列表"""
    response = await session.get(f'https://tieba.baidu.com/f?kw={tie_ba_name}')
    response.raise_for_status()
    html = await response.text()
    soup = BeautifulSoup(html, 'lxml')

    urls_msg = []
    for item in soup.select('.thread_item_box'):
        urls_msg.append({
            'tid': item.get('data-tid')
        })

    return urls_msg


async def spider_main(tie_ba_names: List[str], save_path: str):
    """
    异步爬虫主函数

    :param tie_ba_names: 吧名列表

    :param save_path: 结果保存路径
    """
    async with aiohttp.ClientSession() as session:
        session.headers.update(HEADERS)
        catch_url_tasks = [catch_content_urls(session, name) for name in tie_ba_names]

        results = []
        for result in await asyncio.gather(*catch_url_tasks):
            results.extend(result)

        catch_comment_tasks = [catch_comments_from_tie_ba(session, result['tid']) for result in results]
        comments = await asyncio.gather(*catch_comment_tasks)

asyncio.run(spider_main(['孙笑川'], ''))