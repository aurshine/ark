from typing import List, Union

from ark.spider.classify import get_lines
from ark.spider.comment import Comment, permutes


rep_emoji = [('🤡', '小丑'),
            ('🐶', '舔狗'),
            ('🐭🐭', '鼠鼠'),
            ('🐭', '我'),
            ('∠', '胶'),
            ('lz', '楼主'),
            ('打🦶', '打胶'),
            ('😆', ''),
            ('🍋', '你妈'),
            ('⭐', '性'),
            ('🌟', '性'),
            ('🐟', '欲'),
            ('💦', '喷水'),
            ('💩', '屎'),
            ('🐢', '龟'),
            ('🍉', '瓜'),
            ('🔒', '说'),
            ('🚪', '们'),
            ('😡', ''),
            ('🤣', ''),
            ('😰', ''),
            ('😋', ''),
            ('✈', '飞机'),
            ('💤', '睡'),
            ('😅', ''),
            ('🐮', '牛'),
            ('🍺', '批'),
            ('🍠', '小红书'),
            ('⭕', '拳'),
            ('➕', '+'),
            ('➗', '畜生'),
            ('👊', '拳'),
            ('🧠', '脑子'),
            ('😭', ''),
            ('👉🏼', ''),
            ('🥵', ''),
            ('👴🏻', '爷'),
            ('🎣', '钓鱼'),
            ('🐴', '马')]


def wash_emoji(comment: str):
    """
    文字替换emoji
    """
    for pair in rep_emoji:
        emoji, rep = pair
        comment = comment.replace(emoji, rep)

    return comment


def wash_reply(comment: str):
    """清除 回复 用户名 :回复内容 的格式

    >>> wash_reply('回复 九享阳 :原始人启动')
    '原始人启动'
    """
    if comment.startswith('回复'):
        end_idx = comment.find(' :')
        comment = comment[end_idx + 2:]
    return comment


def wash_comments(comments: Union[Comment, List[str]], wash_rule=None, filter_rule=None) -> Comment:
    """
    清洗评论，包括emoji替换，回复格式清除，评论长度限制

    :param comments: 评论列表或Comment对象

    :param wash_rule: 评论清洗规则函数，输入为评论字符串，输出为清洗后的评论字符串

    :param filter_rule: 评论过滤规则函数，输入为评论字符串，输出为True或False，True表示保留该评论，False表示过滤该评论
    """
    if isinstance(comments, Comment):
        comments = permutes(comments.tolist())

    wash_rules = [str.strip, wash_emoji, wash_reply]
    if wash_rule is not None:
        wash_rules.append(wash_rule)

    filter_rules = [lambda x: 5 < len(x) < 128]
    if filter_rule is not None:
        filter_rules.append(filter_rule)

    washed = Comment()
    for comment in comments:
        for wr in wash_rules:
            comment = wr(comment)

        if all(fr(comment) for fr in filter_rules):
            washed.append(comment)

    return washed


def wash_file(path, wash_rule=None, filter_rule=None, encoding=None):
    """
    清洗文件中的评论，并保存到文件中

    :param path: 文件路径

    :param wash_rule: 评论清洗规则函数，输入为评论字符串，输出为清洗后的评论字符串

    :param filter_rule: 评论过滤规则函数，输入为评论字符串，输出为True或False，True表示保留该评论，False表示过滤该评论

    :param encoding: 文件编码
    """
    lines = get_lines(path, encoding=encoding)
    wash_comments(lines, wash_rule=wash_rule, filter_rule=filter_rule).download(path=path, encoding=encoding, mode='w')