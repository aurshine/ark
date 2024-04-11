from .classify import getLines, writeLines
from .comment import Comment, set_comment_lock, permutes

repEmoji = [('🤡', '小丑'),
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
    """文字替换emoji
    """
    for pair in repEmoji:
        emoji, rep = pair
        comment = comment.replace(emoji, rep)

    return comment


def wash_reply(comment: str):
    """清除 回复<a ...>B</a> : 的格式

    >>> wash_reply('回复<a href="...." ...>九享阳</a> :原始人启动')
    '原始人启动'
    """
    comment = comment.strip()

    reply_idx = comment.find('回复')
    tail_idx = comment.find('</a> :', reply_idx)

    if tail_idx != -1:
        comment = comment[:reply_idx] + wash_reply(comment[tail_idx + 6:])
    return comment


def wash_img(comment: str):
    """清除img标签
    """
    comment = comment.strip()
    l_idx = comment.find('<img')
    if l_idx == -1:
        return comment

    r_idx = comment.find('>', l_idx)
    return comment[: l_idx] + wash_img(comment[r_idx + 1:])


def wash_comments(comments):
    if isinstance(comments, Comment):
        comments = permutes(comments.tolist())

    set_comment_lock(1)
    washed = Comment()

    def work(cmt: str, processes: list):
        for process in processes:
            cmt = process(cmt)
        return cmt

    for comment in comments:
        comment = work(comment, [wash_emoji, wash_reply, wash_img])
        if 5 < len(comment) < 128:
            washed.append(comment)

    return washed


def wash_file(path, encoding='utf-8'):
    lines = getLines(path, encoding=encoding)
    wash_comments(lines).download(path=path, encoding=encoding, mode='w')