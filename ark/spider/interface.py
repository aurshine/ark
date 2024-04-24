from time import time
import tkinter as tk
from ark.spider.classify import get_lines
from ark.spider.comment import Comment
from ark.setting import *
from ark.data.load import update_tie_ba


font_style = ('宋体', 20, 'bold')
comments = Comment(get_lines(path=UN_CLASSIFY_PATH))

curIdx, maxLength = 0, len(comments)  # 当前读到
labels = [0 for _ in range(maxLength)]  # 初始所有语句都分在unClassify

# 类别
SKIP, UN_CLASSIFY, N_BAD, BAD = -1, 0, 1, 2

# 特殊命令
SPECIAL_DEL = 0
SPECIAL_MOD = 1
SPECIAL_REPLACE = 2


windows = tk.Tk()


def special_command(command: str):
    command = command.strip().lower()

    if command.startswith('del') or len(command) == 0:
        return SPECIAL_DEL
    elif command.startswith('mod'):
        return SPECIAL_MOD
    else:
        return SPECIAL_REPLACE


def reFlashText():
    text.delete(1.0, tk.END)
    text.insert(tk.END, comments[curIdx][-1])


def clickNext():
    global curIdx
    curIdx = min(curIdx + 1, maxLength - 1)
    reFlashText()
    index.set(getIndex())


def clickLast():
    global curIdx
    curIdx = max(curIdx - 1, 0)
    reFlashText()
    index.set(getIndex())


def clickNBad():
    labels[curIdx] = N_BAD
    clickNext()


def clickBad():
    labels[curIdx] = BAD
    clickNext()


def clickSkip():
    labels[curIdx] = SKIP
    clickNext()


def clickReplace():
    _old, _new = entry1.get(), entry2.get()

    sc = special_command(_new)
    if sc == SPECIAL_REPLACE:
        comments[curIdx].append(_old, _new)
    elif sc == SPECIAL_MOD:
        comments[curIdx].base_string = comments[curIdx].base_string.replace(_old, _new[3:].strip())
    elif sc == SPECIAL_DEL:
        comments[curIdx].base_string = comments[curIdx].base_string.replace(_old, "")

    reFlashText()
    entry1.delete(0, "end")
    entry2.delete(0, "end")


def getIndex():
    return f"{str(curIdx + 1)}/{maxLength}"


windows.title("语义恶意分类识别")
windows.geometry("700x700+400+50")
divs = [tk.Frame(windows) for _ in range(4)]

index = tk.Variable(windows, value=getIndex())

tk.Label(divs[0], textvariable=index, font=font_style).pack()

inlineReplace = tk.Label(divs[1], pady='10px')
tk.Label(inlineReplace, text='原词: ', font=font_style).grid(row=0, column=0)
entry1 = tk.Entry(inlineReplace, width=10, font=font_style)
entry1.grid(row=0, column=1)
tk.Button(inlineReplace, text='<=替换=>', font=('宋体', 15), command=clickReplace).grid(row=0, column=2)
tk.Label(inlineReplace, text='替换词: ', font=font_style).grid(row=0, column=3)
entry2 = tk.Entry(inlineReplace, width=10, font=font_style)
entry2.grid(row=0, column=4)

text = tk.Text(divs[1], bg='#fff', width=32, height=8, font=font_style, wrap=tk.CHAR)
text.pack()
reFlashText()
inlineReplace.pack()

tk.Button(divs[2], text="非恶意", font=font_style, command=clickNBad).grid(row=0, column=0, padx='25px')
tk.Button(divs[2], text="跳过", font=font_style, command=clickSkip).grid(row=0, column=1, padx='25px')
tk.Button(divs[2], text="恶意", font=font_style, command=clickBad).grid(row=0, column=2, padx='25px')

tk.Button(divs[3], text="上一个", width=10, command=clickLast).pack(side=tk.LEFT, padx='25px')
tk.Button(divs[3], text="下一个", width=10, command=clickNext).pack(side=tk.RIGHT, padx='25px')

for div in divs:
    div.pack(pady='5px')


def tkDrive():
    start = time()
    try:
        windows.mainloop()
    finally:
        class_bad, class_n_bad, un_class = Comment(), Comment(), Comment()
        for comment, label in zip(comments, labels):
            if label == SKIP:
                continue
            elif label == BAD:
                class_bad.append(comment)
            elif label == N_BAD:
                class_n_bad.append(comment)
            else:
                un_class.append(comment)

        class_bad.download(path=BAD_TXT_PATH, mode='a')
        class_n_bad.download(path=NOT_BAD_TXT_PATH, mode='a')
        un_class.download(path=UN_CLASSIFY_PATH, mode='w')

        print(f'\n分类完成: 共耗时{time() - start: .2f} sec 共浏览 {curIdx} 条数据\n'
              f'恶意语义有 {len(class_bad)} 条\n'
              f'非恶意语义有 {len(class_n_bad)} 条\n'
              f'未分类语义有 {len(un_class)}条\n')

        update_tie_ba()