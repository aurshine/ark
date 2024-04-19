# 安装
```commandline
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

# 使用 analyse 分析恶意语句
```python
from ark.analyse import analyse, ByType

comments = ['你就是歌姬吧，你记住你什么都不是',
            '杀马特团长你等着我，我和你没完，你等着！！',
            '小亮给他整个活',
            'python是世界上最好的语言',
            'python是世界上最烂的语言']
print(analyse(comments, by=ByType.BY_TEXT))
```
