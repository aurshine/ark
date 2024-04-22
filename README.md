# 安装
```commandline
git clone https://github.com/aurshine/ark.git
cd ark
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

# 自调参训练模型
`ark/train_module.py`
提供参数调整:
- 隐藏层大小
- 多头注意力头数
- 编码器层数
- 解码器层数
- 每个文本的步长
- 随机失活率
- 交叉验证折数
- 验证次数
- 批量大小
- 最大训练轮数
- 最小停止轮数
- 最小停止损失值
- 优化器参数

```python
from ark import train_module

train_module.HIDDEN_SIZE = 128 # 修改隐藏层大小
train_module.NUM_HEADS = 8 # 修改多头注意力头数
train_module.OPTIMIZER_PARAMS['lr'] = 0.001 # 修改优化器学习率

train_module.train() # 开始训练模型, 并验证效果

train_module.train_only() # 仅训练模型, 不验证效果
```