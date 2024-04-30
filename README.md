# 安装

要求python3.9及以上

### github

```commandline
git clone https://github.com/aurshine/ark.git
cd ark
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### gitee (不保证同步更新)

```commandline
git clone https://gitee.com/jiuxiangyang/ark.git
cd ark
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

# 使用 analyse 分析恶意语句

```python
from ark.analyse import analyse

comments = ['你就是歌姬吧，你记住你什么都不是',
            '杀马特团长你等着我，我和你没完，你等着！！',
            '小亮给他整个活',
            'python是世界上最好的语言',
            '鸡你太美',
            '原神启动']
print(analyse(comments))
```

# 自调参训练模型

文件地址: `ark/train_module.py`

### 提供参数调整:

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

训练的模型均保存在 `ark/data/result-models/`路径下

可使用 setting.MODEL_LIB 指定`ark/data/result-models/`路径

```python
from ark.setting import MODEL_LIB

print(MODEL_LIB)
```

使用`train_module.train()`在训练结束后会在 `ark/` 下生成 `train.png` 和 `valid.png` 两个图片，分别表示训练集和验证集的准确率变化曲线和`matrix_confusion.png`图片表示混淆矩阵。

# 添加数据集集

$\textcolor{red}{所有非恶意数据用0表示，所有恶意数据用1表示}$

### 通过.txt文件添加数据集

1. 将非恶意数据集复制粘贴至 `ark/spider/cache/notBad.txt`, 并将恶意数据集复制粘贴至 `ark/spider/cache/bad.txt`
2. 使用 ark.data.load.update_tie_ba 合并数据集

```python
from ark.data.load import update_tie_ba
from ark.setting import NOT_BAD_TXT_PATH, BAD_TXT_PATH
from ark.spider.classify import write_lines

some_not_bad = []
some_bad = []

write_lines(some_bad, BAD_TXT_PATH, mode='a')
write_lines(some_not_bad, NOT_BAD_TXT_PATH, mode='a')

update_tie_ba() # 合并数据集
```

### 通过.csv文件添加数据集

1. 合并的csv文件需要包含TEXT列和label列
2. 与 `ark/data/COLD/tie_ba.csv` 合并
3. 使用 ark.data.load.update_tie_ba 合并数据集

```python
import pandas as pd
from ark.setting import TIE_BA_CSV_PATH
from ark.data.load import update_tie_ba

df1 = pd.read_csv(TIE_BA_CSV_PATH, encoding='utf-8')
df2 = pd.DataFrame({'TEXT': [], 'label': []})

df = pd.concat([df1, df2], ignore_index=True)
df.to_csv(TIE_BA_CSV_PATH, index=False, encoding='utf-8')

update_tie_ba()
```

# 抓取数据

```python
from ark.spider.spider_main import spider_main
from ark.setting import UN_CLASSIFY_PATH

tie_ba = ['孙笑川', '弱智'] # 吧名
spider_main(tie_ba, save_path=UN_CLASSIFY_PATH, num_work=5) # 开始抓取数据
```

# 分析数据

```python
from ark.spider.interface import tkDrive

tkDrive() # gui分析数据
```