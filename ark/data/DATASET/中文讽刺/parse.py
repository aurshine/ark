import pandas as pd

path = './dataset/review.json'


df = {'TEXT': [], 'label': []}

with open(path, encoding='gbk', mode='r') as f:
    for line in f.readlines():
        d = eval(line)
        df['TEXT'].append(d['review'])
        df['label'].append(d['isSarcasm'])

pd.DataFrame(df, dtype=str).to_csv('train.csv', encoding='utf-8', sep=',', index=False)

