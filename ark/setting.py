import os


# setting.py 所在的文件夹地址
ARK_PATH = os.path.abspath(os.path.dirname(__file__))

# pretrain tokenizer 地址
PRETRAIN_TOKENIZER_PATH = os.path.join(ARK_PATH, 'data/tokenizer')

# DATASET 文件夹地址
DATASET_PATH = os.path.join(ARK_PATH, 'data/DATASET')

# 中文常见字地址
COMMON_CHAR_PATH = os.path.join(DATASET_PATH, 'common_char.txt')

# 预训练数据集
PRETRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'pretrain.txt')

# 训练结果的存储地址
TRAIN_RESULT_PATH = os.path.join(ARK_PATH, 'data/TRAIN_RESULT')


if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)
