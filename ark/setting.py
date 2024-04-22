import os

with open(os.path.join(os.path.dirname(__file__), 'spider/cache/hasUrls')) as f:
    URLS = set(line.strip() for line in f.readlines())

# setting.py 所在的文件夹地址
SETTING_PATH = os.path.abspath(os.path.dirname(__file__))

# 未分类文件的存储地址
UN_CLASSIFY_PATH = os.path.join(SETTING_PATH, 'spider/cache/un_classify.txt')

# 恶意语句的存储地址
BAD_TXT_PATH = os.path.join(SETTING_PATH, 'spider/cache/bad.txt')

# 非恶意语句的存储地址
NOT_BAD_TXT_PATH = os.path.join(SETTING_PATH, 'spider/cache/notBad.txt')

# 词表存储地址
VOCAB_PATH = os.path.join(SETTING_PATH, 'data/COLD/vocab.txt')

# 拼音表储存地址
PINYIN_VOCAB_PATH = os.path.join(SETTING_PATH, 'data/COLD/pinyin.txt')

# 首字母表存储地址
LETTER_VOCAB_PATH = os.path.join(SETTING_PATH, 'data/COLD/letter.txt')

# 爬取过的url的存储地址
HAS_URLS_PATH = os.path.join(SETTING_PATH, 'spider/cache/hasUrls')

# 贴吧数据集地址
TIE_BA_CSV_PATH = os.path.join(SETTING_PATH, 'data/COLD/tie-ba.csv')

# 训练好的模型的存放文件夹
MODEL_LIB = os.path.join(SETTING_PATH, 'data/result-models')
if not os.path.exists(MODEL_LIB):
    os.mkdir(MODEL_LIB)

# 爬虫 headers 配置
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
    'Host': 'tieba.baidu.com',
    'Referer': 'https://tieba.baidu.com/',
    'Connection': 'keep-alive',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Microsoft Edge";v="120"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': "Windows",
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1'
}


# 重连最大重试次数
MAX_RETRIES = 3

PROXIES = {
    # 'https': '47.106.107.212:3128',
}

# tuple对象, 每次请求后的时延区间
DELAY_SECONDS = (1, 3)