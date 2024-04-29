import os


# setting.py 所在的文件夹地址
SETTING_PATH = os.path.abspath(os.path.dirname(__file__))

# cache文件夹地址
CACHE_PATH = os.path.join(SETTING_PATH, 'spider/cache')

# 未分类文件的存储地址
UN_CLASSIFY_PATH = os.path.join(CACHE_PATH, 'un_classify.txt')

# 恶意语句的存储地址
BAD_TXT_PATH = os.path.join(CACHE_PATH, 'bad.txt')

# 非恶意语句的存储地址
NOT_BAD_TXT_PATH = os.path.join(CACHE_PATH, 'notBad.txt')

# 爬取过的url的存储地址
HAS_URLS_PATH = os.path.join(CACHE_PATH, 'hasUrls')

# DATASET 文件夹地址
DATASET_PATH = os.path.join(SETTING_PATH, 'data/DATASET')

# 词表存储地址
VOCAB_PATH = os.path.join(DATASET_PATH, 'vocab.txt')

# 拼音表储存地址
PINYIN_VOCAB_PATH = os.path.join(DATASET_PATH, 'pinyin.txt')

# 首字母表存储地址
LETTER_VOCAB_PATH = os.path.join(SETTING_PATH, 'letter.txt')

# 贴吧数据集地址
TIE_BA_CSV_PATH = os.path.join(DATASET_PATH, 'tie-ba.csv')

# 中文常见字地址
COMMON_CHAR_PATH = os.path.join(DATASET_PATH, 'common_char.txt')

# 训练好的模型的存放文件夹
MODEL_LIB = os.path.join(SETTING_PATH, 'data/result-models')

for dir_name in [CACHE_PATH, DATASET_PATH, MODEL_LIB]:
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

with open(HAS_URLS_PATH, encoding='utf-8', mode='r') as f:
    URLS = set(f.readlines())

# 爬虫 headers 配置
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Host': 'tieba.baidu.com',
    'Referer': 'https://tieba.baidu.com',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Cookies': 'BAIDUID=6C4F7A6BC5D09FEA4DF6888654BC7836:FG=1; BIDUPSID=A22BAC4469DCE881045BA85E363DE0BE; BAIDUID_BFESS=6C4F7A6BC5D09FEA4DF6888654BC7836:FG=1; BAIDU_WISE_UID=wapp_1704972177528_985; BDUSS=JUcDFsbVlnOE9wflltbE9xaXNSWU1WZnQzUy1keTZYNXAxVTRVOU11MWdLc2hsSVFBQUFBJCQAAAAAAQAAAAEAAADh1gpMvsXP7dH0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGCdoGVgnaBlYU; BDUSS_BFESS=JUcDFsbVlnOE9wflltbE9xaXNSWU1WZnQzUy1keTZYNXAxVTRVOU11MWdLc2hsSVFBQUFBJCQAAAAAAQAAAAEAAADh1gpMvsXP7dH0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGCdoGVgnaBlYU; ZFY=QGHsDI7QjwlxKp1fQWV3:BNXGlzncswak2avhCUJKopg:C; __bid_n=18e84c4d1d343686668152; H_PS_PSSID=40171_40080_40368_40415_40299_40467_40458_40479_40317_40512_40445_60045_60028_60048_40510; STOKEN=a33e3d28624d24dc1f5d2238bacaf925bad249a08ce9306393e2252402236286; IS_NEW_USER=c57bab2c65bea05802f2cbed; arialoadData=false; USER_JUMP=-1; Hm_lvt_98b9d8c2fd6608d564bf2ac2ae642948=1713790250,1713878058,1713970010,1714012818; st_key_id=17; 5570746081_FRSVideoUploadTip=1; video_bubble5570746081=1; wise_device=0; tb_as_data=3492ae0ee2b8312e01c9a6da5dbf736699a733976b137df298855f05127132d4bc96b02e789571600005f83ac49b3dce2204d0b66081b773721a966d4f749ca95cea1a4cb9c5243d1417f6800382522632e194370fcc19118b84bca7a1a7cb54ec414345f63951c402c7e608df3cf484; Hm_lpvt_98b9d8c2fd6608d564bf2ac2ae642948=1714015343; XFI=08c6e9f0-02b3-11ef-bd1c-b57471cc166b; BA_HECTOR=8lak000h8g0hcha40500a0a46b76ef1j2jj3g1t; ab_sr=1.0.1_MmFiMTIwMDQyZWQxMmMxMzY4OGFjNDMyNTBhMzdhMGZiZWQzOTAzZTI4OTVjMjRiNjI4MGY3YzA5YjMyNjRhMmM3ZGMzYTdhZWMwZDA3MjNkZGVjMTNkMmQ0MTZlYTI5MWQ3YTdjZDJiZjc1YTFiYzNjZDJkODhjZWRhNjM2YjY1MGJhOGVhZWI1NjkzMTljN2RlM2FmYjMwZDU0ZDFhNmNjNzQ2NTM5YThkZGM5MDQ5YzhiNTkyMWU0MDQ5ODc0; st_data=8fde35937ed2d71269287ba3068d105afbe18fcb2be1e9ca24ac9881f27cfc4acd25bc1e37dadad6296ea95f0028d31870b479b0ffedfd2a7eef069956bfbeb14236e1d672111bccc76fafdde0bad91cdad3e9929aab0625d31b5ee289df95a51682b569a33e48c0a0429dd33fdcef8a63f6beb834846dbcc64c3ff53f7e8d57ed1713639a8d128f4e27c3b3082bcd21; st_sign=5f07d1c0; XFCS=C7C9CDB9AEEDDE938268A778820716F067911F79AF6567F6EC8523D81B4DBBE8; XFT=emZgP0iFT16a3WJIAdu8+iRS+jgBOqAz5/jV506Bfkc=; RT="z=1&dm=baidu.com&si=4cb8b2fb-3fef-4189-b0bf-7eb211c7ca08&ss=lvemz5av&sl=1i&tt=10hh&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf&ld=1i940&ul=1iasc"',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Sec-Ch-Ua': '" Not A;Brand";v="99", "Chromium";v="124", "Google Chrome";v="124"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1'
}


# 重连最大重试次数
MAX_RETRIES = 3

PROXIES = {
    # 'https': '47.106.107.212:3128',
}

# tuple对象, 每次请求后的时延区间
DELAY_SECONDS = (5, 10)