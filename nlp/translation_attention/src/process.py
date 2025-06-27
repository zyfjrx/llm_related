import pandas as pd
import config
from sklearn.model_selection import train_test_split
from tokenizer import ChineseTokenizer,EnglishTokenizer
def process():
    print("开始处理数据")
    # 读取数据
    df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt',header=None,sep='\t',usecols=[0,1],names=['en', 'zh'],encoding='utf-8')

    # 过滤空值
    df = df.dropna()
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2)

    # 构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(),config.PROCESSED_DATA_DIR / 'vocab_zh.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(),config.PROCESSED_DATA_DIR / 'vocab_en.txt')
    # 获取tokenizer
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab_en.txt')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab_zh.txt')

    # 计算长度
    zh_len = train_df['zh'].apply(lambda x: len(zh_tokenizer.tokenizer(x))).max()
    en_len = train_df['en'].apply(lambda x: len(zh_tokenizer.tokenizer(x))).max()

    # 构建并保存训练数据集
    train_df['zh'] = train_df['zh'].apply(lambda x:zh_tokenizer.encode(x,config.SEQ_LEN,add_special_tokens=False))
    train_df['en'] = train_df['en'].apply(lambda x:en_tokenizer.encode(x,config.SEQ_LEN,add_special_tokens=True))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl',orient='records',lines=True)

    # 构建并保存测试数据集
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, config.SEQ_LEN, add_special_tokens=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, config.SEQ_LEN, add_special_tokens=True))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)

    print("数据处理完成")


if __name__ == '__main__':
    process()