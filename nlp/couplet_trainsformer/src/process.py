import pandas as pd
import config
from sklearn.model_selection import train_test_split

from tokenizer import ChineseTokenizer, EnglishTokenizer

def process():
    print("开始处理数据")
    # 读取数据
    df = pd.read_json(config.RAW_DATA_DIR / 'couplet.jsonl',orient='records',lines=True)
    # 划分训练测试数据集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 构建词表
    words = df['in'].tolist() + df['out'].tolist()
    ChineseTokenizer.build_vocab(words, config.PROCESSED_DATA_DIR / 'vocab.txt')
    # ChineseTokenizer.build_vocab(train_df['out'].tolist(), config.PROCESSED_DATA_DIR / 'vocab_out.txt')
    # 获取tokenizer
    tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')




    # 构建并保存训练数据集
    train_df['in'] = train_df['in'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN, add_special_tokens=False))
    train_df['out'] = train_df['out'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN, add_special_tokens=True))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)

    # 构建并保存测试数据集
    test_df['in'] = test_df['in'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN, add_special_tokens=False))
    test_df['out'] = test_df['out'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN, add_special_tokens=True))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)

    print("数据处理完成")


if __name__ == '__main__':
    process()
