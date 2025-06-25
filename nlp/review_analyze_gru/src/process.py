import pandas as pd
import config
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer


def process():
    print("开始处理数据")
    df = pd.read_csv(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv',usecols=['review','label'],encoding='utf-8')
    df = df.dropna()

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2,stratify=df['label'])
    print(len(train_df))

    # 构建词表
    JiebaTokenizer.build_vocab(train_df['review'].tolist(), config.PROCESSED_DATA_DIR / 'vocab.txt')
    # 构建tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 构建训练数据
    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))
    # print(train_df['review'].apply(lambda x: len(x)).quantile(0.95))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', lines=True,orient='records')

    # 构建测试数据
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x,config.SEQ_LEN ))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', lines=True,orient='records')

    print("数据处理完成")

if __name__ == '__main__':
    process()