import pandas as pd
import config
from sklearn.model_selection import train_test_split
import jieba
from tqdm import tqdm

jieba.setLogLevel(jieba.logging.WARNING)
def process():
    print("开始处理数据")

    df = pd.read_json(config.RAW_DATA_DIR / 'synthesized_.jsonl',lines=True,orient='records').sample(frac=0.1)
    print(df.head())
    # 抽取数据
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])
    print(f"句子总数: {len(sentences)}")

    # 划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)
    print(f"训练集句子总数: {len(train_sentences)}")
    print(f"测试集句子总数: {len(test_sentences)}")

    # 构建词表，用训练集构建词表
    vocab_set = set()
    for sentence in tqdm(train_sentences,desc="构建词表"):
        for word in jieba.lcut(sentence):
            vocab_set.add(word)
    vocab_list = ['<UNK>'] + list(vocab_set)
    print(f"词表大小: {len(vocab_list)}")

    word2index = {word:index for index, word in enumerate(vocab_list)}
    indexed_train_sentences = [[word2index[word] for word  in jieba.lcut(sentence)] for sentence in train_sentences]
    indexed_test_sentences = [[word2index[word] for word  in jieba.lcut(sentence)] for sentence in test_sentences]







    print("数据处理完成")

if __name__ == '__main__':
    process()