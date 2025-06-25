import jieba
from tqdm import tqdm

import config

class JiebaTokenizer:
    unk_token = '<UNK>'
    pad_token = '<PAD>'
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: idx for idx, word in enumerate(vocab_list)}
        self.index2word = {idx: word for idx, word in enumerate(vocab_list)}
        self.unk_token_id = self.word2index[self.unk_token]
        self.pad_token_id = self.word2index[self.pad_token]
    @staticmethod
    def tokenizer(text):
        return jieba.lcut(text)

    def encode(self,text,seq_len):
        tokens = self.tokenizer(text)
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens.extend([self.pad_token] * (seq_len - len(tokens)))
        return [self.word2index.get(token, self.unk_token_id) for token in tokens]
    def decode(self,tokens):
        return [self.index2word[token] for token in tokens]
    @classmethod
    def from_vocab(cls, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        print(f"vocab size: {len(vocab_list)}")
        return cls(vocab_list)
    @classmethod
    def build_vocab(cls, sentences,vocab_file):
        vocab_set = set()
        for sentence in tqdm(sentences, desc="构建词表"):
            for word in jieba.lcut(sentence):
                if word.split() != '': # 去掉不可见字符
                   vocab_set.add(word)
        vocab_list = [cls.pad_token,cls.unk_token] + list(vocab_set)
        print(f"词表大小: {len(vocab_list)}")
        # 保存词表
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')
        print("词表保存完成")


if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    word_list = tokenizer.encode("我喜欢坐地铁", )
    print(word_list)