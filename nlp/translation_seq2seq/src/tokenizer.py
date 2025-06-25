import jieba
from tqdm import tqdm
from abc import abstractmethod
import config
from nltk import word_tokenize,TreebankWordDetokenizer

class BaseTokenizer:
    unk_token = '<UNK>'
    pad_token = '<PAD>'
    sos_token = '<SOS>'
    eos_token = '<EOS>'
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: idx for idx, word in enumerate(vocab_list)}
        self.index2word = {idx: word for idx, word in enumerate(vocab_list)}
        self.unk_token_id = self.word2index[self.unk_token]
        self.pad_token_id = self.word2index[self.pad_token]
        self.sos_token_id = self.word2index[self.sos_token]
        self.eos_token_id = self.word2index[self.eos_token]

    @staticmethod
    @abstractmethod
    def tokenizer(text):
        pass

    def encode(self,text,seq_len,add_special_tokens=True):
        tokens = self.tokenizer(text)
        if add_special_tokens:
            if len(tokens) == seq_len -2:
                tokens = [self.sos_token] + tokens + [self.eos_token]
            elif len(tokens) < seq_len -2:
                tokens = [self.sos_token] + tokens + [self.eos_token] + [self.pad_token] * (seq_len - len(tokens) -2)
            else:
                tokens = [self.sos_token] + tokens[:seq_len-2] + [self.eos_token]
        else:
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            elif len(tokens) < seq_len:
                tokens.extend([self.pad_token] * (seq_len - len(tokens)))
        return [self.word2index.get(token, self.unk_token_id) for token in tokens]


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
            for word in cls.tokenizer(sentence):
                if word.split() != '': # 去掉不可见字符
                   vocab_set.add(word)
        vocab_list = [cls.pad_token,cls.unk_token, cls.sos_token, cls.eos_token] + list(vocab_set)
        print(f"词表大小: {len(vocab_list)}")
        # 保存词表
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')
        print("词表保存完成")

    @abstractmethod
    def decode(self,word_ids):
        pass

class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenizer(cls, text):
        return list(text)

    def decode(self,word_ids):
        word_list = [self.index2word[word_id] for word_id in word_ids]
        return ''.join(word_list)


class EnglishTokenizer(BaseTokenizer):
    @classmethod
    def tokenizer(cls, text):
        return word_tokenize(text)

    def decode(self,word_ids):
        word_list = [self.index2word[word_id] for word_id in word_ids]
        return TreebankWordDetokenizer.detokenize(word_list)


if __name__ == '__main__':
    tokenizer = BaseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    # word_list = tokenizer.encode("我喜欢坐地铁", )
    # print(word_list)