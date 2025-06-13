import torch
import torch.nn as nn
import jieba

torch.manual_seed(42)
text = "自然语言是由文字构成的，而语言的含义是由单词构成的。即单词是含义的最小单位。因此为了让计算机理解自然语言，首先要让它理解单词含义。"
stopwords = {"的", "是", "而", "由", "，", "。", "、"}
words = [word for word in jieba.lcut(text) if word not in stopwords]
vocab = list(set(words))
word2idx = dict()
for i, word in enumerate(vocab):
    word2idx[word] = i
embedding = nn.Embedding(len(vocab), 5)
for idx, word in enumerate(vocab):
    word_embedding = embedding(torch.tensor(idx))
    print(f"{idx:>2}:{word:8}\t{word_embedding.detach().numpy()}")