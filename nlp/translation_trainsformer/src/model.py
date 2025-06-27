import torch
import torch.nn as nn
import config
import math

class PositionEncoding(nn.Module):
    def __init__(self, dim_model,max_len=100):
        super().__init__()
        self.pe = torch.zeros(max_len, dim_model,dtype=torch.float32)
        for pos in range(max_len):
            for _2i in range(0, dim_model, 2):
                self.pe[pos, _2i] = math.sin(pos / (10000.0 ** (_2i / dim_model)))
                self.pe[pos, _2i + 1] = math.cos(pos / (10000.0 ** (_2i / dim_model)))

    def forward(self, x):
        return x + self.pe[:x.shape[1]]







class Attention(nn.Module):
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden.shape [1, batch_size, decoder_hidden_size]
        # encoder_outputs.shape [batch_size, seq_len, decoder_hidden_size]
        # attention_score.shape [batch_size, 1, seq_len]
        attention_score = torch.bmm(decoder_hidden.transpose(0, 1), encoder_outputs.transpose(1, 2))
        attention_weight = torch.softmax(attention_score, dim=-1)
        # context_vector.shape [batch_size, 1, decoder_hidden_size]
        context_vector = torch.bmm(attention_weight, encoder_outputs)
        return context_vector



class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.ENCODER_HIDDEN_SIZE,
                          num_layers=config.ENCODER_LAYERS,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        # x.shape[batch_size, seq_len]
        embedd = self.embedding(x)
        # embedd.shape[batch_size, seq_len, embedding_dim]
        output, hidden = self.gru(embedd)
        # hidden.shape[num_layers * num_directions, batch_size, hidden_size]
        last_hidden_forward = hidden[-2]
        last_hidden_backward = hidden[-1]
        last_hidden = torch.cat([last_hidden_forward, last_hidden_backward], dim=-1)
        # last_hidden.shape[batch_size, hidden_size * 2]
        # output.shape [batch_size, seq_len, hidden_size * 2]
        return output, last_hidden


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.DECODER_HIDDEN_SIZE,
                          num_layers=config.DECODER_LAYERS,
                          batch_first=True)
        self.attention = Attention()
        self.linear = nn.Linear(2 * config.DECODER_HIDDEN_SIZE, vocab_size)

    def forward(self, x, hidden_0, encoder_outputs):
        # hidden_0.shape[ num_layers, batch_size, hidden_size * 2]
        # x.shape[batch_size, 1]
        embedd = self.embedding(x)
        # embedd.shape [batch_size, 1, embedding_dim]
        output, hidden_n = self.gru(embedd, hidden_0)
        # output.shape [batch_size, 1, hidden_size]
        # hidden_n.shape [1, batch_size, hidden_size]


        # 注意力机制
        # context_vector.shape [batch_size, 1, decoder_hidden_size]
        context_vector = self.attention(hidden_n,encoder_outputs)
        # 拼接上下文向量和输出向量
        # combined.shape [batch_size, 1, decoder_hidden_size + decoder_hidden_size]
        combined = torch.cat((output, context_vector), dim=2)
        output = self.linear(combined)
        # output.shape [batch_size,1, vocab_size]
        return output, hidden_n


if __name__ == '__main__':
    pose = PositionEncoding(512,128)
    pose(torch.randn(32,128,512))