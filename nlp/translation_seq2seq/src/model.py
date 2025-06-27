import torch
import torch.nn as nn
import config

class TranslationEncoder(nn.Module):
    def __init__(self,vocab_size,padding_idx=0):
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
        return last_hidden

class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size,padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.DECODER_HIDDEN_SIZE,
                          num_layers=config.DECODER_LAYERS,
                          batch_first=True)
        self.linear = nn.Linear(config.DECODER_HIDDEN_SIZE,vocab_size)

    def forward(self,x, hidden_0):
        # hidden_0.shape[ num_layers,batch_size, hidden_size * 2]
        # x.shape[batch_size, 1]
        embedd = self.embedding(x)
        # embedd.shape [batch_size, 1, embedding_dim]
        output, hidden_n = self.gru(embedd, hidden_0)
        # output.shape [batch_size, 1, hidden_size]
        # hidden_n.shape [1, batch_size, hidden_size]
        output = self.linear(output)
        # output.shape [batch_size,1, vocab_size]
        return output, hidden_n
