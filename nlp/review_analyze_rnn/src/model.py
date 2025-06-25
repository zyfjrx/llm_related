import torch.nn as nn
import config


class ReviewAnalyzeModel(nn.Module):
    def __init__(self,vocab_size,padding_idx=0):
        super(ReviewAnalyzeModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=config.EMBEDDING_DIM,padding_idx=padding_idx)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_SIZE,1)

    def forward(self,x):
        # x.shape [batch_size, seq_len]
        embedding = self.embedding(x) # [batch_size, seq_len, embedding_dim]
        rnn_out, _ = self.rnn(embedding) # lstm_out.shape [batch_size,seq_len, hidden_dim]
        last_hidden = rnn_out[:,-1,:] # last_hidden.shape [batch_size, hidden_dim]
        out = self.linear(last_hidden) # out.shape [batch_size, 1]
        return out.squeeze(1) # out.shape [batch_size]