import torch.nn as nn
import config


class ReviewAnalyzeModel(nn.Module):
    def __init__(self,vocab_size,padding_idx=0):
        super(ReviewAnalyzeModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=config.EMBEDDING_DIM,padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_SIZE,num_layers=2,batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_SIZE,1)

    def forward(self,x):
        # x.shape [batch_size, seq_len]
        embedding = self.embedding(x) # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedding) # lstm_out.shape [batch_size,seq_len, hidden_dim]
        last_hidden = lstm_out[:,-1,:] # last_hidden.shape [batch_size, hidden_dim]
        out = self.linear(last_hidden) # out.shape [batch_size, 1]
        return out.squeeze(1) # out.shape [batch_size]