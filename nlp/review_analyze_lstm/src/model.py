import torch.nn as nn
import config
import torch

class ReviewAnalyzeModel(nn.Module):
    def __init__(self,vocab_size,padding_idx=0):
        super(ReviewAnalyzeModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=config.EMBEDDING_DIM,padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_SIZE,num_layers=config.NUM_LAYERS,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(2 * config.HIDDEN_SIZE,1)

    def forward(self,x):
        # x.shape [batch_size, seq_len]
        embedding = self.embedding(x) # [batch_size, seq_len, embedding_dim]
        # n层单向
        # lstm_out, _ = self.lstm(embedding) # lstm_out.shape [batch_size,seq_len, hidden_dim]
        # last_hidden = lstm_out[:,-1,:] # last_hidden.shape [batch_size, hidden_dim]

        # n层双向
        _, (h_n, _) = self.lstm(embedding)
        # hn.shape [num_layers * num_directions, batch_size, hidden_dim]
        # last_hidden_forward.shape [batch_size, hidden_dim]
        last_hidden_forward = h_n[-2]
        # last_hidden_backward.shape [batch_size, hidden_dim]
        last_hidden_backward = h_n[-1]
        # last_hidden.shape [batch_size, hidden_dim * 2]
        last_hidden = torch.cat((last_hidden_forward, last_hidden_backward), dim=-1)
        out = self.linear(last_hidden) # out.shape [batch_size, 1]
        return out.squeeze(1) # out.shape [batch_size]