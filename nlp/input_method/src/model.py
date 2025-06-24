import torch
import config
import torch.nn as nn
import dataset
class InputMethodModel(nn.Module):
    def __init__(self,vocab_size):
        super(InputMethodModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_SIZE,num_layers=2,batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_SIZE,vocab_size)

    def forward(self,x):
        # [batch_size,len]
        embedd = self.embedding(x) # [batch_size,len,embedding_dim]
        output, _ = self.rnn(embedd)  # output.shape[batch_size,len,hidden_dim],hidden.shape[num_layers,len,hidden_dim]
        last_hidden = output[:,-1,:]  # [batch_size,hidden_dim]
        output = self.linear(last_hidden)  # output.shape[batch_size,vocab_size]
        return output


if __name__ == '__main__':
    dataloader = dataset.get_dataloader(train=False)
    for input_tensor, target_tensor in dataloader:
        model = InputMethodModel(21199)
        output = model(input_tensor)