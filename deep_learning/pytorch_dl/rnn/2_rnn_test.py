import torch

rnn = torch.nn.RNN(input_size=8,hidden_size=16,num_layers=2,batch_first=True)
input = torch.randn(3, 4, 8)
hx = torch.randn(2, 3, 16)
output, hidden = rnn(input, hx)
print(output.shape)
print(hidden.shape)