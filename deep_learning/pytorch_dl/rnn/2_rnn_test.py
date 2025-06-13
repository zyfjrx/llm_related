import torch

rnn = torch.nn.RNN(input_size=8,hidden_size=16,num_layers=2)
input = torch.randn(2, 3, 8)
hx = torch.randn(2, 3, 16)
output, hidden = rnn(input, hx)
print(output.shape)
print(hidden.shape)