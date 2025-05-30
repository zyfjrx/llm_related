import torch

x = torch.randn(4, 10, 8, 3)
print(x.shape)
print(x)
xx = x.new_empty(4, 10, 8, 6)
print(xx.shape)
print(xx)