import  torch
import torch.nn as nn
from torchsummary import summary

x = torch.randn(10,3).to("cuda")

model = nn.Sequential(
    nn.Linear(3,4),
    nn.Tanh(),
    nn.Linear(4,4),
    nn.ReLU(),
    nn.Linear(4,2),
    nn.Softmax(dim=1)
).to("cuda")

# 参数初始化
def init_weights(m):
    # 对Linear层进行初始化
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.01)
model.apply(init_weights)

out = model(x)
print("模型参数：\n", model.state_dict())
summary(model, input_size=(3,),batch_size=1,device="cuda")