import torch
import torch.nn as nn
from torchsummary import summary

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = nn.Linear(3, 4)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(4, 4)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.softmax(x, dim=1)
        return x

x = torch.randn(10,3).to("cuda")
model = Model().to("cuda")
out = model(x)
print(out)
print(out.shape)
print("*"*100)
print("模型参数:")
for name, param in model.named_parameters():
    print(name,param)
    print("*"*100)

# 使用state_dict()查看各层参数
print("模型参数：\n", model.state_dict())

# 统计信息
summary(model,input_size=(3,),batch_size=20,device="cuda")
