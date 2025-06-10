import torch
from torch import nn,optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(5,3)
        self.linear.weight.data = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.10, 1.1, 1.2],
                [1.3, 1.4, 1.5],
            ]
        ).T
        self.linear.bias.data = torch.tensor([1.0, 2.0, 3.0])

    def forward(self, x):
        x = self.linear(x)
        return x

# 创建数据
# 输入值
X = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float)
# 目标值
target = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float)

# 实例化模型
model = Model()

# 前向传播
out = model(X)
print(out.shape)
# 计算损失
loss = nn.MSELoss()
loss_value = loss(out, target)
loss_value.backward()

# 更新参数
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.step()
optimizer.zero_grad()

for i in model.state_dict():
    print(i,model.state_dict()[i])
    print()