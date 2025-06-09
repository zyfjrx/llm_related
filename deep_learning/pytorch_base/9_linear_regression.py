import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset  # 数据加载器和数据集
import matplotlib.pyplot as plt

# 1.构建数据集
X = torch.randn(100, 1)
w = torch.tensor(2.5)
b = torch.tensor(5.2)
noise = torch.randn(100, 1) * 0.5
y = w * X + b + noise
# print(y.shape)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10,shuffle=True)
# print(len(dataloader))

# 2.构建模型
model = nn.Linear(1, 1)

# 3.损失函数和优化器
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4.模型训练
loss_list = []
for epoch in range(100):
    total_loss = 0
    iter_num = 0
    for x_train, y_train in dataloader:
        # 前线传播预测
        y_pred = model(x_train)
        # 计算损失
        loss_value = loss(y_pred, y_train)
        total_loss += loss_value.item()
        iter_num += 1
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss_value.backward()
        # 更新参数
        optimizer.step()

    loss_list.append(total_loss / iter_num)
print(model.weight, model.bias)  # 打印权重和偏置
plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
plt.scatter(X,y)
# y_pred = model(X).cpu().detach().numpy()
y_pred = model.weight.item() * X + model.bias.item()
plt.plot(X,y_pred,color='red')
plt.show()



