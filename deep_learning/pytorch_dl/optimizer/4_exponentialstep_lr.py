import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler


X = torch.tensor([-7, 2], dtype=torch.float, requires_grad=True)
W = torch.tensor([0.05, 1.0], dtype=torch.float, requires_grad=True)

# 定义二元函数 f(x1,x2) = 0.05 * x1^2 + x2^2
def f(X):
    return W.dot(X ** 2)

lr = 0.8
n_iter = 500

# 定义优化器
optimizer = torch.optim.SGD([X], lr=lr)
# 学习率衰减，指定间隔衰减
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# 迭代更新参数
X_arr = X.detach().numpy().copy()
lr_list = []
for iter in range(n_iter):
    y = f(X)
    y.backward()
    optimizer.step()
    optimizer.zero_grad()
    X_arr = np.vstack((X_arr, X.detach().numpy()))
    # 记录学习率变化
    lr_list.append(optimizer.param_groups[0]['lr'])
    # 执行学习率衰减
    scheduler.step()

# 画图
fig,ax = plt.subplots(1,2,figsize=(12,4))

x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid ** 2 + x2_grid ** 2
ax[0].contour(x1_grid, x2_grid, y_grid, colors='gray', levels=30)
ax[0].plot(X_arr[:, 0], X_arr[:, 1], color='red')

ax[1].plot(lr_list,'k')

plt.show()
