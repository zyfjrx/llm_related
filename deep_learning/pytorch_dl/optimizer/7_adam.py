import torch
import numpy as np
import matplotlib.pyplot as plt

X = torch.tensor([-7, 2], dtype=torch.float, requires_grad=True)
W = torch.tensor([0.05, 1.0], dtype=torch.float, requires_grad=True)


# 定义二元函数 f(x1,x2) = 0.05 * x1^2 + x2^2
def f(X):
    return W.dot(X ** 2)


# 定义函数：梯度下降方式迭代，更新X 并保存X的变化列表
def grad_desc(X, optimizer, n_iter):
    X_arr = X.detach().numpy().copy()
    for iter in range(n_iter):
        y = f(X)
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
        X_arr = np.vstack((X_arr, X.detach().numpy()))
    return X_arr


# 手动实现adam迭代过程
def adam(X, lr, n_iter,betas=(0.9, 0.999)):
    X_arr = X.detach().numpy().copy()
    V = torch.zeros_like(X)
    H = torch.zeros_like(X)
    for iter in range(n_iter):
        grad = 2 * X * W
        V = betas[0] * V + (1.0 - betas[0]) * grad
        H = betas[1] * H + (1.0 - betas[1]) * grad ** 2
        V_hat = V / (1-betas[0] ** (iter+1))
        H_hat = H / (1-betas[1] ** (iter+1))
        X.data -= lr / (torch.sqrt(H_hat) + 1e-7) * V_hat
        X_arr = np.vstack((X_arr, X.detach().numpy()))
    return X_arr


lr = 0.1
n_iter = 500

# 梯度下降寻找最小值
# SGD
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], lr=lr)
X_arr = grad_desc(X_clone, optimizer, n_iter)
plt.plot(X_arr[:, 0], X_arr[:, 1], 'r')

# Adam
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([X_clone], lr=lr,betas=(0.9, 0.999))
X_arr2 = grad_desc(X_clone, optimizer, n_iter)
plt.plot(X_arr2[:, 0], X_arr2[:, 1], 'b')

# RMSProp手动实现
X_clone = X.clone().detach().requires_grad_(True)
X_arr3 = adam(X_clone, lr, n_iter)
plt.plot(X_arr3[:, 0], X_arr3[:, 1], color='orange', linestyle='--')

# 等高线
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid ** 2 + x2_grid ** 2
plt.contour(x1_grid, x2_grid, y_grid, colors='gray', levels=30)
plt.legend(['SGD', 'Adam','Manual Adam'])

plt.show()
