import numpy as np


# 损失函数
def J(beta):
    return np.sum((X @ beta - y) ** 2, axis=0).reshape(-1, 1) / n


# 计算梯度
def gradient(beta):
    return X.T @ (X @ beta - y) / n * 2


X = np.array([[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]])  # 自变量，每周学习时长
y = np.array([[55], [65], [70], [75], [85], [50], [60], [72], [80], [58]]) # 因变量，数学考试成绩

n = X.shape[0]  # 样本数
X = np.hstack([np.ones((n, 1)), X])  # X添加一列1，与偏置项相乘
print(X.shape)
beta = np.array([[1], [1]])  # 初始化参数
alpha = 1e-2  # 学习率
epoch = 1000  # 迭代次数

while (epoch := epoch - 1) >= 0:
    # 计算梯度
    grad = gradient(beta)
    # 更新参数
    beta = beta - alpha * grad
    # 每迭代10轮打印一次参数值和损失函数值
    if epoch % 10 == 0:
        print(f'beta = {beta.reshape(-1)}\tJ = {J(beta)}')

