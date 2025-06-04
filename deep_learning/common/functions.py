import numpy as np
import torch


# 激活函数

# 阶跃函数
def step_function0(x):
    if x >= 0:
        return 1
    else:
        return 0


def step_function(x):
    return np.array(x >= 0, dtype=int)


# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# relu
def relu(x):
    return np.maximum(0, x)


# softmax
# 输入为向量
def softmax0(x):
    x = x - np.max(x)  # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))


# 输入为矩阵 N x C
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # 防止溢出
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)  # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))


def identity(x):
    return x


# 损失函数
# MSE
def mean_squared_error(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)


# 交叉商损失
def cross_entropy_error(y_pred, y_true):
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, y_pred.size)
        y_true = y_true.reshape(1, y_true.size)
    if y_true.size == y_pred.size:
        y_true = y_true.argmax(axis=1)

    n = y_pred.shape[0]
    return -np.sum(np.log(y_pred[np.arange(n), y_true] + 1e-7)) / n
# # 2. 交叉熵误差
# def cross_entropy_error(y, t):
#     # 对于一维情况，直接转换为二维
#     if y.ndim == 1:
#         y = y.reshape(1, y.size)
#         t = t.reshape(1, t.size)
#
#     # t 是独热编码表示，转换为正确类别标签的索引
#     if t.size == y.size:
#         t = t.argmax(axis=1)
#
#     n = y.shape[0]
#     return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n


if __name__ == '__main__':
    x1 = np.array([0.1, 0.2, 0.3])
    y1 = np.array([2])
    x2 = np.array([
        [0.1, 0.2, 0.3],  # 第0行
        [0.21, 0.22, 0.03],  # 第1行
        [0.11, 0.32, 0.63],  # 第2行
        [0.41, 0.2, 0.13]  # 第2行
    ])
    y2 = np.array([
        [0, 0, 1],  # 第0行
        [0, 1, 0],  # 第1行
        [0, 0, 1],  # 第2行
        [1, 0, 0]  # 第2行
    ])
    out = sigmoid(y2)
    print(out)
    # for i , y in enumerate(y2):
    #     print(i, y)
    # loss = cross_entropy_error(x1, y1)
    # print(loss)
    # print(softmax(X))
    # print(x.argmax())
