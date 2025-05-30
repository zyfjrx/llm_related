import numpy as np

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
    x = x - np.max(x) # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))

# 输入为矩阵 N x C
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x,axis=0) # 防止溢出
        y = np.exp(x) / np.sum(np.exp(x),axis=0)
        return y.T
    x = x - np.max(x)  # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))

def identity(x):
    return x

if __name__ == '__main__':
    x = np.array([1, 2, 3, 10, 20])
    X = np.array([
        [1, 2, 3],  # 第0行
        [4, 5, 6],  # 第1行
        [7, 8, 9],  # 第2行
        [12, 28, 99]  # 第2行
    ])
    print(softmax(X))