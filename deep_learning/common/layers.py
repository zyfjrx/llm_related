import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.functions import sigmoid,softmax,cross_entropy_error
import numpy as np

# ReLU
class ReLU:
    def __init__(self):
        # 记录哪些 x <= 0
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = sigmoid(x)
        self.y = y
        return y

    def backward(self, dout):
        dx = dout * self.y(1.0 - self.y)
        return dx


class Affine:
    def __init__(self, W, b):
        # 保存权重和偏置参数
        self.W = W
        self.b = b
        # 保存本层输入的X
        self.X = None
        self.X_original_shape = None

        # 权重和偏置参数的导数
        self.dW = None
        self.db = None
    def forward(self, X):
        self.X_original_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        Y = np.dot(self.X, self.W) + self.b
        return Y
    def backward(self, dout):
        dX = np.dot(dout,self.W.T)
        # 计算参数的梯度
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        return dX.reshape(*self.X_original_shape)


# 输出层Softmax+Loss
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None # 损失值
    def forward(self, X, t):
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 标签是独热编码
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        # 标签是分类编号就将预测值对应
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx






if __name__ == '__main__':
    import numpy as np

    x2 = np.array([
        [0.1, -0.2, 0.3],  # 第0行
        [0.21, 0.22, -0.03],  # 第1行
        [0.11, 0.32, -0.63],  # 第2行
        [-0.41, 0.2, 0.13]  # 第2行
    ])
    dout = np.array([
        [0.1, 0.2, 0.3],  # 第0行
        [0.21, 0.22, 0.03],  # 第1行
        [0.11, 0.32, 0.63],  # 第2行
        [0.41, 0.2, 0.13]  # 第2行
    ])
    relu = ReLU()
    out = relu.forward(x2)
    y = relu.backward(dout)
    print(y)
