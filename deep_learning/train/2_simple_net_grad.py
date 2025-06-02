import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.gradient import numerical_gradient
from common.functions import softmax, cross_entropy_error
import numpy as np


# 定义一个单层网络
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    # 前向传输
    def forward(self, x):
        a = np.dot(x, self.W)
        return softmax(a)

    # 定义损失函数
    def loss(self, x, t):
        y = self.forward(x)  # 预测输出
        loss = cross_entropy_error(y, t)
        return loss


# 主流程
if __name__ == '__main__':
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    net = SimpleNet()
    loss = lambda w: net.loss(x, t)
    dw = numerical_gradient(loss, net.W)
    print(dw)

