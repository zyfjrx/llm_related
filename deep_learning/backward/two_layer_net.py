import numpy as np
import sys
import os

from deep_learning.common.layers import Affine, ReLU, SoftmaxWithLoss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.gradient import numerical_gradient
from common.functions import softmax, cross_entropy_error, sigmoid
from common.layers import *
from collections import OrderedDict # 有序字典



class TwoLayerNet:
    def __init__(self, input_size=784, hidden_size=20, output_size=10, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 按照顺序生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def forward(self, x):
        # 遍历每个层，调用forward方法
        for layer in self.layers.values():
            x = layer.forward(x)
        return x



    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def loss(self, x, t):
        y = self.forward(x)
        return self.lastLayer.forward(y, t)

    # 反向传播计算梯度
    def gradient(self, x, t):
        # 前向传播计算中间值
        self.loss(x, t)
        # 反向传播
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 提取各Affine层参数梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads



    # 计算梯度
    def numerical_gradient(self, x, t):
        loss = lambda w: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss,self.params['W1'])
        grads['b1'] = numerical_gradient(loss,self.params['b1'])
        grads['W2'] = numerical_gradient(loss,self.params['W2'])
        grads['b2'] = numerical_gradient(loss,self.params['b2'])
        return grads


# if __name__ == '__main__':
#     x2 = np.array([
#         [0.1, 0.2, 0.3],  # 第0行
#         [0.21, 0.22, 0.03],  # 第1行
#         [0.11, 0.32, 0.63],  # 第2行
#         [0.41, 0.2, 0.13]  # 第2行
#     ])
#     # x2 = np.argmax(x2, axis=1)
#     mask = np.random.choice(4, size=2)
#     print(mask)
#     print(x2[mask])
