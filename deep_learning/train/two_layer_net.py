import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.gradient import numerical_gradient

from common.functions import softmax, cross_entropy_error, sigmoid


class TwoLayerNet:
    def __init__(self, input_size=784, hidden_size=20, output_size=10, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.forward(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

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
