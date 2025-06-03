import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.gradient import numerical_gradient

# 目标函数
def f(x):
    return x[0] ** 2 + x[1] ** 2

def gradient_descent(f, init_x, lr=0.01,steps=100):
    x = init_x
    x_history = []
    for i in range(steps):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x = x - lr * grad
    return x, np.array(x_history)

if __name__ == '__main__':

    init_x = np.array([-3.0, 4.0])
    lr = 0.1
    steps = 15
    x, x_history = gradient_descent(f, init_x, lr, steps)
    plt.plot(x_history[:,0],x_history[:,1],'o')
    plt.plot([-5,5],[0,0],'--b')
    plt.plot([0,0],[-5,5],'--b')
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
