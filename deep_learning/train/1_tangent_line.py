import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.gradient import numerical_diff
import numpy as np
import matplotlib.pyplot as plt


# 目标函数 f(x) = 0.01x ** 2 + 0.1x
def f(x):
    return 0.01 * x ** 2 + 0.1 * x


def tangent_function(f, x):
    # 计算斜率
    a = numerical_diff(f, x)
    print("切线斜率：",a)
    # 计算截距
    b = f(x) - a * x
    # print(b)
    return lambda x: a * x + b

# 原始曲线
x = np.arange(0.0, 20, 0.1)
y = f(x)

# 切线
tf = tangent_function(f, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()