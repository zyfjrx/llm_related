# 感知机 神经元传递信号 ，神经元分层级连实现复杂问题，神经网络
# 与门
def AND0(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    result = w1 * x1 + w2 * x2
    if result <= theta:
        return 0
    else:
        return 1


# 测试与门
print(AND0(0, 0))
print(AND0(0, 1))
print(AND0(1, 0))
print(AND0(1, 1))

import numpy as np


def AND(x1, x2):
    X = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    result = np.sum(w * X) + b
    if result <= 0:
        return 0
    else:
        return 1

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))

print("*"*100)


def NAND(x1, x2):
    X = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    result = np.sum(w * X) + b
    if result <= 0:
        return 0
    else:
        return 1
print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))
print("*"*100)


def OR(x1, x2):
    X = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    result = np.sum(w * X) + b
    if result <= 0:
        return 0
    else:
        return 1
print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))
print("*"*100)
# 异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)
print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))