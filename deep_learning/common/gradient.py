import numpy as np

# 数值微分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)  # 中心差分

# 数值微分梯度 f为多元函数，x[x1,x2,x3,.....xn]
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # 初始梯度向量

    for i in range(x.shape[0]):
        tmp_x = x[i]
        x[i] = tmp_x + h
        fxh1 = f(x) # f(x + h)
        x[i] = tmp_x - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp_x
    return grad

def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient(f, x)
    else:
        grad = np.zeros_like(x)
        for i, x in enumerate(x):
            grad[i] = _numerical_gradient(f, x)
        return grad


if __name__ == '__main__':
    def f(x):
        return x[0] ** 2 + x[0] * x[1] + x[1] ** 2
    grad = numerical_gradient(f,np.array([1,1],dtype=float))
    print(grad)