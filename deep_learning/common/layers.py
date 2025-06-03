
# ReLU
class ReLU:
    def __init__(self):
        # 记录哪些 x <= 0
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
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