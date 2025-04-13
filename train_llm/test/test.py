import torch
import numpy as np

# angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
# print(angle)
# print(torch.ones_like(angle))
# z = torch.polar(torch.ones_like(angle), angle)
# print(z)

import torch

# 创建一个初始值为0的长度为5的张量(储蓄罐)
target = torch.zeros(5)

# 要添加的数据(硬币)
src = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 指定添加的位置(第0,2,3,1位置)
index = torch.tensor([0, 2, 3, 1])

# 执行scatter_add_
target.scatter_add_(0, index, src)

print(target)  # 输出: tensor([1., 4., 2., 3., 0.])

