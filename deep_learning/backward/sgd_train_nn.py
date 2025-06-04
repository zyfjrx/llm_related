import numpy as np
from two_layer_net import TwoLayerNet
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.load_data import get_data
import math
import matplotlib.pyplot as plt
# 加载数据集
x_train, x_test, t_train, t_test = get_data()
print(x_train.shape)
print(x_test.shape)
print(t_train.shape)
print(t_test.shape)

# 创建神经网络模型
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 设置超参数
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = math.ceil(x_train.shape[0] / batch_size) # 每轮迭代次数

iters_num = 10000 # 迭代总次数
learning_rate = 0.1

train_loss_list = [] # 训练误差
train_acc_list = [] # 训练准确度
test_acc_list = [] # 测试准确度

# 梯度下降法，迭代训练模型，计算参数
for i in range(iters_num):
    # 4.1 随机选取batch_size个训练数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 4.2 计算当前参数下的梯度值
    grad = network.gradient(x_batch, t_batch)

    # 4.3 遍历梯度字典，更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 4.4 计算当前训练损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # print(f'iter {i}, loss {loss}')

    # 4.5 计算每个轮次准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Train loss:{loss}, Train acc:{train_acc} ,Test acc:{test_acc}")

# 绘制图形
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
x = np.arange(len(train_acc_list))
ax1.plot(x, train_acc_list, label='train acc')
ax1.plot(x, test_acc_list, label='test acc', linestyle='--')
ax1.set_xlabel("epochs")
ax1.set_ylabel("accuracy")
ax1.set_ylim(0, 1.0)
ax1.legend(loc='lower right')
x2 = np.arange(len(train_loss_list))
ax2.plot(x2, train_loss_list, label='train loss')
ax2.set_xlabel("iter")
ax2.set_ylabel("loss")
plt.show()
