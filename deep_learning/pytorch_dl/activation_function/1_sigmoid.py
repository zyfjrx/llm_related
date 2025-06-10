import torch
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10,1000,requires_grad=True)
y = torch.sigmoid(x)

# 创建子图
fig, ax = plt.subplots(1,2)
fig.set_size_inches(12,4)
# 画出sigmoid(x)函数图像
ax[0].plot(x.detach(), y.detach(),color='purple')
ax[0].set_title('sigmoid(x)')
ax[0].axhline(y=1, color='gray',alpha=0.5,linewidth=1)
ax[0].axhline(y=0.5, color='gray',alpha=0.5,linewidth=1)
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["left"].set_position("zero")
ax[0].spines["bottom"].set_position("zero")

# 画出导函数
y.sum().backward()
ax[1].plot(x.detach(), x.grad,color='purple')
ax[1].set_title("sigmoid'(x)")
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["left"].set_position("zero")
ax[1].spines["bottom"].set_position("zero")
plt.show()