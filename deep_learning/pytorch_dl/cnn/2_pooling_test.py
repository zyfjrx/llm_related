import torch
import torch.nn as nn
import matplotlib.pyplot as plt

img = plt.imread("../data/duck.jpg")
print(img.shape)
input = torch.tensor(img).permute(2, 0, 1).float()
print(f"输入特征图维度：{input.shape}")

conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=9, stride=3, padding=0)
# 池化层增大感受野，便于提取图像轮廓结构信息，
pool = nn.MaxPool2d(kernel_size=6, stride=6,padding=1)
out1 = conv(input)
print(f"卷积后输出特征图形状：{out1.shape}")
out2 = pool(out1)
print(conv.weight.shape)
print(f"池化特征图维度：{out2.shape}")

out1 = (out1 - torch.min(out1)) / (torch.max(out1) - torch.min(out1)) * 255
out2 = (out2 - torch.min(out2)) / (torch.max(out2) - torch.min(out2)) * 255
# out = torch.clamp(out.int(), 0, 255)  # 限制输出在0到255之间
out1 = out1.int().permute(1, 2, 0).detach().cpu().numpy()
out2 = out2.int().permute(1, 2, 0).detach().cpu().numpy()
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(img)
ax[1].imshow(out1)
ax[2].imshow(out2)
plt.show()