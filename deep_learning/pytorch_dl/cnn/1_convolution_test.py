import torch
import torch.nn as nn
import matplotlib.pyplot as plt

img = plt.imread("../data/duck.jpg")
print(img.shape)
input = torch.tensor(img).permute(2, 0, 1).float()
print(f"输入特征图维度：{input.shape}")

conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=9, stride=3, padding=0,groups=3)
out = conv(input)
print(conv.weight.shape)
print(f"输出特征图维度：{out.shape}")

out = (out - torch.min(out)) / (torch.max(out) - torch.min(out)) * 255
# out = torch.clamp(out.int(), 0, 255)  # 限制输出在0到255之间
out = out.int().permute(1, 2, 0).detach().cpu().numpy()
print(out.shape)
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(img)
ax[1].imshow(out)
plt.show()