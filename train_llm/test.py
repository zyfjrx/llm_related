import torch
import numpy as np

angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
print(angle)
print(torch.ones_like(angle))
z = torch.polar(torch.ones_like(angle), angle)
print(z)

