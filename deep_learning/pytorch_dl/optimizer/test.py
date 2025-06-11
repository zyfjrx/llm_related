import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
a_list = [0.2, 0.5, 0.8]  # 不同底数

plt.figure(figsize=(10, 6))
for a in a_list:
    y = a ** x
    plt.plot(x, y, label=f'y = {a}^x')

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('y = a^x(0 < a < 1)')
plt.legend()
plt.grid(True)
plt.show()