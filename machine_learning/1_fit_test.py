import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 划分训练和测试集
from sklearn.linear_model import LinearRegression # 线性回归模型
from sklearn.metrics import mean_squared_error # 均方误差损失函数

'''
机器学习训练模型步骤
1.读取数据
2.划分训练集和测试集
3.定义损失函数和模型
4.训练模型
5.预测结果，计算误差（测试误差）
'''
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STKaiti', 'Arial Unicode MS']  # 优先使用苹方，其次是楷体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def polynomial(x, degree):
    """构成多项式，返回 [x^1,x^2,x^3,...,x^n]"""
    return np.hstack([x**i for i in range(1, degree + 1)])


# 1.读取数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)
fig, ax = plt.subplots(1,3,figsize=(15,4))
ax[0].plot(X, y)
ax[1].plot(X, y)
ax[2].plot(X, y)
# plt.show()

# 2.划分训练集和测试集
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.定义损失函数和模型
model = LinearRegression()

# 4.训练模型
model.fit(trainX, trainY)

# 5.预测结果，计算误差（测试误差）
y_pred = model.predict(testX)
test_loss = mean_squared_error(testY, y_pred)
train_loss = mean_squared_error(trainY, model.predict(trainX))

# 画出拟合曲线，并标出误差
ax[0].plot(X, model.predict(X))
ax[0].text(-3, 1, f"测试集均方误差：{test_loss:.4f}")
ax[0].text(-3, 1.3, f"训练集均方误差：{train_loss:.4f}")



# 拟合
X_train2 = polynomial(trainX, 5)
X_test2 = polynomial(testX, 5)
model.fit(X_train2, trainY)
y_pred2 = model.predict(X_test2)
test_loss2 = mean_squared_error(testY, y_pred2)
train_loss2 = mean_squared_error(trainY, model.predict(X_train2))
# 画出拟合曲线，并标出误差
ax[1].plot(X, model.predict(polynomial(X,5)),'r')
ax[1].text(-3, 1, f"测试集均方误差：{test_loss2:.4f}")
ax[1].text(-3, 1.3, f"训练集均方误差：{train_loss2:.4f}")


# 过拟合
X_train3 = polynomial(trainX, 50)
X_test3 = polynomial(testX, 50)
model.fit(X_train3, trainY)
y_pred3 = model.predict(X_test3)
test_loss3 = mean_squared_error(testY, y_pred3)
train_loss3 = mean_squared_error(trainY, model.predict(X_train3))
# 画出拟合曲线，并标出误差
ax[2].plot(X, model.predict(polynomial(X,50)),'r')
ax[2].text(-3, 1, f"测试集均方误差：{test_loss3:.4f}")
ax[2].text(-3, 1.3, f"训练集均方误差：{train_loss3:.4f}")
plt.show()