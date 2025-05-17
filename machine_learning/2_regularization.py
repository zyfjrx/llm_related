import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分训练和测试集
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归模型
from sklearn.metrics import mean_squared_error  # 均方误差损失函数

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
    return np.hstack([x ** i for i in range(1, degree + 1)])


# 1.读取数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)

# 画出散点图
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].plot(X, y,c='y')
ax[0,1].plot(X, y,c='y')
ax[0,2].plot(X, y,c='y')
# plt.show()

# 2.划分训练集和测试集
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)


#
X_train = polynomial(trainX, 20)
X_test = polynomial(testX, 20)
model = LinearRegression()
model.fit(X_train, trainY)
y_pred1 = model.predict(X_test)
test_loss1 = mean_squared_error(testY, y_pred1)
# 画出拟合曲线，并标出误差
ax[0,0].plot(X, model.predict(polynomial(X, 20)), 'r')
ax[0,0].text(-3, 1, f"测试集均方误差：{test_loss1:.4f}")
ax[1,0].bar(np.arange(20), model.coef_.reshape(-1),color='b')


# L1正则化 Lasso回归
ridge = Lasso(alpha=0.01)

ridge.fit(X_train, trainY)
y_pred2 = ridge.predict(X_test)
test_loss2 = mean_squared_error(testY, y_pred2)
ax[0,1].plot(X, ridge.predict(polynomial(X, 20)), 'r')
ax[0,1].text(-3, 1, f"测试集均方误差：{test_loss2:.4f}")
ax[0,1].text(-3, 1.3, "Lasso回归")
ax[1,1].bar(np.arange(20), ridge.coef_.reshape(-1), color='b')


# L2正则化 Lasso回归
ridge = Ridge(alpha=1)

ridge.fit(X_train, trainY)
y_pred3 = ridge.predict(X_test)
test_loss3 = mean_squared_error(testY, y_pred3)
ax[0,2].plot(X, ridge.predict(polynomial(X, 20)), 'r')
ax[0,2].text(-3, 1, f"测试集均方误差：{test_loss3:.4f}")
ax[0,2].text(-3, 1.3, "Ridge回归")
ax[1,2].bar(np.arange(20), ridge.coef_.reshape(-1), color='b')
plt.show()