import pandas as pd
from sklearn.linear_model import LinearRegression,SGDRegressor # 解析法和梯度下降法求解线性回归模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = pd.read_csv("/Users/zhangyf/PycharmProjects/train/llm_related/machine_learning/data/advertising.csv")

# 数据预处理
data.drop(data.columns[0], axis=1, inplace=True)
data.dropna(inplace=True)

# 划分数据集
X = data.drop(columns=['Sales'],axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 特征工程 (标准化)
transformer = StandardScaler()
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

# 正规方程法
regressor_normal = LinearRegression()
regressor_normal.fit(X_train, y_train)


print("正规方程法解得模型系数:", regressor_normal.coef_)
print("正规方程法解得模型偏置:", regressor_normal.intercept_)


# 梯度下降法
regressor_sgd = SGDRegressor()
regressor_sgd.fit(X_train, y_train)
print("梯度下降法解得模型系数:", regressor_sgd.coef_)
print("梯度下降法解得模型偏置:", regressor_sgd.intercept_)

# 测试
y_pred1 = regressor_sgd.predict(X_test)
y_pred2 = regressor_normal.predict(X_test)
print("正规方程法均方误差",mean_squared_error(y_test,y_pred2))
print("梯度下降法均方误差",mean_squared_error(y_test,y_pred1))

