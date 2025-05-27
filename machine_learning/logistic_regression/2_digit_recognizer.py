import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib
# 加载数据集
digit = pd.read_csv("../data/train.csv")

# 图片测试
# plt.imshow(digit.iloc[15,1:].values.reshape(-1,28), cmap="Greys")
# plt.show()

# 划分训练集和测试集
X = digit.drop(columns="label",axis=1)
y = digit["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

# 特征转换，归一化处理
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建模型并训练
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# 模型保存
# joblib.dump(model, "digit_recognizer.joblib")

# 加载模型
model = joblib.load("digit_recognizer.joblib")

# 测试准确率
print(model.score(X_test, y_test))

plt.imshow(digit.iloc[123,1:].values.reshape(-1,28), cmap="Greys")
plt.show()

print(model.predict(digit.iloc[123,1:].values.reshape(1,-1)))