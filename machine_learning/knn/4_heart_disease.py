import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer  # 列转换，做特征转换
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 加载数据集
heart_disease_data = pd.read_csv("../data/heart_disease.csv")

# 1.数据清洗
heart_disease_data.dropna(inplace=True)
# heart_disease_data.info()
# print(heart_disease_data.head())

# 2.数据集划分
# 定义特征
X = heart_disease_data.drop("是否患有心脏病", axis=1)
y = heart_disease_data["是否患有心脏病"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 特征工程
# 数值型特征
numerical_features = ["年龄", "静息血压", "胆固醇", "最大心率", "运动后的ST下降", "主血管数量"]
# 类别型特征
categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]

# 创建列转换器
transformer = ColumnTransformer(
    transformers=[
        # (名称，操作，特征列表）
        # 对数值型特征进行标准化
        ("num", StandardScaler(), numerical_features),
        # 对类别型特征进行独热编码，使用drop="first"避免多重共线性
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        # 二元特征不进行处理
        ("binary", "passthrough", binary_features),
    ]
)

# 执行特征转换
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)
print(X_train.shape, X_test.shape)

# 4.创建模型
# 使用KNN进行二分类
# knn = KNeighborsClassifier(n_neighbors=3)
knn = KNeighborsClassifier()
knn.kneighbors()
# 网格搜索确定最优超参数
param_grid = {"n_neighbors": list(range(1, 11))}
knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10)

knn.fit(X_train, y_train)

# 5.测试、评估
# print(knn.score(X_test, y_test))

# 6.模型保存
# joblib.dump(knn, "knn_heart_disease.joblib")

# 加载保存的模型
# knn_loaded = joblib.load("knn_heart_disease.joblib")
# print(knn_loaded.score(X_test, y_test))

# 7.预测
# y_pred = knn_loaded.predict(X_test[100:101])
# print(y_pred, y_test.iloc[100])

# 8.打印验证结果
pd.set_option("display.max_columns", None)
# print(pd.DataFrame(knn.cv_results_))
print(knn.best_params_)
print(knn.best_score_)
print(knn.best_estimator_)

# 用最佳模型做预测
knn = knn.best_estimator_
print(knn.score(X_test, y_test))

