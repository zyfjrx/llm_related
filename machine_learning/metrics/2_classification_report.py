from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 逻辑回归 （分类模型）
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
# 增加AUC指标
# 计算预测类别概率值
y_pred_proba = model.predict_proba(X_test)[:,1]
# 计算AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(auc_score)

print(report)