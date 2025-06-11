import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 针对数值型和类别特征的处理操作
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  # 处理缺失值
from torch.utils.data import Dataset, DataLoader, TensorDataset


# 1.读取数据,返回数据集
def create_dataset():
    # 从文件读取数据
    data = pd.read_csv("./data/house_prices.csv")

    # 特征工程：去除无关特征（特征选择）
    data.drop(['Id'], axis=1, inplace=True)

    # 划分特征和目标值
    X = data.drop(['SalePrice'], axis=1)
    y = data['SalePrice']

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)

    # 特征预处理
    # 筛选数值特征和类别特征
    numerical_features = X.select_dtypes(exclude="object").columns
    categorical_features = X.select_dtypes(include="object").columns
    # 数值型特征先用平均值填充缺省值，在进行标准化
    numerical_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='mean')),
            ('std', StandardScaler())
        ]
    )
    # 类别特征独热编码
    categorical_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='constant', fill_value='NaN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    # 构建列转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

    # 列转换
    x_train = pd.DataFrame(preprocessor.fit_transform(x_train).toarray(),
                           columns=preprocessor.get_feature_names_out())
    x_test = pd.DataFrame(preprocessor.transform(x_test).toarray(),
                          columns=preprocessor.get_feature_names_out())
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())
    return train_dataset, test_dataset, x_train.shape[1]


train_dataset, test_dataset, features = create_dataset()
# print(features)

# 搭建模型
model = nn.Sequential(
    nn.Linear(in_features=features, out_features=128),
    nn.BatchNorm1d(num_features=128),  # 正则化
    nn.ReLU(),
    nn.Dropout(0.2),  # 正则化
    nn.Linear(in_features=128, out_features=1),
)
# # 模型参数统计信息
# summary(model, input_size=(features,), batch_size=10, device='cpu')


# 损失函数
def log_rmse(pred, target):
    pred.squeeze_()
    mse = nn.MSELoss()
    pred = torch.clamp(pred, min=1, max=float("inf"))
    return torch.sqrt(mse(torch.log(pred), torch.log(target)))


def train(model, train_dataset, test_dataset, lr, epochs,batch_size,device):
    # linear权重初始化
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)
    model.apply(init_weights)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []
    test_loss_list = []
    for epoch in range(epochs):
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_total_loss = 0
        for X, target in train_loader:
            X,target = X.to(device), target.to(device)
            output = model(X)
            loss_value = log_rmse(output, target)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_total_loss += loss_value.item()
        train_loss_avg = train_total_loss/len(train_loader)
        train_loss_list.append(train_loss_avg)

        # 验证
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_total_loss = 0
        with torch.no_grad():
            for X, target in test_loader:
                X, target = X.to(device), target.to(device)
                output = model(X)
                loss_value = log_rmse(output, target)
                test_total_loss += loss_value.item()
            test_loss_avg = test_total_loss/len(test_loader)
            test_loss_list.append(test_loss_avg)
        # 打印输出
        print(f"train loss: {train_loss_avg:.6f}, test loss: {test_loss_avg:.6f}")
    return train_loss_list, test_loss_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loss_list,test_loss_list = train(model,train_dataset,test_dataset,lr=0.1,epochs=200,batch_size=64,device=device)


plt.plot(train_loss_list, 'r-', label='train loss', linewidth=3)
plt.plot(test_loss_list, 'k--', label='test loss', linewidth=2)
plt.legend(loc='best')
plt.show()

