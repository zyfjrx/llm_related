import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
fashion_mnist_train = pd.read_csv("../data/fashion-mnist_train.csv")
fashion_mnist_test = pd.read_csv("../data/fashion-mnist_test.csv")
# N * 784 ----> N,1,28,28
X_train = torch.tensor(fashion_mnist_train.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
X_test = torch.tensor(fashion_mnist_test.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train = torch.tensor(fashion_mnist_train.iloc[:, 0].values, dtype=torch.int64).reshape(-1)
y_test = torch.tensor(fashion_mnist_test.iloc[:, 0].values, dtype=torch.int64).reshape(-1)
# plt.imshow(X_train[222, 0, :, :, ], cmap="gray")
# plt.show()
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
# print(len(train_dataset))
print(X_test[666].shape)
# 搭建模型
model = nn.Sequential(
    nn.Conv2d(1, 6, 5, 1, 2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
    nn.Conv2d(6, 16, 5, 1, 0),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10),
)
# 查看各层输出数据形状
# X = torch.randn(1, 1, 28, 28, dtype=torch.float32)
# for layer in model:
#     X = layer(X)
#     print(f"{layer.__class__.__name__:<12}output shape: {X.shape}")


def train(model, train_dataset, test_dataset, lr, epochs, batch_size, device):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epochs):
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_total_loss = 0
        train_correct_count = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_total_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct_count += pred.eq(y).sum().item()

        train_avg_loss = train_total_loss / len(train_loader)
        train_acc = train_correct_count / len(train_dataset)
        train_loss_list.append(train_avg_loss)
        train_acc_list.append(train_acc)

        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_correct_count = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                pred = output.argmax(dim=1)
                test_correct_count += pred.eq(y).sum().item()
        test_acc = test_correct_count / len(test_dataset)
        test_acc_list.append(test_acc)
        print(f"epoch: {epoch+1},train loss: {train_avg_loss:.6f}, train acc: {train_acc:.6f}, test acc: {test_acc:.6f}")
    return train_loss_list, train_acc_list, test_acc_list

device = torch.device("mps" if torch.mps.is_available() else "cpu")
train_loss_list, train_acc_list, test_acc_list = train(model, train_dataset, test_dataset, lr=0.01, epochs=20, batch_size=256, device=device)

# 选取一个测试数据进行验证
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(X_test[666, 0, :, :, ], cmap="gray")
ax[1].plot(train_loss_list, 'r-', label='train loss', linewidth=3)
ax[1].plot(train_acc_list, 'k--', label='train acc', linewidth=2)
ax[1].plot(test_acc_list, 'k--', label='test acc', linewidth=1)
ax[1].legend(loc='best')
plt.show()
print("真实分类标签：",y_test[666])

# 模型预测结果
output = model(X_test[666].unsqueeze(0).to(device))
y_pred = output.argmax(dim=1)
print("模型预测分类标签：",y_pred)

