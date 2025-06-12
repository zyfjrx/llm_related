import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 读取数据
def get_data():
    data = pd.read_csv('../pytorch_dl/data/train.csv')
    X = data.drop(columns=['label'], axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test, y_train.values, y_test.values
