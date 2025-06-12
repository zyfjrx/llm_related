import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.functions import sigmoid,identity,softmax

# 读取数据
def get_data():
    data = pd.read_csv('../pytorch_dl/data/train.csv')
    X = data.drop(columns=['label'],axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    return  x_test, y_test

def init_network():
    network = joblib.load('../pytorch_dl/data/nn_sample')
    return network

def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

if __name__ == '__main__':
    x_test, y_test = get_data()
    network = init_network()
    batch_size = 100
    accuracy = 0
    for i in range(0,len(x_test),batch_size):
        x_batch = x_test[i:i+batch_size]
        y_batch = forward(network, x_batch)
        prediction = np.argmax(y_batch, axis=1)
        accuracy += np.sum(prediction==y_test[i:i+batch_size])
    n = x_test.shape[0]
    accuracy /= n
    print(x_test.shape)
    print('Accuracy: %.2f%%' % (accuracy*100))

