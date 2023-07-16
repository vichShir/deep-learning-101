import numpy as np
import pandas as pd
from layers import ReLU, Softmax, Dense
from models import Model, get_accuracy

if __name__ == '__main__':

    data = pd.read_csv('mnist_data/train.csv')

    data = np.array(data)
    m1, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m1].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    model = Model()
    #model.add_layer(Input(41000, 784))
    model.add_layer(Dense(10, 784, activation=ReLU))
    model.add_layer(Dense(20, 10, activation=ReLU))
    model.add_layer(Dense(10, 20, activation=Softmax))
    model.gradient_descent(X_train, Y_train, .2, 500)

    dev_predictions = model.predict(X_dev)
    test_acc = get_accuracy(dev_predictions, Y_dev)
    print(test_acc)

    df = pd.read_csv('mnist_data/test.csv')
    data = np.array(df).T
    X_test = data
    X_test = X_test / 255.

    y_test = model.predict(X_test)
    print(y_test.shape, y_test)

    df_sub = pd.DataFrame({
        'ImageId': range(1, y_test.shape[0]+1)
    })
    df_sub['Label'] = y_test
    df_sub.to_csv('mnist_data/kaggle_submission.csv', index=False)

    #0.8376585365853658
    #0.8584634146341463