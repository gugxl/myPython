import numpy as np
import pandas as pd
from keras.losses import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit

#  线性回归示例
if __name__ == '__main__':

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    X = data
    y = target
    model = LinearRegression()
    best_model = model
    best_test_mse = 100
    cv = ShuffleSplit(n_splits=3, test_size=1, random_state=0)
    for train, test in cv.split(X):
        model.fit(X[train], y[train])
        train_pred = model.predict(X[train])
        train_mse = mean_squared_error(y[train], train_pred)
        test_pred = model.predict(X[test])
        test_mse = mean_squared_error(y[test], test_pred)
        print('train mse:' + str(train_mse) + 'test mse:' + str(test_mse))
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_model = model

    print('lr best mse score: '+ str(best_test_mse))

