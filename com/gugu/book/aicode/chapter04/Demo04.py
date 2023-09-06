import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

if __name__ == '__main__':
    path = 'iris.data'
    data = pd.read_csv(path, header=None)
    data[4] = pd.Categorical(data[4]).codes
    x, y = np.split(data.values, (4,), axis=1)
    x = x[:, :2]
    lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=3)),
                   ('clf', LogisticRegression())
                   ])
    lr.fit(x, y.ravel())
    y_hat = lr.predict(x)
    y_hat_prob = lr.predict_proba(x)
    np.set_printoptions(suppress=True)
    print('y_hat= \n', y_hat)
    print('y_hat_prob= \n', y_hat_prob)
    print('准确度： %2f%% ' % (100 * np.mean(y_hat == y.ravel())))
