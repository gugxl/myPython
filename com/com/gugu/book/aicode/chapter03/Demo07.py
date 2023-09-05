import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation

if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model1 = Sequential()
    model1.add(Dense(1, input_dim=2))
    model1.add(Activation('sigmoid'))
    model1.compile(loss='mean_squared_error', optimizer='adam')

    model1.fit(X, y, batch_size=1, epochs=10_000)
    # print(model1.predict(X))

    model2 = Sequential()
    model2.add(Dense(4, input_dim=2))
    model2.add(Activation('sigmoid'))
    model2.add(Dense(1))
    model2.add(Activation('sigmoid'))

    model2.compile(loss='mean_squared_error', optimizer='adam')
    model2.fit(X, y, batch_size=1, epochs=10_000)
    print(model2.predict(X))
