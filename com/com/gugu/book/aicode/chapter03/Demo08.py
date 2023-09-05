from keras import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing


def createModel():
    model = Sequential()
    model.add(Dense(32, input_shape=(13,), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def createModel2():
    model = Sequential()
    model.add(Dense(32, input_shape=(13,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    model = createModel()
    # 训练模型
    model.fit(x_train, y_train, batch_size=8, epochs=10000)
    print(model.metrics_names)
    print(model.evaluate(x_test, y_test))

    for i in range(10):
        y_pred = model.predict([[x_test[i]]])
        print("predict:{}, target:{}".format(y_pred[0][0], y_test[i]))

