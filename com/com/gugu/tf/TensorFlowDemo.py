
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# AI 版hello world
def get_num_class():
    model = Sequential()
    model.add(Dense(units=8, activation='relu', input_dim=1))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='sgd')

    x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
    y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    model.fit(x, y, epochs=10, batch_size=4)

    test_x = [30, 40, -20, -60]
    test_y = model.predict(test_x)

    for i in range(0, len(test_y)):
        print('input {} => predict: {}'.format(test_x[i], test_y[i]))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_num_class()
    # print(tf.__version__)
