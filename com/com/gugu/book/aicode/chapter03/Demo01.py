import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model

if __name__ == '__main__':
    model = Sequential([  # 顺序神经网络
        Dense(4, input_shape=(2,)),  # Dense 全连接层 4个神经元，输入是的长度为2的一维数组
        Activation('sigmoid'),  # 激活函数为sigmoid
        Dense(1),
        Activation('sigmoid'),
    ])
    # 直观展示网络模型
    plot_model(model, to_file='training_model.png', show_shapes=True)

    # 配置 learning rate=0.001 损失函数是mse，把准确率设置为评价标准
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['accuracy'])

    training_number = 1000
    training_data = np.random.random((training_number, 2))
    labels = [(1 if data[0] < data[1] else 0) for data in training_data]
    model.fit(training_data, np.array(labels), epochs=500, batch_size=32)

    test_number = 100
    test_data = np.random.random((test_number, 2))
    expected = [(1 if data[0] < data[1] else 0) for data in test_data]
    error = 0
    for i in range(0, test_number):
        data = test_data[i].reshape(1, 2)
        pred = 0 if model.predict(data) < 0.5 else 1

        if(pred != expected[i]):
            error += 1

    print("total errors:{},accuracy:{}".format(error, 1.0-error/test_number))
