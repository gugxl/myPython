from keras import Sequential
from keras.layers import Conv2D, Flatten

if __name__ == '__main__':
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(3, 32, 32), padding='same', ))
    print(model.output_shape)
    # 把输入变成一维模式
    model.add(Flatten())
    print(model.output_shape)
