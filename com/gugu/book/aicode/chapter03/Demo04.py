import numpy as np
from keras import Sequential, Input
from keras.layers import Dense

if __name__ == '__main__':
    model1 = Sequential()
    model1.add(Dense(32, input_shape=(32,), activation='sigmoid'))
    input1 = Input(shape=(2,))
    h1 = Dense(3, activation='sigmoid')(input1)

    model = Sequential([
        Dense(4, input_shape=(2,))

    ])
    training_number = 1000
    training_data = np.random.random((training_number, 2))
    # model.fit(training_data, np.array(labels), epochs=20, batch_size=32)