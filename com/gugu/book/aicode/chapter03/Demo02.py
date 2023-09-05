from keras.utils import plot_model

from keras.models import Sequential, Model
from keras.layers import Input, Dense

if __name__ == '__main__':
    model1 = Sequential()
    model1.add(Dense(32, input_shape=(32,), activation='sigmoid'))
    plot_model(model1, to_file='m1.png', show_shapes=True)

    a = Input(shape=(32,))
    b = Dense(1, activation='sigmoid')(a)
    model2 = Model(inputs=a, outputs=b)
    plot_model(model2, to_file='m2.png', show_shapes=True)
