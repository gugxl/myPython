from keras.utils import plot_model

from keras.models import Model
from keras.layers import Input, Dense, concatenate

if __name__ == '__main__':
    input1 = Input(shape=(2,))
    h1 = Dense(3, activation='sigmoid')(input1)
    output1 = Dense(1, activation='sigmoid')(h1)

    input2 = Input(shape=(3,))
    new_input = concatenate([output1, input2])
    h2 = Dense(4, activation='sigmoid')(new_input)
    output2 = Dense(2, activation='sigmoid')(h2)
    model3 = Model(inputs=[input1, input2], outputs=[output1, output2])
    plot_model(model3, to_file='m3.png', show_shapes=True)
