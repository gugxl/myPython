import keras.backend as  K
import numpy as np
from keras import Sequential
from keras.engine.base_layer import Layer


class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建可以进行训练调节的权重变量
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], self.output_dim),
            initializer='uniform',
            trainable=True
        )
        super(MyLayer, self).build(input_shape=input_shape)

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

if __name__ == '__main__':
    model = Sequential()
    model.add(MyLayer(1, input_shape=(2,)))
    print(model.predict(np.array([[3, 4]])))