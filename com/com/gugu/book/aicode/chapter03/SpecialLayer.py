import numpy as np
from keras import Sequential
from keras.engine.base_layer import Layer

# 自定义输入层对输入向量 *2
class SpecialLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SpecialLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SpecialLayer, self).build(input_shape=input_shape)

    def call(self, x):
        return x * 2

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

if __name__ == '__main__':
    model = Sequential()
    model.add(SpecialLayer(1, input_shape=(2,)))
    print(model.predict(np.array([[3, 4]])))
