import keras.layers
from keras import Model
from keras.utils import plot_model
import numpy as np


class SimpleMLP(Model):
    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

if __name__ == '__main__':
    pass
    # model = SimpleMLP()
    # model.compile()
    # model.fit()