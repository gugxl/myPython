import numpy as np
from keras import Sequential, Input, Model
from keras.layers import *


def reshape_test():
    model = Sequential()
    # Reshape  把输出转换成给定目标的形状
    model.add(Reshape((3, 4), input_shape=(12,)))
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    y = model.predict(x)
    print(y)


def permute_test():
    model = Sequential()
    x = np.array([[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]])
    model.add(Permute([2, 1], input_shape=(2, 6)))
    y = model.predict(x)
    print(y)


def permute_test2():
    model = Sequential()
    x = np.array([[
        [[1, 1], [2, 2], [3, 3]],
        [[4, 4], [5, 5], [6, 6]],
        [[7, 7], [8, 8], [9, 9]],
        [[10, 10], [11, 11], [12, 12]],
    ]])
    model.add(Permute([2, 3, 1], input_shape=(4, 3, 2)))
    y = model.predict(x)
    print(y)


# RepeatVector 重复n次输入
def repeat_vector_test():
    x = np.array([[1, 2, 3, 4]])
    model = Sequential()
    model.add(RepeatVector(3, input_shape=(4,)))
    y = model.predict(x)
    print(y)


# Lambda 可以把表达式作为与一个参数传入
def lambda_test():
    x = np.array([[1, 2, 3, 4]])
    model = Sequential()
    model.add(Lambda(lambda x: x * 2, input_shape=(4,)))
    y = model.predict(x)
    print(y)


def calculation(tensors):
    output1 = tensors[0] - tensors[1]
    output2 = tensors[0] + tensors[1]
    output3 = tensors[0] * tensors[1]
    return [output1, output2, output3]


def lambda_test2():
    input1 = Input(shape=[4, ])
    input2 = Input(shape=[4, ])
    layer = Lambda(calculation)
    out1, out2, out3 = layer([input1, input2])
    model = Model(inputs=[input1, input2], outputs=[out1, out2, out3])

    x1 = np.array([[1, 2, 3, 4]])
    x2 = np.array([[1, 1, 1, 1]])
    print(model.predict([x1, x2]))


def masking_test():
    model = Sequential()
    # Masking 忽略某次输入 下面是全为1的被替换[0,0,0],或者替换为[7,8,9]
    model.add(Masking(1, input_shape=(4, 3)))
    x = np.array([[
        [1, 2, 3],
        [1, 1, 1],
        [7, 8, 9],
        [10, 11, 12],
    ]])
    print(model.predict(x))


if __name__ == '__main__':
    # reshape_test()
    # permute_test()
    # permute_test2()
    # repeat_vector_test()
    # lambda_test()
    # lambda_test2()
    masking_test()
