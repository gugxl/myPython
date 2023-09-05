
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# AI 版hello world
def get_num_class():
    # 建立两层Sequential类型神经网络
    model = Sequential()
    # 第一层   input_dim = 1 只有一个输入;units=8  8个输出(可以简单理解神经元的个数)
    model.add(Dense(units=8, activation='relu', input_dim=1))
    # 第二层 一个输出
    model.add(Dense(units=1, activation='sigmoid'))
    # 进行配置 loss='mean_squared_error' 损失函数为平均方差； optimizer='sgd' 优化方式为sgd随机梯度下降
    model.compile(loss='mean_squared_error', optimizer='sgd')
    # 设定训练集
    x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
    y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # 调用model.fit函数训练, epochs=10 训练10次，batch_size=4 每次挑选4组数据
    model.fit(x, y, epochs=10, batch_size=4)
    # 构建4个输入数据测试打印结果，model.predict进行预测
    test_x = [30, 40, -20, -60]
    test_y = model.predict(test_x)

    for i in range(0, len(test_y)):
        print('input {} => predict: {}'.format(test_x[i], test_y[i]))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_num_class()
    # print(tf.__version__)
