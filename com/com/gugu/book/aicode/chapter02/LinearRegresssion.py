import random


class LinearRegression(object):
    def __init__(self, eta=0.01, itrations=10):
        self.lr = eta
        self.itrations = itrations
        self.w = 0.0
        self.bias = 0.0

    # 损失函数
    def cost_function(self, X, Y, weight, bias):
        n = len(X)
        total_error = 0.0
        for i in range(n):
            total_error += (Y[i] - (weight * X[i] + bias)) ** 2
        return total_error / n

    # 利用梯度调整w和bias
    def update_weights(self, X, Y, weight, bias, learning_rate):
        dw = 0
        db = 0
        n = len(X)

        for i in range(n):
            dw += -2 * X[i] * (Y[i] - (weight * X[i] + bias))
            db += -2 * (Y[i] - (weight * X[i] + bias))

        weight -= (dw / n) * learning_rate
        bias -= (db / n) * learning_rate

        return weight, bias

    # 随机梯度下降
    def update_weights22(self, X, Y, weight, bias, learning_rate):
        dw = 0
        db = 0
        n = len(X)
        indexes = [0 for _ in range(n)]
        random.shuffle(indexes)
        batch_size = 4

        for k in range(batch_size):
            i = indexes[k]
            dw += -2 * X[i] * (Y[i] - (weight * X[i] + bias))
            db += -2 * (Y[i] - (weight * X[i] + bias))

        weight -= (dw / n) * learning_rate
        bias -= (db / n) * learning_rate

        return weight, bias

    # 反复调用update_weights调整w和bias
    def fit(self, X, Y):
        cost_history = []
        for i in range(self.itrations):
            self.w, self.bias = self.update_weights(X, Y, self.w, self.bias, self.lr)
            #             计算误差，用于观察和监控训练过程
            cost = self.cost_function(X, Y, self.w, self.bias)
            cost_history.append(cost)

            if i % 10 == 0:
                print("iter={:d}   weight={:.2f}   bias={:.4f}   cost={:.2}".format(i, self.w, self.bias, cost))
        return self.w, self.bias, cost_history

    # 使用w和bias进行预测分类 1. 对x进行归一化，然后根据激活函数的设定，把wx+bias直接作为输出
    def predict(self, x):
        x = (x + 100) / 200
        return self.w * x + self.bias


if __name__ == '__main__':
    # 训练过程，第53行对数据进行归一化，使其落在[0,1] 区间，然后fit函训练500次(每次使用全量的数据)
    x = [1, 2, 3, 10, 20, 50, 100, -2, -10, -100, -5, -20]
    y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    model = LinearRegression(0.01, 500)
    X = [(k + 100) / 200 for k in x]
    model.fit(X, y)

    # 对于输入数据进行预测
    test_x = [90, 80, 81, 82, 75, 40, 32, 15, 5, 1, -1, -15, -20, -22, -33, -45, -60, -92]
    for i in range(len(test_x)):
        print('input {} => predict: {}'.format(test_x[i], model.predict(test_x[i])))
