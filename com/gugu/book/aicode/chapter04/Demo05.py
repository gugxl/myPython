import numpy as np
from keras import Sequential
from keras.layers import Dense

if __name__ == '__main__':
    np.random.seed(7)
    # 读取糖尿病数据
    # pregnants,Plasma_glucose_concentration,blood_pressure,Triceps_skin_fold_thickness,serum_insulin,BMI,Diabetes_pedigree_function,Age,Target
    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # 将数据分为训练数据和标签，前8列为特征
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # 创建模型，3层神经网络，分别为输入层，隐藏层，输出层
    model = Sequential()
    model.add(Dense(4, input_dim=8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 吧特征和标签放到模型中
    model.fit(X, Y, epochs=10000, batch_size=32)

    # 衡量模型预测效果
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
