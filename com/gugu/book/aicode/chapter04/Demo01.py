import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    # 声明特征数据
    weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny',
               'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
    temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot',
                   'Mild']
    # 定义分类标签
    play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    # 将字符串数据通过label encoding转换成数字
    le = preprocessing.LabelEncoder()
    weather_encoded = le.fit_transform(weather)
    temp_encoded = le.fit_transform(temp)
    label = le.fit_transform(play)
    # 通过pands的concat方法把两列特征合并
    df1 = pd.DataFrame(weather_encoded, columns=['weather'])
    df2 = pd.DataFrame(temp_encoded, columns=['temp'])
    result = pd.concat([df1, df2], axis=1, sort=False)
    # 生成朴素贝叶斯分类模型，并把数据代入模型进行训练
    model = GaussianNB()
    trainx = np.array(result)
    model.fit(trainx, label)

    predicted = model.predict([[0,2]])
    print("Predicted Value:", predicted)

