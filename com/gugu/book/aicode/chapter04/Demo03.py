from sklearn import datasets

from com.com.gugu.book.aicode.chapter02.LinearRegresssion import LinearRegression

if __name__ == '__main__':
    loaded_data = datasets.load_boston()
    feature = loaded_data['feature_name']
    X = loaded_data.data
    y = loaded_data.target
    model = LinearRegression()