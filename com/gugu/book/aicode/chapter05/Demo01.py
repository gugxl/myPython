import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def getRatingInformation(ratings):
    rates = []
    for line in ratings:
        rate = line.split('\t')
        rates.append([int(rate[0]), int(rate[1]), int(rate[2])])
    return rates


def createUserRankDic(rates):
    '''
    生成用户评分的数据结构
    :param rates: 索引数据[[2,1,5],[2,4,2]...]
    :return: 1. 用户打分字典，2.电影字典
    使用字典 key是用户ID, value是用户对电影的评价
    rate_dic[2] = [(1,5),(4,2)...] 用户2对电影1的评价是5分，对电影4的评价是2分
    '''
    user_rate_dic = {}
    item_to_user = {}
    for i in rates:
        user_rank = (i[1], i[2])
        if i[0] in user_rate_dic:
            user_rate_dic[i[0]].append(user_rank)
        else:
            user_rate_dic[i[0]] = [user_rank]
        if i[1] in item_to_user:
            item_to_user[i[1]].append(i[0])
        else:
            item_to_user[i[1]] = [i[0]]
    return user_rate_dic, item_to_user


'''
使用userCF进行推荐
输入，电影名、用户ID、邻居数量
输出：推荐电影的ID、输入用户的电影列表，电影对用户的序列表、邻居列表
'''
def recommendByUserCF(file_name, user_id, k=5):
    test_contents = readFile(file_name)
    test_rates = getRatingInformation(test_contents)

    test_dic, test_item_to_user = createUserRankDic(test_rates)
    neighbors = calcNearestNeighbor(user_id, test_dic, test_item_to_user)[:k]
    recommend_dic = {}
    for neighbor in neighbors:
        neighbor_user_id = neighbor[1]
        movies = test_dic[neighbor_user_id]
        for movie in movies:
            if movie[0] not in recommend_dic:
                recommend_dic[movie[0]] = neighbor[0]
            else:
                recommend_dic[movie[0]] += neighbor[0]
        # 建立推荐列表
        recommend_list = []
        for key in recommend_dic:
            recommend_list.append([recommend_dic[key], key])
        recommend_list.sort(reverse=True)
        user_movies = [i[0] for i in test_dic[user_id]]
    return [i[1] for i in recommend_list], user_movies, test_item_to_user, neighbors


def convert_2_one_hot(df):
    genres_vals = df['genres'].values.tolist()
    genres_set = set()
    for row in genres_vals:
        genres_set.update(row.split('|'))
    genres_list = list(genres_set)
    row_num = 0
    df_new = pd.DataFrame(columns=genres_list)

    for row in genres_vals:
        init_genres_vals = [0] * len(genres_list)
        genres_names = row.split('|')
        for name in genres_names:
            init_genres_vals[genres_list.index(name)] = 1
        df_new.loc[row_num] = init_genres_vals
        row_num += 1

    df_update = pd.concat([df, df_new], axis=1)
    return df_update


# 把rating转换0,1分类
def convert_rating_2_labels(ratings):
    label = []
    ratings_list = ratings.values.tolist()
    for rate in ratings_list:
        if rate > 3.0:
            label.append(1)
        else:
            label.append(0)
    return label


# 构建逻辑回归模型
def training_lr(X, y):
    model = LogisticRegression(penalty='l2', C=1, solver='sag', max_iter=500, verbose=1, n_jobs=8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model.fit(X_train, y_train)
    train_pred = model.predict_proba(X_train)
    train_auc = roc_auc_score(y_train, train_pred[:, 1])

    test_pred = model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, test_pred[:, 1])

    print('lr train auc socre: ' + str(train_auc))
    print('lr test auc socre: ' + str(test_auc))


def load_data():
    movie_df = pd.read_csv('movies.csv')
    rating_df = pd.read_csv('ratings.csv')
    df_update = convert_2_one_hot(movie_df)
    df_final = pd.merge(rating_df, df_update, on='movieId')
    ratings = df_final['rating']
    df_final = df_final.drop(columns=['userId', 'movieId', 'timestamp', 'title', 'genres', 'rating'])
    labels = convert_rating_2_labels(ratings)
    trainx = df_final.values.tolist()
    return trainx, labels


if __name__ == '__main__':
    trainx, labels = load_data()
    training_lr(trainx, labels)
