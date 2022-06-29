from functools import partial
import numpy as np
import scipy as sci

from scipy import io
from scipy import optimize as opt

class MovieRecommender:
    def __init__(self, movies, users, features) -> None:
        self.movies = movies
        self.users = users
        self.features = features
        return

    def serialize(self, X, theta):
        return np.concatenate((X.ravel(), theta.ravel()))

    def deserialize(self, param):
        X = param[:self.movies * self.features].reshape(self.movies, self.features)
        theta = param[self.movies * self.features:].reshape(self.users, self.features)
        return X, theta

    def initializer(self,Y):
        X = np.random.random(size=(self.movies, self.features))
        theta = np.random.random(size=(self.users, self.features))
        Y = Y - Y.mean()
        return self.serialize(X, theta), Y

    def cost(self, params, Y, R, lam = 0.0):
        X, theta = self.deserialize(params)
        inner = np.multiply(X @ theta.T - Y, R)
        c = np.power(inner, 2).sum() / 2
        reg = (np.power(params, 2).sum()) * lam / 2
        return c + reg

    def gradient(self, params, Y, R, lam = 0.0):
        X, theta = self.deserialize(params)
        inner = np.multiply(X @ theta.T - Y, R)
        x_grad = inner @ theta + lam * X
        theta_grad = inner.T @ X + lam * theta
        return self.serialize(x_grad, theta_grad)

    def train(self, params, Y, R, lam = 0.0):
        fmin = opt.minimize(fun=partial(MovieRecommender.cost, self), 
                            x0=params, 
                            args=(Y, R, lam),
                            method='TNC',
                            jac=partial(MovieRecommender.gradient, self))
        return fmin


if __name__ == '__main__1':
    # Y是包含从1到5的等级的（数量的电影x数量的用户）
    # R是包含指示用户是否给电影评分的二进制值的“指示符”数组。
    data = sci.io.loadmat('./data_sets/ex8_movies.mat')
    Y, R = np.array( data['Y']), np.array(data['R'])
    print('Y.shape = {}, R.shape = {}'.format(Y.shape, R.shape))

    data = sci.io.loadmat('./data_sets/ex8_movieParams.mat')
    X, theta = np.array(data['X']), np.array(data['Theta'])
    print('X.shape = {}, theta.shape = {}'.format(X.shape, theta.shape))

    filter = MovieRecommender()
    print(filter.cost(X, theta, Y, R, lam = 1))
    print(filter.gradient(X, theta, Y, R, lam = 1))



if __name__ == '__main__':
    data = sci.io.loadmat('./data_sets/ex8_movies.mat')
    Y, R = np.array( data['Y']), np.array(data['R'])
    print('Y.shape = {}, R.shape = {}'.format(Y.shape, R.shape))
    # 载入电影列表
    movies_list = []
    f = open('./data_sets/movie_ids.txt', encoding= 'ISO-8859-1')
    for line in f:
        tokens = line.strip().split(' ')
        movies_list.append(' '.join(tokens[1:]))
    movies_list = np.array(movies_list)

    # 将自己的评级向量添加到现有数据集中以包含在模型中。
    ratings = np.zeros((1682, 1))
    ratings[0] = 4
    ratings[6] = 3
    ratings[11] = 5
    ratings[53] = 4
    ratings[63] = 5
    ratings[65] = 3
    ratings[68] = 5
    ratings[97] = 2
    ratings[182] = 4
    ratings[225] = 5
    ratings[354] = 5

    Y = np.append(ratings, Y, axis=1)
    R = np.append(ratings != 0, R, axis=1)

    # 初始化
    (movies, users) = Y.shape
    features = 10
    # print(movies, users, features)
    filter = MovieRecommender(movies, users, features)

    params, Y = filter.initializer(Y)
    # print(Y.mean())
    fmin = filter.train(params, Y, R, 10)
    X_trained, theta_trained = filter.deserialize(fmin.x)

    prediction = X_trained @ theta_trained.T
    my_preds = prediction[:, 0] + Y.mean()
    idx = np.argsort(my_preds)[::-1]
    print(idx)
    for m in movies_list[idx][:10]:
        print(m)

