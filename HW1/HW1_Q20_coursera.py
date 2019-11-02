import numpy as np
import matplotlib.pyplot as plt

class PocketAlgo(object):
    def __init__(self, η, model="naive cycle"):
        self.__η = η 
        self.__model = model

    def load_train_data(self, path, seed):
        data = np.genfromtxt(path, encoding='utf8', dtype=float)
        x0 = np.ones((500,1))
        data = np.concatenate((x0,data), axis=1)
        if self.__model == "random cycle":
            np.random.seed(seed)
            np.random.shuffle(data)
        train_x = data[:, 0:5]
        train_y = data[:, 5]
        return train_x, train_y

    def train(self, path, seed):
        count = 0
        train_x, train_y = self.load_train_data(path, seed)
        w = np.zeros(5)
        w_pocket = np.zeros(5)
        best_error = len(train_y)
        sign = lambda x: 1 if x > 0 else -1 
        for idx, x in enumerate(train_x):
            if sign(np.dot(x, w)) != train_y[idx]:
                w = w + self.__η * train_y[idx] * x
                count += 1
                error = 0
                for idx, x in enumerate(train_x):
                    if sign(np.dot(x, w)) != train_y[idx]:
                        error += 1
                if error < best_error:
                    best_error = error
                    w_pocket = w
                if count == 100:
                    break
        return w_pocket

    def load_test_data(self, path):
        data = np.genfromtxt(path, encoding='utf8', dtype=float)
        x0 = np.ones((500,1))
        data = np.concatenate((x0,data), axis=1)
        test_x = data[:, 0:5]
        test_y = data[:, 5]
        return test_x, test_y

    def test(self, train_path, test_path, seed=None):
        w_pocket = self.train(train_path, seed)
        test_x, test_y = self.load_test_data(test_path)
        count = 0.
        sign = lambda x: 1 if x > 0 else -1 
        for idx, x in enumerate(test_x):
            if sign(np.dot(x, w_pocket)) != test_y[idx]:
                count += 1
        return count / len(test_y)

if __name__ == '__main__':
    train_path = "./data/hw1_18_train.dat"
    test_path = "./data/hw1_18_test.dat"
    error_rate = 0.
    for i in range(2000):
        Pocket = PocketAlgo(η=1, model="random cycle")
        error_rate+= Pocket.test(train_path, test_path, i)
    print('Average error rate: %f' % (error_rate / 2000.0))