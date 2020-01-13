import numpy as np
import matplotlib.pyplot as plt

class decision_stump(object):
    def __init__(self):
        self.s = 1
        self.best_theta = 0.
        self.best_error = 1.
    def Artifical_data(self, size, p):
        x = np.random.uniform(-1, 1, size)
        noise = np.random.uniform(0, 1, size)
        y = np.sign(x)
        y[noise <= p] *= -1
        return x, y
    def hypothesis(self, x, y, s):
        x_sort = np.sort(x)
        theta = (x_sort[:-1] + x_sort[1:]) / 2.
        theta = np.insert(theta, 0, (-1+x_sort[0])/2.)
        theta = theta.reshape(theta.shape[0],1)
        X = np.broadcast_to(x, (theta.shape[0], x.shape[0]))
        h = np.sign(X - theta) * s
        error = np.sum(h!=y, axis = 1)/ len(y)
        best_error = np.min(error)
        best_theta = theta[np.argmin(error)][0]
        return best_theta, best_error
    def train_1d(self, x, y):
        theta1, error1 = self.hypothesis(x, y, s = 1)
        theta2, error2 = self.hypothesis(x, y, s = -1)
        if error1 <= error2:
            s = 1
            best_theta = theta1
            best_error = error1
        else:
            s = -1
            best_theta = theta2
            best_error = error2
        return s, best_theta, best_error
    def load_data(self, train_path, test_path):
        train_data = np.genfromtxt(train_path)
        test_data = np.genfromtxt(test_path)
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]
        return train_x, train_y, test_x, test_y
    def train_multi_d(self, x, y):
        for i in range(x.shape[1]):
            xi = x[:, i]
            s, theta, error = self.train_1d(xi, y)
            if error < self.best_error:
                self.best_error = error
                self.dim = i
                self.best_theta = theta
                self.s = s
        return self.s, self.best_theta, self.dim, self.best_error