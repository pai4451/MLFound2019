import numpy as np
import matplotlib.pyplot as plt

class Logistic_Regression(object):
    def load_data(self,path):
        data = np.genfromtxt(path, encoding='utf8', dtype=float)
        x0 = np.ones((data.shape[0],1))
        data = np.concatenate((x0,data), axis=1)
        X = data[:, :-1]
        y = data[:, -1]
        return X, y

    def sigmoid(self,s):
        return 1 / (np.exp(-s) + 1)

    def GD(self,X, y, w):
        grad = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            grad += -y[i] * X[i] * self.sigmoid(-w.dot(X[i]) * y[i])
        grad /= X.shape[0]
        return grad

    def SGD(self, X, y, w, i):
        return -y[i] * X[i] * self.sigmoid(-w.dot(X[i]) * y[i])

    def fit(self, X_train, y_train, X_test, y_test, N):
        Ein = []
        Eout = []
        Ein_S = []
        Eout_S = []
        w = np.zeros(X_train.shape[1])
        # GD
        for i in range(N):
            grad = self.GD(X_train, y_train, w)
            w -= 0.01 * grad
            predict_in = X_train.dot(w)
            predict_out = X_test.dot(w)
            ein = np.mean(np.sign(predict_in) != y_train)
            eout = np.mean(np.sign(predict_out) != y_test)
            Ein.append(ein)
            Eout.append(eout)
        # SGD
        w.fill(0)
        for i in range(2000):
            grad = self.SGD(X_train, y_train, w, i%1000)
            w -= 0.001 * grad
            predict_in = X_train.dot(w)
            predict_out = X_test.dot(w)
            ein = np.mean(np.sign(predict_in) != y_train)
            eout = np.mean(np.sign(predict_out) != y_test)
            Ein_S.append(ein)
            Eout_S.append(eout)
        return Ein, Eout, Ein_S, Eout_S






