import numpy as np

class PLA(object):
    def __init__(self, η, model="naive cycle"):
        self.__η = η
        self.__model = model
    def load_train_data(self, path, seed):
        data = np.genfromtxt(path, encoding='utf8', dtype=float)
        x0 = np.ones((400,1))
        data = np.concatenate((x0,data), axis=1)
        if self.__model == "random cycle":
            np.random.seed(seed)
            np.random.shuffle(data)
        train_x = data[:, 0:5]
        train_y = data[:, 5]
        return train_x, train_y
    def train(self, path, seed=None):
        count = 0
        train_x, train_y = self.load_train_data(path, seed)
        w = np.zeros(5)
        iteration = True
        sign = lambda x: 1 if x > 0 else -1 
        while iteration:
            for idx, x in enumerate(train_x):
                if sign(np.dot(x, w)) != train_y[idx]:
                    count += 1
                    w = w + self.__η * train_y[idx] * x
            iteration = False
            for idx, x in enumerate(train_x):
                if sign(np.dot(x, w)) != train_y[idx]:
                    iteration = True
                    break
        return count

if __name__ == '__main__':
    data_path = './data/hw1_15_train.dat'
    pla = PLA(η=1, model = "naive cycle")
    num_of_updates = pla.train(data_path)
    print ('Number of updates: %d' % num_of_updates) 