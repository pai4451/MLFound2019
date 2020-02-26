from Logistic_Regression import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 2000
    model = Logistic_Regression()
    train_x, train_y = model.load_data('./hw3_train.dat')
    test_x, test_y = model.load_data('./hw3_test.dat')
    Ein, Eout, Ein_S, Eout_S = model.fit(train_x, train_y, test_x, test_y, N)
    t = range(N)
    plt.style.use('ggplot')
    plt.xlabel('$t$')
    plt.ylabel('$E_{in}$')
    plt.plot(t, Ein, t, Ein_S)
    plt.legend(['GD ($\eta=0.01$)','SGD ($\eta=0.001$)'])
    plt.title('$E_{in}(\mathbf{w}_t)$ as a function of $t$')
    plt.savefig('./Ein.pdf',format="pdf")
    plt.show()
    