from decision_stump import *
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    np.random.seed(1126)
    DiffList = []
    size = 20
    iterations = 1000
    for i in range(iterations):
        d = decision_stump()
        x, y = d.Artifical_data(size=size,p=0.2)
        s, theta, Ein = d.train_1d(x, y)
        Eout = 0.5 + 0.3 * s * (abs(theta) - 1)
        DiffList.append(Ein-Eout)

    arr = plt.hist(DiffList, color='y', edgecolor='black')
    plt.xlabel('$E_{in}-E_{out}$')
    plt.ylabel('Frequency')
    for i in range(10):
        plt.text(arr[1][i]+0.01,arr[0][i],str(int(arr[0][i])))
    plt.title('Histogram of $E_{in}-E_{out}$')
    plt.savefig('./p7.pdf',format="pdf")
    plt.show()
    print(np.mean(DiffList))
    print(np.var(DiffList))