import numpy as np
import matplotlib.pyplot as plt

def reject_outliers(data, m = 2.):
    mask = mask_outliers(data, m)
    return data[mask]

def mask_outliers(data, m):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return s < m

# data: Nx3xP
def print_points(data):
    plt.subplot(131)
    plt.plot(data[:,0,:])
    plt.subplot(132)
    plt.plot(data[:, 1, :])
    plt.subplot(133)
    plt.plot(data[:, 2, :])
    plt.show()


def load(path):
    return np.load(path)


if __name__ == "__main__":
    m = load('/home/jcremona/output/02/points.npy')
    print_points(m)
    # m.shape -> Nx3xP

    # Take some example
    X = m[432]
    for i in range(3):
        # Each axis (X,Y,Z) is filtered by mask_outliers
        mask = mask_outliers(X[i], 1000)
        X = X[:, mask]
    print(X.shape)