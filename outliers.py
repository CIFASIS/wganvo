import numpy as np
import matplotlib.pyplot as plt


def reject_outliers(data, m=2.):
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
    plt.plot(data[:, 0, :])
    plt.subplot(132)
    plt.plot(data[:, 1, :])
    plt.subplot(133)
    plt.plot(data[:, 2, :])
    plt.show()


def load(path):
    return np.load(path)

# def fix_array(m, K, N):
#     new_m = np.empty((m.shape[0], m.shape[1], 25))
#     for idx, points in enumerate(m):
#         points_h = np.ones((4, 150))
#         points_h[0:3, :] = points
#         x1 = np.matmul(K, points_h)
#         x1 /= x1[2]
#         c_mask = center_crop_mask(x1)
#         points = points[:, c_mask]
#         front_mask = in_front_of_cam_mask(points, 0.)
#         points = points[:, front_mask]
#         replace = points.shape[1] <= N
#         random_selection = np.random.choice(points.shape[1], N, replace=replace)
#         points = points[:3, random_selection]
#         new_m[idx] = points
#     return new_m

if __name__ == "__main__":
    m = load('/home/jcremona/output/02/points.npy')
    print_points(m)
    # m.shape -> Nx3xP
    # new_m = fix_array(m, K, N)
    #np.save('/home/jcremona/output/09/points.npy', new_m)

    # Take some example
    X = m[432]
    for i in range(3):
        # Each axis (X,Y,Z) is filtered by mask_outliers
        mask = mask_outliers(X[i], 1000)
        X = X[:, mask]
    print(X.shape)