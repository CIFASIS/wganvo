#
# This file is part of wganvo.
#
# Copyright (C) 2019 Javier Cremona (CIFASIS-CONICET)
# For more information see <https://github.com/CIFASIS/wganvo>
#
# wganvo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wganvo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with wganvo. If not, see <http://www.gnu.org/licenses/>.
#

from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt


def read(filename, delimiter=','):
    return np.genfromtxt(filename, delimiter=delimiter)


def plot(array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 111 means "1x1 grid, first subplot"
    p = ax.plot(array[:, 0], array[:, 1], array[:, 2], label='target')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()


def main():
    import transformations
    data = read('vo.csv')
    data = data[1:len(data), 2:8]

    current = np.array([0., 0., 0.])  # .transpose()
    # current = np.matrix(np.identity(4))
    num_examples = len(data)
    ts = np.empty((num_examples, 3))
    poses = np.empty((num_examples, 12))
    i = 0
    for t in data:
        # Devuelve una matriz 4x4
        # t[3] = roll, t[4] = pitch, t[5] = yaw
        T = transformations.euler_matrix(t[3], t[4], t[5], 'sxyz')
        T[0:3, 3] = t[0:3]
        current = t[0:3] + current  # np.linalg.inv(T) *current   #np.linalg.inv(T) * current
        ts[i] = current  # [0:3,3].transpose()
        # poses[i] = current[0:3,:].reshape(12)
        i += 1

    np.savetxt("poses.txt", poses, delimiter=" ")
    plot(ts)


if __name__ == "__main__":
    main()
