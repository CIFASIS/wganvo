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

import tensorflow as tf
import numpy as np
import argparse
import sys, os
import matplotlib.pyplot as plt

def plot(X_axis, Y_axis, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(X_axis, Y_axis, 'r.')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #plt.show()
    return fig, ax
    #

def main(_):
    points = np.loadtxt(FLAGS.file, delimiter=' ')
    X_axis = points[:, 0]
    Y_axis = points[:, 1]
    fig, ax = plot(X_axis, Y_axis, "frames", "distance(m)")
    fig.savefig(os.path.join(FLAGS.output_dir, 'frames_vs_dist.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file',
        type=str,
        help='File'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.getcwd(),
        help='Output Dir'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
