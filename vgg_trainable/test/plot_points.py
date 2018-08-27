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
