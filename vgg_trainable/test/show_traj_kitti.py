import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import argparse
import sys, os, inspect
from mpl_toolkits.mplot3d import Axes3D
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from input_data import read_data_sets, DataSet

def show(images, poses):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    artist = ax1.imshow(images[0], cmap='gray')
    ax2 = fig.add_subplot(212)
    x = []
    y = []
    ax2.plot(x,y)
    for img,pose in zip(images, poses):
        artist.set_data(img)
        #plt.xticks([])
        #plt.yticks([])
        pose = pose.reshape(3,4)
        x.append(pose[0,3])
        y.append(pose[2,3])
        # plt.pause(0.05)
        plt.gca().lines[0].set_xdata(x)
        plt.gca().lines[0].set_ydata(y)

        plt.gca().relim()
        plt.gca().autoscale_view()
        plt.pause(0.01)

def main():
    images,_,_,_ = read_data_sets(FLAGS.img_file)
    poses = np.loadtxt(FLAGS.poses, delimiter=" ")
    print(len(images))
    print(len(poses))
    #assert len(images) == len(poses)
    show(images[...,1], poses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'img_file',
        type=str,
        help='Images file'
    )
    parser.add_argument(
        'poses',
        type=str,
        help='Poses'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()


