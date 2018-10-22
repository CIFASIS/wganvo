import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy import linalg
import argparse
import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from input_data import read_data_sets, DataSet
from matplotlib import gridspec

def show(images, poses, pred_poses):
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[0])
    artist = ax1.imshow(images[0], cmap='gray')
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[1], projection='3d')

    x = []
    y = []
    z = []
    lin = ax2.plot(x, y, z, label='Ground Truth')[0]
    lin_pred = ax2.plot(x, y, z, label='Prediction')[0]
    ax2.legend()
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')

    max_range = np.array([poses[:,0].max() - poses[:,0].min(), poses[:,1].max() - poses[:,1].min(),
                          poses[:,2].max() - poses[:,2].min()]).max() / 2.0
    mean_x = poses[:,0].mean()
    mean_y = poses[:,1].mean()
    mean_z = poses[:,2].mean()

    ax2.set_xlim(mean_x - max_range, mean_x + max_range)
    ax2.set_ylim(mean_y - max_range, mean_y + max_range)
    ax2.set_zlim(mean_z - max_range, mean_z + max_range)

    def update(num, img, datalines, lines):
        artist.set_data(img[num])
        idx = num + 1
        for lin, data in zip(lines, datalines):
            lin.set_xdata(data[:idx,0])
            lin.set_ydata(data[:idx,1])
            lin.set_3d_properties(data[:idx,2])
        return lines

    #assert len(images) == len(poses)
    ani = animation.FuncAnimation(fig, update, frames=len(pred_poses), fargs=(images,[poses, pred_poses],[lin,lin_pred]),
                                  interval=10, blit=False)
    plt.show()

def main():
    images,_,_,_ = read_data_sets(FLAGS.img_file)
    poses = np.loadtxt(FLAGS.poses, delimiter=" ")
    poses_pred = np.loadtxt(FLAGS.poses_pred, delimiter=" ")
    last = images[-1][..., 1]
    last = last.reshape((-1,last.shape[0],last.shape[1]))
    im = np.append(images[..., 0], last,axis=0)
    print(len(im))
    print(len(poses))
    #assert len(im) == len(poses)

    poses = poses.reshape((-1,3,4))
    poses_pred = poses_pred.reshape((-1,3,4))
    print(poses.shape)
    print(poses_pred.shape)
    show(im, poses[:, 0:3, 3], poses_pred[:, 0:3, 3])


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
    parser.add_argument(
        'poses_pred',
        type=str,
        help='Poses Pred'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()


