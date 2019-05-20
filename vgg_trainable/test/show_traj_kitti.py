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

# Example:
# python show_traj_kitti.py ~/KITTI/ gt_pose_00.txt --poses_pred orb_slam2_pose_00.txt wganvo_pose_00.txt --labels ORB-SLAM2 WGANVO
# Labels and filenames must be the same length

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def show(images, poses, pred_poses, labels, points=None):
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[0])
    artist = ax1.imshow(images[0], cmap='gray')
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[1], projection='3d')

    x = []
    y = []
    z = []
    lin = ax2.plot(x, y, z, label='Ground Truth', linewidth=1.1)[0]
    data_ = [poses]
    lines_ = [lin]
    for i, label in zip(pred_poses, labels):
        lin_pred = ax2.plot(x, y, z, label=label, linewidth=1.1)[0]
        data_.append(i)
        lines_.append(lin_pred)


    cloud = ax2.scatter(x,y,z, c="red", s=0.1)
        # data_.append(points)
        # lines_.append(cloud)
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

    def update(num, img, datalines, lines, scatter, points):
        artist.set_data(img[num])
        idx = num + 1
        for lin, data in zip(lines, datalines):
            lin.set_xdata(data[:idx,0])
            lin.set_ydata(data[:idx,1])
            lin.set_3d_properties(data[:idx,2])
        #scatter._offsets3d = (points[num,0], points[num,1], points[num,2])
        return lines

    #assert len(images) == len(poses)
    # FIXME originalmente era frames = len(pred_poses)
    ani = animation.FuncAnimation(fig, update, frames=len(poses), fargs=(images,data_,lines_, cloud, points),
                                  interval=10, blit=False)
    plt.show()

def main():
    images,_,_,_, points = read_data_sets(FLAGS.img_file)
    poses = np.loadtxt(FLAGS.poses, delimiter=" ")
    assert len(FLAGS.poses_pred) == len(FLAGS.labels), "Num. of pose files and num. of labels must be the same"
    poses_pred = []
    for pose_file in FLAGS.poses_pred:
        pose_pred = np.loadtxt(pose_file, delimiter=" ")
        pose_pred = pose_pred.reshape((-1, 3, 4))
        pose_pred = pose_pred[:, 0:3, 3]
        poses_pred.append(pose_pred)

    # points = None
    # if FLAGS.points != None:
    #     points = np.load(FLAGS.points)

    last = images[-1][..., 1]
    last = last.reshape((-1,last.shape[0],last.shape[1]))
    im = np.append(images[..., 0], last,axis=0)
    print(len(im))
    print(len(poses))
    #assert len(im) == len(poses)

    poses = poses.reshape((-1,3,4))
    poses = poses[:, 0:3, 3]
    # print(poses.shape)
    #print(poses_pred.shape)
    # print(points.shape)
    show(im, poses, poses_pred, FLAGS.labels)


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
        '--poses_pred',
        nargs="*",
        help='Poses Pred'
    )
    parser.add_argument(
        '--labels',
        nargs="*",
        help='Labels/Legends to be used in the plot'
    )
    parser.add_argument(
        '--points',
        type=str,
        help='3D Points'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()


