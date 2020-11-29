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

import sys, os, inspect

import numpy as np
import transformations
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trajectory


def infer_relative_poses(sess, dataset, batch_size, images_placeholder, outputs,
                         targets_placeholder, train_mode=None):
    steps_per_epoch = dataset.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    relative_poses_prediction = np.empty((num_examples, 3, 4))
    relative_poses_target = np.empty((num_examples, 3, 4))
    standardize_targets = False
    #        rmse, mse, norm_mse = do_evaluation(sess,outputs,images_placeholder, targets_placeholder, dataset, batch_size, True)
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(dataset, images_placeholder, targets_placeholder, feed_with_batch=True,
                                   batch_size=batch_size, shuffle=False, standardize_targets=standardize_targets)
        if train_mode is not None:
            feed_dict[train_mode] = False
        prediction_batch, target_batch = sess.run([outputs, targets_placeholder], feed_dict=feed_dict)
        batch_relative_poses_pred = get_transformation_matrices(dataset, batch_size,
                                                                prediction_batch,
                                                                standardize_targets)
        batch_relative_poses_target = get_transformation_matrices(dataset, batch_size,
                                                                  target_batch,
                                                                  standardize_targets)
        init = batch_size * step
        end = batch_size * (step + 1)
        relative_poses_prediction[init:end] = batch_relative_poses_pred
        relative_poses_target[init:end] = batch_relative_poses_target
    if train_mode is not None:
        print("Train Mode: " + str(sess.run(train_mode, feed_dict)))
    return relative_poses_prediction, relative_poses_target


def get_absolute_poses(relative_poses, inv=False):
    current = np.matrix(np.identity(4))
    num_examples = relative_poses.shape[0]
    absolute_poses = np.empty(shape=relative_poses.shape)
    for i in xrange(num_examples):
        T = np.matrix(np.identity(4))
        T[0:3, :] = relative_poses[i]
        if inv:
            T = np.linalg.inv(T)
        current = current * T
        absolute_poses[i] = current[0:3, :]
    return absolute_poses


def get_transformation_matrices(dataset, batch_size, batch,
                                standardize_targets):
    transformation_matrices = np.empty((batch_size, 3, 4))
    # poses_target = np.empty((batch_size, 3, 4))
    for i in xrange(batch_size):
        transformation = batch[i]
        # Original scale
        if standardize_targets:
            transformation = transformation * dataset.targets_std + dataset.targets_mean

        # prediction = prediction.reshape(3,4)
        # pred_transformation = inverse_intrinsic_matrix * prediction
        # u,_ = linalg.polar(pred_transformation[0:3,0:3])
        # pred_transf_correction = np.empty((3,4))
        # pred_transf_correction[0:3, 0:3] = u
        # pred_transf_correction[0:3, 3] = pred_transformation[0:3,3].transpose()

        # target = target_batch[i]
        # if standardize_targets:
        #    target = target * dataset.targets_std + dataset.targets_mean
        # target = target.reshape(3,4)
        # target_transformation = inverse_intrinsic_matrix * target
        # poses_prediction[i] = pred_transf_correction.reshape(12)
        # poses_target[i] = target_transformation.reshape(12)

        transformation_matrices[i] = vector_to_transformation_mtx(transformation)
        # poses_target[i] = x_q_to_mtx(target)

    return transformation_matrices


def vector_to_transformation_mtx(xq):
    mtx = transformations.quaternion_matrix(xq[3:])
    mtx[0:3, 3] = xq[0:3]
    out = mtx[0:3, :]
    return out  # .reshape(12)


def fill_feed_dict(data_set, images_pl, labels_pl, points_pl=None, feed_with_batch=False, batch_size=None, shuffle=True,
                   standardize_targets=False, fake_data=False):
    """Fills the feed_dict for training the given step or for evaluating the entire dataset.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    if (feed_with_batch):
        if (batch_size is None):
            raise ValueError("batch_size not specified")
        images_feed, labels_feed, points = data_set.next_batch(batch_size,
                                                       fake_data,
                                                       shuffle=shuffle,
                                                       standardize_targets=standardize_targets)
    # Create the feed_dict for the placeholders filled with the entire dataset
    else:
        images_feed = data_set.images
        labels_feed = data_set.labels
        points = data_set.points

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        #points_pl: points,
    }
    if points_pl is not None:
        feed_dict[points_pl] = points
    return feed_dict


def plot_frames_vs_abs_distance(relative_poses_prediction, relative_poses_target, dataset, output_dir, save_txt=False,
                                plot=False, samples=30):
    groups = dataset.groups
    datasets_idxs = {}
    for i, _ in enumerate(relative_poses_prediction):
        group = str(groups[i])
        if group in datasets_idxs:
            datasets_idxs[group].append(i)
        else:
            datasets_idxs[group] = [i]
    # acc_rmse_tr = 0.
    # acc_rmse_rot = 0.
    X_axis = []
    Y_axis = []
    for grp, idxs in datasets_idxs.iteritems():
        relative_prediction = relative_poses_prediction[idxs]
        relative_target = relative_poses_target[idxs]
        max_num_of_frames = len(relative_prediction)
        assert max_num_of_frames == len(relative_target)
        # Get SAMPLES sub-trajectories from sequence
        for i in xrange(samples):
            # Random sub-trajectory
            N = random.randint(1, max_num_of_frames)
            start = random.randint(0, max_num_of_frames - N)
            traslation_error = get_traslation_error(relative_prediction[start:start + N],
                                                    relative_target[start:start + N])
            assert len(traslation_error) == N
            d = traslation_error[-1]
            X_axis.append(N)
            Y_axis.append(d)
            if save_txt:
                np.savetxt(os.path.join(output_dir, 'abs_poses_target_{}.txt'.format(grp)),
                           get_absolute_poses(relative_target).reshape(-1, 12))
                np.savetxt(os.path.join(output_dir, 'abs_poses_prediction_{}.txt'.format(grp)),
                           get_absolute_poses(relative_prediction).reshape(-1, 12))
                # print("Num of frames")
                # print(N)
                # print("d")
                # print(d)

                # if save_txt:
                #    np.savetxt(os.path.join(output_dir, 'orig_relative_target.txt'), relative_poses_target.reshape(-1, 12))
                #    np.savetxt(os.path.join(output_dir, 'orig_relative_prediction.txt'), relative_poses_prediction.reshape(-1, 12))
                # rmse_tr, rmse_rot = calc_trajectory_rmse(relative_poses_prediction[idxs], relative_poses_target[idxs])
                # print('*' * 50)
                # print(grp, len(idxs))
                # print(rmse_tr, rmse_rot)
                # acc_rmse_tr += rmse_tr
                # acc_rmse_rot += rmse_rot
    if plot:
        fig, ax = plt.subplots()
        ax.plot(X_axis, Y_axis, 'r.')
        fig.savefig(os.path.join(output_dir, 'f_vs_d.png'))
    return X_axis, Y_axis
    # return acc_rmse_tr / len(datasets_idxs), acc_rmse_rot / len(datasets_idxs)


def get_traslation_error(relative_poses_prediction, relative_poses_target):
    absolute_poses_prediction = get_absolute_poses(relative_poses_prediction).reshape(-1, 12)
    absolute_poses_target = get_absolute_poses(relative_poses_target).reshape(-1, 12)
    poses_prediction = se3_pose_list(absolute_poses_prediction)
    poses_target = se3_pose_list(absolute_poses_target)
    poses_prediction = trajectory.PosePath3D(poses_se3=poses_prediction)
    poses_target = trajectory.PosePath3D(poses_se3=poses_target)
    E_tr = poses_prediction.positions_xyz - poses_target.positions_xyz
    traslation_error = [np.linalg.norm(E_i) for E_i in E_tr]
    return traslation_error


def se3_pose_list(kitti_format):
    return [np.array([[r[0], r[1], r[2], r[3]],
                      [r[4], r[5], r[6], r[7]],
                      [r[8], r[9], r[10], r[11]],
                      [0, 0, 0, 1]]) for r in kitti_format]


def our_metric_evaluation(relative_prediction, relative_target, test_dataset, curr_fold_log_path,
                          save_txt):
    frames, abs_distance = plot_frames_vs_abs_distance(relative_prediction, relative_target, test_dataset,
                                                       curr_fold_log_path, save_txt=save_txt)
    frames = np.array(frames)
    abs_distance = np.array(abs_distance)
    te_eval = np.mean(np.square(np.log(abs_distance) / np.log(frames + 1)))
    return te_eval
