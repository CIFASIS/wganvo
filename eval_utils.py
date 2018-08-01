import sys, os, inspect

import numpy as np
import transformations


def get_relative_poses(sess, dataset, batch_size, images_placeholder, outputs,
                       targets_placeholder):
    steps_per_epoch = dataset.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    relative_poses_prediction = np.empty((num_examples, 3, 4))
    relative_poses_target = np.empty((num_examples, 3, 4))
    standardize_targets = False
    #        rmse, mse, norm_mse = do_evaluation(sess,outputs,images_placeholder, targets_placeholder, dataset, batch_size, intrinsic_matrix, True)
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(dataset, images_placeholder, targets_placeholder, feed_with_batch=True,
                                   batch_size=batch_size, shuffle=False, standardize_targets=standardize_targets)
        prediction_batch, target_batch = sess.run([outputs, targets_placeholder], feed_dict=feed_dict)
        batch_relative_poses_pred, batch_relative_poses_target = get_trajectories(dataset, batch_size,
                                                                                  prediction_batch, target_batch,
                                                                                  standardize_targets)
        init = batch_size * step
        end = batch_size * (step + 1)
        relative_poses_prediction[init:end] = batch_relative_poses_pred
        relative_poses_target[init:end] = batch_relative_poses_target
    return relative_poses_prediction, relative_poses_target


def get_absolute_poses(relative_poses):
    current = np.matrix(np.identity(4))
    num_examples = relative_poses.shape[0]
    absolute_poses = np.empty(shape=relative_poses.shape)
    for i in xrange(num_examples):
        T = np.matrix(np.identity(4))
        T[0:3, :] = relative_poses[i]
        current = current * np.linalg.inv(T)
        absolute_poses[i] = current[0:3, :]
    return absolute_poses

def get_trajectories(dataset, batch_size, prediction_batch, target_batch,
                     standardize_targets):
    poses_prediction = np.empty((batch_size, 3, 4))
    poses_target = np.empty((batch_size, 3, 4))
    for i in xrange(batch_size):
        prediction = prediction_batch[i]
        # Original scale
        if standardize_targets:
            prediction = prediction * dataset.targets_std + dataset.targets_mean

        # prediction = prediction.reshape(3,4)
        # pred_transformation = inverse_intrinsic_matrix * prediction
        # u,_ = linalg.polar(pred_transformation[0:3,0:3])
        # pred_transf_correction = np.empty((3,4))
        # pred_transf_correction[0:3, 0:3] = u
        # pred_transf_correction[0:3, 3] = pred_transformation[0:3,3].transpose()

        target = target_batch[i]
        if standardize_targets:
            target = target * dataset.targets_std + dataset.targets_mean
        # target = target.reshape(3,4)
        # target_transformation = inverse_intrinsic_matrix * target
        # poses_prediction[i] = pred_transf_correction.reshape(12)
        # poses_target[i] = target_transformation.reshape(12)

        poses_prediction[i] = x_q_to_mtx(prediction)
        poses_target[i] = x_q_to_mtx(target)

    return poses_prediction, poses_target


def x_q_to_mtx(xq):
    mtx = transformations.quaternion_matrix(xq[3:7])
    mtx[0:3, 3] = xq[0:3]
    out = mtx[0:3, :]
    return out  # .reshape(12)


def fill_feed_dict(data_set, images_pl, labels_pl, feed_with_batch=False, batch_size=None, shuffle=True,
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
        images_feed, labels_feed = data_set.next_batch(batch_size,
                                                       fake_data,
                                                       shuffle=shuffle,
                                                       standardize_targets=standardize_targets)
    # Create the feed_dict for the placeholders filled with the entire dataset
    else:
        images_feed = data_set.images
        labels_feed = data_set.labels

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict
