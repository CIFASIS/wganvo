# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.
Implements the inference/loss/training pattern for model building.
1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

MATRIX_MATCH_TOLERANCE = 1e-4


def inference(images, train_mode, pruned_vgg=False, pooling_type="max", activation_function="relu"):
    """Build the model up to where it may be used for inference.
    Args:
      images: Images placeholder, from inputs().
    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    import vgg
    v = vgg.Vgg19(int(images.shape[2]), int(images.shape[1]), activation_function=activation_function)
    if pruned_vgg:
        return v.build_pruned_vgg(images)
    return v.build(images, train_mode, pooling_type=pooling_type)


# FIXME revisar
def rmse(outputs, targets):
    return tf.sqrt(tf.reduce_mean(squared_error(outputs, targets)))


def loss(outputs, targets):
    """Calculates the loss from the logits and the labels.
    Args:
      output:
      target:
    Returns:
      loss: Loss tensor of type float.
    """
    return kendall_loss_naive(outputs, targets)


def kendall_loss_uncertainty(outputs, targets, sx, sq):
    outputs_x, outputs_q = split_x_q(outputs)
    targets_x, targets_q = split_x_q(targets)
    loss_x = tf.norm(outputs_x - targets_x, axis=1)
    #q_norm = tf.norm(outputs_q, axis=1)
    #loss_q = tf.norm(targets_q - outputs_q / tf.reshape(q_norm, (-1, 1)), axis=1)
    dot = tf.reduce_sum(tf.multiply(targets_q, outputs_q), 1, keepdims=True)
    loss_q = 2 * tf.acos(tf.abs(dot))
    print(loss_x.shape)
    print(loss_q.shape)
    noise_x = tf.exp(-sx)
    noise_q = tf.exp(-sq)
    tf.summary.scalar("sx", sx)
    tf.summary.scalar("sq", sq)
    tf.summary.scalar("noise_x", noise_x)
    tf.summary.scalar("noise_q", noise_q)
    return tf.reduce_mean(loss_x * noise_x + sx + loss_q * noise_q + sq)

def acos(x):
    return (-0.69813170079773212 * x * x - 0.87266462599716477) * x + 1.5707963267948966

def kendall_loss_naive(outputs, targets):
    outputs_x, outputs_q = split_x_q(outputs)
    targets_x, targets_q = split_x_q(targets)
    loss_x = tf.norm(outputs_x - targets_x, axis=1)
    #absolute_x = tf.reduce_mean(tf.abs(tf.subtract(outputs_x, targets_x)))
    #q_norm = tf.norm(outputs_q, axis=1)
    #loss_q = tf.norm(targets_q - outputs_q / tf.reshape(q_norm, (-1, 1)), axis=1)
    #dot = tf.reduce_sum(tf.multiply(targets_q, outputs_q), 1, keepdims=True)
    #loss_q = 2 * acos(tf.abs(dot))
    loss_q = tf.norm(targets_q - outputs_q, axis=1)
    beta = 100
    #tf.summary.scalar("x_cost", loss_x)
    #tf.summary.scalar("abs_x_cost", absolute_x)
    #tf.summary.scalar("q_scaled_cost", beta * loss_q)
    return tf.reduce_mean(loss_x + beta * loss_q ) # tf.reduce_mean(tf.abs(tf.subtract(outputs, targets)))


def split_x_q(batch):
    print(batch.shape)
    x = batch[:, 0:3]
    q = batch[:, 3:7]
    return x, q


def loss_(logits, labels):
    components = tf.Variable([])
    i = tf.constant(0)
    while_condition = lambda i, p: tf.less(i, logits.shape[0])

    def body(i, pred_components):
        p_matrix = tf.reshape(logits[i], [3, 4])
        cost = get_cost(p_matrix)
        res = tf.concat([pred_components, se3_to_components(p_matrix)], 0)
        return i + 1, res

    r, pred_components = tf.while_loop(while_condition, body, [i, components], shape_invariants=[i.get_shape(),
                                                                                                 tf.TensorShape(
                                                                                                     [None])])
    return rmse(labels, tf.convert_to_tensor(pred_components))  # + cost


def get_cost(pred):
    r_matrix = pred[:3, :3]
    n_id = tf.matmul(r_matrix, r_matrix, transpose_b=True)
    identity = tf.eye(3)
    alpha = 1
    return euclidean_distance(n_id, identity) * alpha


def se3_to_components(se3):
    # xyzrpy[0:3]
    xyz = tf.transpose(se3[0:3, 3])  # .transpose()
    # xyzrpy[3:6] \
    rpy = so3_to_euler(se3[0:3, 0:3])

    return tf.concat([xyz, rpy], 0)


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """

    R_x = tf.stack([tf.stack([1., 0., 0.]),
                    tf.stack([0., tf.cos(rpy[0]), tf.negative(tf.sin(rpy[0]))]),
                    tf.stack([0., tf.sin(rpy[0]), tf.cos(rpy[0])])])
    R_y = tf.stack([tf.stack([tf.cos(rpy[1]), 0., tf.sin(rpy[1])]),
                    tf.stack([0., 1., 0.]),
                    tf.stack([tf.negative(tf.sin(rpy[1])), 0., tf.cos(rpy[1])])])
    R_z = tf.stack([tf.stack([tf.cos(rpy[2]), tf.negative(tf.sin(rpy[2])), 0.]),
                    tf.stack([tf.sin(rpy[2]), tf.cos(rpy[2]), 0.]),
                    tf.stack([0., 0., 1.])])
    R_zyx = tf.matmul(tf.matmul(R_z, R_y), R_x)
    return R_zyx


def so3_to_euler(so3):
    # if so3.shape != (3, 3):
    #    raise ValueError("SO3 matrix must be 3x3")
    roll = tf.atan2(so3[2, 1], so3[2, 2])
    yaw = tf.atan2(so3[1, 0], so3[0, 0])
    denom = tf.sqrt(tf.add(tf.pow(so3[0, 0], 2), tf.pow(so3[1, 0], 2)))
    pitch_poss = [tf.atan2(-so3[2, 0], denom), tf.atan2(-so3[2, 0], -denom)]

    R = euler_to_so3((roll, pitch_poss[0], yaw))

    def throw_exc(): print("Error")

    # raise ValueError("Could not find valid pitch angle")
    def true_fn(): return tf.stack([roll, pitch_poss[0], yaw])

    def false_fn():
        R = euler_to_so3((roll, pitch_poss[1], yaw))
        return tf.cond(tf.reduce_sum(tf.subtract(so3, R)) > MATRIX_MATCH_TOLERANCE, lambda: tf.constant([1.]),
                       lambda: tf.stack([roll, pitch_poss[1], yaw]))

    return tf.cond(tf.reduce_sum(tf.subtract(so3, R)) < MATRIX_MATCH_TOLERANCE, true_fn, false_fn)


def euclidean_distance(a, b):
    return tf.sqrt(tf.reduce_sum(squared_error(a, b)))


def mse_norm(outputs, targets, variance):
    return tf.reduce_mean(tf.reduce_mean(squared_error(outputs, targets), axis=0) / variance)


def squared_error(a, b):
    return tf.square(tf.subtract(a, b))


def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    tf.summary.scalar('learning_rate', learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# Deprecated
def evaluation(outputs, targets):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    outputs: [batch_size, NUM_CLASSES].
    targets: [batch_size]
    Returns:

    """
    return squared_error(outputs, targets)
