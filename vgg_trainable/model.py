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

import vgg
import tensorflow as tf

def inference(images):
  """Build the model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  v = vgg.Vgg19()
  return v.build(images)




def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable. We instead use tf.nn.softmax_cross_entropy_with_logits
  #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
  #    labels=labels, logits=logits, name='xentropy')
  #return tf.reduce_mean(cross_entropy, name='xentropy_mean')
  p_matrix = tf.reshape(logits, [3,4])
  r_matrix = p_matrix[:3, :3]
  n_id = tf.matmul(r_matrix, r_matrix, transpose_b = True)
  identity = tf.eye(3)
  alpha = 1
  cost = euclidean_distance(n_id, identity) * alpha
  return rmse(labels, logits) + cost


def euclidean_distance(a, b):
  return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(a, b))))

def rmse(labels, logits):
  return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, logits))))

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
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  #correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  #return tf.reduce_sum(tf.cast(correct, tf.int32))
  return loss(logits,labels)
