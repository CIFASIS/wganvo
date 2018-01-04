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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time

from debian.debtags import output
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import model
import numpy as np
# Basic model parameters as external flags.
FLAGS = None

def placeholder_inputs(batch_size):
	"""Generate placeholder variables to represent the input tensors.
	These placeholders are used as inputs by the rest of the model building
	code and will be fed from the downloaded data in the .run() loop, below.
	Args:
		batch_size: The batch size will be baked into both placeholders.
	Returns:
		images_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
	"""
	# Note that the shapes of the placeholders match the shapes of the full
	# image and label tensors, except the first dimension is now batch_size
	# rather than the full size of the train or test data sets.
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         input_data.IMAGE_HEIGHT, input_data.IMAGE_WIDTH, 2))
	labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, input_data.LABELS_SIZE))
	return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl, feed_with_batch = False):
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
  if(feed_with_batch):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data,
                                                 False)
  # Create the feed_dict for the placeholders filled with the entire dataset
  else:
    images_feed = data_set.images
    labels_feed = data_set.labels

  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            evaluation,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        evaluation: .
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    sum_squared_errors = 0
    for step in xrange(steps_per_epoch):
	    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder,
                               feed_with_batch=True)
	    squared_errors = sess.run(evaluation, feed_dict=feed_dict)
        #sum_squared_errors += np.sum(squared_errors)
    #sum_squared_errors / step


def do_evaluation(sess,
            outputs,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        evaluation: .
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    evaluation = tf.square(tf.subtract(outputs, labels_placeholder))
    batch_size = FLAGS.batch_size
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    prediction_matrix = np.empty((num_examples, input_data.LABELS_SIZE), dtype="float32")
    accum_squared_errors = np.zeros((batch_size, input_data.LABELS_SIZE), dtype="float32")
    batch_index = 0
    for step in xrange(steps_per_epoch):
      feed_dict = fill_feed_dict(data_set,
                             images_placeholder,
                             labels_placeholder,
                             feed_with_batch=True)
      print(step, batch_index)
      batch_squared_errors, prediction = sess.run([evaluation, outputs], feed_dict=feed_dict)
      accum_squared_errors += batch_squared_errors
      init = batch_index * batch_size
      end = (batch_index + 1) * batch_size
      prediction_matrix[init:end] = prediction
      batch_index += 1
    squared_errors = np.sum(accum_squared_errors, axis = 0)
    mean_squared_errors = squared_errors / num_examples
    print("dtype_pred_matrix", prediction_matrix.dtype)
    variance = np.var(prediction_matrix, axis=0) # variance = std ** 2
    norm_mse = mean_squared_errors / variance
    return mean_squared_errors, norm_mse
    #    print('  RMSE @ 1: %0.04f' % (rmse))
    #    return rmse

def add_array_to_tensorboard(arr, prefix_tagname, summary_writer, step):
    ind = 1
    summary = tf.Summary()
    for std in arr:
        tagname = prefix_tagname + str(ind)
        summary.value.add(tag=tagname, simple_value=std)
        ind += 1
    summary_writer.add_summary(summary, step)
    summary_writer.flush()

def run_training():
  print("START")
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.train_data_dir, FLAGS.test_data_dir, FLAGS.validation_data_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    #train_dataset_images_placeholder, train_dataset_labels_placeholder = placeholder_inputs(
    #    data_sets.train.num_examples)

    # Build a Graph that computes predictions from the inference model.
    outputs = model.inference(images_placeholder)

    # Add to the Graph the Ops for loss calculation.
    loss = model.loss(outputs, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = model.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    #evaluation = model.evaluation(outputs, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder,
                                 True)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      # Write the summaries and print an overview fairly often.
      print(step)

      if step % 100 == 0:
        duration = time.time() - start_time
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        #checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        #saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        mse, norm_mse = do_evaluation(sess,
                outputs,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        add_array_to_tensorboard(mse, "tr_mse_", summary_writer, step)
        add_array_to_tensorboard(norm_mse, "tr_norm_mse_", summary_writer, step)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        mse, norm_mse = do_evaluation(sess,
                outputs,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        add_array_to_tensorboard(mse, "v_mse_", summary_writer, step)
        add_array_to_tensorboard(norm_mse, "v_norm_mse_", summary_writer, step)
        # Evaluate against the test set.
        print('Test Data Eval:')
        mse, norm_mse = do_evaluation(sess,
                outputs,
                images_placeholder,
                labels_placeholder,
                data_sets.test)
        add_array_to_tensorboard(mse, "te_mse_", summary_writer, step)
        add_array_to_tensorboard(norm_mse, "te_norm_mse_", summary_writer, step)


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'train_data_dir',
      type=str,
      default=".",
      help='Directory to put the train data.'
  )
  parser.add_argument(
      'test_data_dir',
      type=str,
      default=".",
      help='Directory to put the test data.'
  )

  parser.add_argument(
      '--validation_data_dir',
      type=str,
      help='Directory to put the test data.'
  )

  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=10000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/jcremona/tesina/logs/'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
