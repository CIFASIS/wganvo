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

#from debian.debtags import output
from six.moves import xrange  # pylint: disable=redefined-builtin

# Scipy
from scipy import linalg

#from ... import transform 
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
#import sys
#from os import path
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from transform import se3_to_components
from array_utils import load
import tensorflow as tf
import input_data
import model
import numpy as np
import numpy.matlib as matlib
# Basic model parameters as external flags.
FLAGS = None
DEFAULT_INTRINSIC_FILE_NAME = "intrinsic_matrix.txt"

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
                                                 shuffle=True)
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
            data_set,
	    k_matrix):
	   # target_variance_vector):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        evaluation: .
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    rows_reshape = 3
    columns_reshape = 4
    components_vector_size = 6
    evaluation = tf.square(tf.subtract(outputs, labels_placeholder))
    batch_size = FLAGS.batch_size
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    #prediction_matrix = np.empty((num_examples, components_vector_size), dtype="float32")
    target_matrix = np.empty((num_examples, components_vector_size), dtype="float32")
    #accum_squared_errors = np.zeros((batch_size, input_data.LABELS_SIZE), dtype="float32")
    squared_errors = np.zeros(components_vector_size, dtype="float32")
    inv_k_matrix = np.linalg.inv(k_matrix)
    for step in xrange(steps_per_epoch):
      feed_dict = fill_feed_dict(data_set,
                             images_placeholder,
                             labels_placeholder,
                             feed_with_batch=True)
      prediction, target = sess.run([outputs, labels_placeholder], feed_dict=feed_dict)
      #accum_squared_errors += batch_squared_errors
      init = step * batch_size
      end = (step + 1) * batch_size
      #prediction_matrix[init:end] = prediction
      #target_matrix[init:end] = target
      for i in xrange(batch_size):
	assert init+i < end
	index = init+i
	current_prediction = prediction[i].reshape(rows_reshape,columns_reshape)
	# P = K * [R|t] => [R|t] = K^(-1) * P
	curr_pred_transform_matrix = inv_k_matrix * current_prediction
	current_target = target[i].reshape(rows_reshape,columns_reshape)
	curr_target_transform_matrix = inv_k_matrix * current_target
	# Get the closest rotation matrix
	u,_ = linalg.polar(curr_pred_transform_matrix[0:3, 0:3])
	# Replace the non-orthogonal R matrix obtained from the prediction with the closest rotation matrix
	closest_curr_pred_s3_matrix = matlib.identity(4)
	closest_curr_pred_s3_matrix[0:3, 0:3] = u
	closest_curr_pred_s3_matrix[0:3, 3] = curr_pred_transform_matrix[0:3,3]
	curr_target_s3_matrix = np.concatenate([curr_target_transform_matrix, [[0,0,0,1]]], axis=0)
	# From [R|t] matrix to components
	# components = [x,y,z, roll, pitch, yaw]
	curr_pred_components = se3_to_components(closest_curr_pred_s3_matrix)
	curr_target_components = se3_to_components(curr_target_s3_matrix)
	curr_squared_error = np.square(curr_pred_components-curr_target_components)
	squared_errors += curr_squared_error
	#prediction_matrix[i] = curr_pred_components
        target_matrix[index] = curr_target_components

    mean_squared_errors = squared_errors / num_examples
    rmse = np.sqrt(np.sum(squared_errors) / num_examples)
    target_variance = np.var(target_matrix, axis=0) # variance = std ** 2
    norm_mse = mean_squared_errors / target_variance
    print("mse")
    print(mean_squared_errors)
    print("var")
    print(target_variance)
    print("norm_mse")
    print(norm_mse)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return rmse, mean_squared_errors, norm_mse

def add_array_to_tensorboard(arr, prefix_tagname, summary_writer, step):
    ind = 1
    summary = tf.Summary()
    for std in arr:
        tagname = prefix_tagname + str(ind)
        summary.value.add(tag=tagname, simple_value=std)
        ind += 1
    summary_writer.add_summary(summary, step)
    summary_writer.flush()

def add_scalar_to_tensorboard(value, tagname, summary_writer, step):
    summary = tf.Summary()
    summary.value.add(tag=tagname, simple_value=value)
    summary_writer.add_summary(summary, step)

def run_training():
  print("START")
  #se3_to_components(np.array([1,2,3]))
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.train_data_dir, FLAGS.test_data_dir, FLAGS.validation_data_dir, FLAGS.fake_data)
  intrinsic_matrix = np.matrix(load(FLAGS.intrinsics_dir))
  # Tell TensorFlow that the model will be built into the default Graph.
  print("Learning rate: " + str(FLAGS.learning_rate))
  print("Steps: " + str(FLAGS.max_steps))
  print("Batch size: " + str(FLAGS.batch_size))
  print(FLAGS)
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    #train_dataset_images_placeholder, train_dataset_labels_placeholder = placeholder_inputs(
    #    data_sets.train.num_examples)

    # Build a Graph that computes predictions from the inference model.
    outputs = model.inference(images_placeholder)

    #train_targets_variance = np.var(data_sets.train.labels, axis=0)
    #(X- np.mean(X, axis=0)) / np.std(X,axis=0) #Guardar la media y el std para volver a los valores originales
    #test_targets_variance = np.var(data_sets.test.labels, axis=0)
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
    total_start_time = time.time()

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
      #print(step)

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
        rmse, mse, norm_mse = do_evaluation(sess,
                outputs,
                images_placeholder,
                labels_placeholder,
                data_sets.train,
		intrinsic_matrix)
	# FIXME renombrar vbles y crear un metodo para agregar un unico escalar a tensorboard
	add_scalar_to_tensorboard(rmse, "tr_rmse", summary_writer, step)
        add_array_to_tensorboard(mse, "tr_mse_", summary_writer, step)
        add_array_to_tensorboard(norm_mse, "tr_norm_mse_", summary_writer, step)
        # Evaluate against the validation set.
        #print('Validation Data Eval:')
        #rmse, mse, norm_mse = do_evaluation(sess,
        #        outputs,
        #        images_placeholder,
        #        labels_placeholder,
        #        data_sets.validation,
	#	intrinsic_matrix)
        #add_array_to_tensorboard(mse, "v_mse_", summary_writer, step)
        #add_array_to_tensorboard(norm_mse, "v_norm_mse_", summary_writer, step)
        # Evaluate against the test set.
        print('Test Data Eval:')
        rmse, mse, norm_mse = do_evaluation(sess,
                outputs,
                images_placeholder,
                labels_placeholder,
                data_sets.test,
		intrinsic_matrix)
	add_scalar_to_tensorboard(rmse, "te_rmse", summary_writer, step)
        add_array_to_tensorboard(mse, "te_mse_", summary_writer, step)
        add_array_to_tensorboard(norm_mse, "te_norm_mse_", summary_writer, step)
    total_duration = time.time() - total_start_time
    print('Total: %.3f sec' % (total_duration))

def main(_):
  #if tf.gfile.Exists(FLAGS.log_dir):
  
    #tf.gfile.DeleteRecursively(FLAGS.log_dir)
  #tf.gfile.MakeDirs(FLAGS.log_dir)
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
      '--intrinsics_dir',
      type=str,
      default=os.path.join(os.getcwd(), DEFAULT_INTRINSIC_FILE_NAME),
      help='Intrinsic matrix path'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
