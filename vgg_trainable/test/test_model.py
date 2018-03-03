import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import argparse
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from input_data import read_data_sets, DataSet
from main import fill_feed_dict

DEFAULT_INTRINSIC_FILE_NAME = "intrinsic_matrix.txt"

def test_model(model_name, intrinsic_matrix, data_dir):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_name+".meta")
    #print(model_name)
    inverse_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)
    saver.restore(sess, model_name)#tf.train.latest_checkpoint('./'))  
    graph = tf.get_default_graph()
    outputs = graph.get_tensor_by_name("outputs:0")
    targets_placeholder = graph.get_tensor_by_name("targets_placeholder:0")
    images_placeholder = graph.get_tensor_by_name("images_placeholder:0") 
    batch_size = 100
    images, targets, _ = read_data_sets(data_dir)
    dataset = DataSet(images, targets, fake_data=False)
    steps_per_epoch = dataset.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    relative_poses_prediction = np.empty((num_examples, 12))
    relative_poses_target = np.empty((num_examples, 12))
    for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(dataset, images_placeholder, targets_placeholder, feed_with_batch=True, batch_size=batch_size, shuffle=False, standardize_targets=True)
		prediction_batch, target_batch, images = sess.run([outputs, targets_placeholder, images_placeholder], feed_dict=feed_dict)
		batch_relative_poses_pred, batch_relative_poses_target = get_trajectories(dataset, batch_size, inverse_intrinsic_matrix, prediction_batch, target_batch)
		init = batch_size * step
		end = batch_size * (step + 1)
		relative_poses_prediction[init:end] = batch_relative_poses_pred
		relative_poses_target[init:end] = batch_relative_poses_target
    np.savetxt("relative_poses_prediction.txt", relative_poses_prediction, delimiter=' ')
    np.savetxt("relative_poses_target.txt", relative_poses_target, delimiter=' ')
	absolute_poses_prediction = get_absolute_poses(relative_poses_prediction.reshape((num_examples,3,4)))
	absolute_poses_target = get_absolute_poses(relative_poses_target.reshape((num_examples,3,4)))
    np.savetxt("absolute_poses_prediction.txt", absolute_poses_prediction, delimiter=' ')
    np.savetxt("absolute_poses_target.txt", absolute_poses_target, delimiter=' ')


def get_trajectories(dataset, batch_size, inverse_intrinsic_matrix, prediction_batch, target_batch):
    poses_prediction = np.empty((batch_size, 12))
    poses_target = np.empty((batch_size,12))
    for i in xrange(batch_size):
		prediction = prediction_batch[i]
		# Original scale
		prediction = prediction * dataset.targets_std + dataset.targets_mean
		# 
		prediction = np.insert(prediction, -1, 1.)
		prediction = prediction.reshape(3,4)
		pred_transformation = inverse_intrinsic_matrix * prediction
		u,_ = linalg.polar(pred_transformation[0:3,0:3])
		pred_transf_correction = np.empty((3,4))
		pred_transf_correction[0:3, 0:3] = u
		pred_transf_correction[0:3, 3] = pred_transformation[0:3,3]

		target = target_batch[i]
		target = target * dataset.targets_std + dataset.targets_mean
		target = np.insert(target, -1, 1.)
		target = target.reshape(3,4)
		target_transformation = inverse_intrinsic_matrix * target
		
		poses_prediction[i] = pred_transf_correction.reshape(12)
		poses_target[i] = target_transformation.reshape(12)
    return poses_prediction, poses_target


def get_absolute_poses(relative_poses):
	current = np.matrix(np.identity(4))
	num_examples = relative_poses.shape[0]
	absolute_poses = np.empty(shape=relative_poses.shape)
	for i in xrange(num_examples):
		current = current * relative_poses[i]
		absolute_poses[i] = current
	return absolute_poses

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def main(_):
    intrinsic_matrix = np.matrix(np.loadtxt(FLAGS.intrinsics_path, delimiter=' '))
    test_model(FLAGS.model_name, intrinsic_matrix, FLAGS.data_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_name',
        type=str,
        help='Model name'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory to put the data'
    )
    parser.add_argument(
      '--intrinsics_path',
      type=str,
      default=os.path.join(os.getcwd(), DEFAULT_INTRINSIC_FILE_NAME),
      help='Intrinsic matrix path'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

