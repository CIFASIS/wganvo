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
    predicted_trajectory = np.empty((num_examples + 1, 3))
    target_trajectory = np.empty((num_examples + 1 ,3))
    pred_current_xyz = np.matrix([0.,0.,0.,1.]).transpose()
    target_current_xyz = np.matrix([0.,0.,0.,1.]).transpose()
    predicted_trajectory[0] = pred_current_xyz[0:3].transpose()
    target_trajectory[0] = target_current_xyz[0:3].transpose()
    for step in xrange(steps_per_epoch):
	print(step)
	feed_dict = fill_feed_dict(dataset, images_placeholder, targets_placeholder, feed_with_batch=True, batch_size=batch_size, shuffle=False, standardize_targets=True)
	prediction_batch, target_batch, images = sess.run([outputs, targets_placeholder, images_placeholder], feed_dict=feed_dict)
        batch_pred_trajectory, batch_target_trajectory = get_trajectories(dataset, batch_size, inverse_intrinsic_matrix, prediction_batch, target_batch, pred_current_xyz, target_current_xyz)
	init = batch_size * step + 1
	end = batch_size * (step + 1) + 1
        predicted_trajectory[init:end] = batch_pred_trajectory
	target_trajectory[init:end] = batch_target_trajectory
	pred_current_xyz = np.matrix([0.,0.,0.,1.])
	pred_current_xyz[:,0:3] = batch_pred_trajectory[-1]
	pred_current_xyz = pred_current_xyz.transpose()
	target_current_xyz = np.matrix([0.,0.,0.,1.])
	target_current_xyz[:,0:3] = batch_target_trajectory[-1]
	target_current_xyz = target_current_xyz.transpose()
	#print(pred_current_xyz)
	#print(batch_pred_trajectory)
	#print(target_current_xyz)
    np.savetxt("prediction.txt", predicted_trajectory, delimiter=' ', fmt='%1.6f')
    np.savetxt("target.txt", target_trajectory, delimiter=' ', fmt='%1.6f')



def get_trajectories(dataset, batch_size, inverse_intrinsic_matrix, prediction_batch, target_batch, pred_current_xyz, target_current_xyz):
    #pred_current_xyz = np.matrix([0.,0.,0.,1.]).transpose()
    #target_current_xyz = np.matrix([0.,0.,0.,1.]).transpose()
    predicted_trajectory = np.empty((batch_size, 3))
    #predicted_trajectory[0] = pred_current_xyz[0:3].transpose()
    target_trajectory = np.empty((batch_size,3))
    inv = False
    #target_trajectory[0] = target_current_xyz[0:3].transpose()
    for i in range(batch_size):
	#print("--------------------------------------------------------------------------------------")
        prediction = prediction_batch[i]
	prediction = prediction * dataset.targets_std + dataset.targets_mean
	prediction = np.insert(prediction, -1, 1.)
	prediction = prediction.reshape(3,4)
	pred_transformation = inverse_intrinsic_matrix * prediction
	u,_ = linalg.polar(pred_transformation[0:3,0:3])
	pred_transf_correction = np.matlib.identity(4)
	pred_transf_correction[0:3, 0:3] = u
	pred_transf_correction[0:3, 3] = pred_transformation[0:3,3]
	#pred_transf_correction = pred_transf_correction[0:3,:]
	if inv:
		pred_transf_correction = np.linalg.inv(pred_transf_correction)
	pred_current_xyz = pred_transf_correction * pred_current_xyz
	predicted_trajectory[i] = pred_current_xyz[0:3].transpose()
        target = target_batch[i]
	target = target * dataset.targets_std + dataset.targets_mean
	target = np.insert(target, -1, 1.)
	target = target.reshape(3,4)
        temp_target_transformation = inverse_intrinsic_matrix * target
	target_transformation = np.matlib.identity(4)
	target_transformation[0:3,:] = temp_target_transformation
	if inv:
		target_transformation = np.linalg.inv(target_transformation)
	target_current_xyz = target_transformation * target_current_xyz
	target_trajectory[i] = target_current_xyz[0:3].transpose()
	print("PREDICTION")
        print(pred_transformation)
	print("CORRECTED [R|t]")
	print(pred_transf_correction[0:3,:])
	print("TARGET [R|t]")
        print(target_transformation)
	
    return predicted_trajectory, target_trajectory

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
    

