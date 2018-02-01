import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from input_data import read_data_sets, DataSet
from main import fill_feed_dict

def test_model(model_name, data_dir):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_name+".meta")
    print(model_name)
    saver.restore(sess, model_name)#tf.train.latest_checkpoint('./'))  
    graph = tf.get_default_graph()
    outputs = graph.get_tensor_by_name("outputs:0")
    targets_placeholder = graph.get_tensor_by_name("targets_placeholder:0")
    images_placeholder = graph.get_tensor_by_name("images_placeholder:0") 
    images, targets, _ = read_data_sets(data_dir)
    feed_dict = fill_feed_dict(DataSet(images, targets, fake_data=False), images_placeholder, targets_placeholder, feed_with_batch=True,batch_size=100, standardize_targets=False)
    # es necesario correr images_placeholder???
    prediction_batch, target_batch, images = sess.run([outputs, targets_placeholder, images_placeholder], feed_dict=feed_dict)
    i1 = images[0,...,0] * 255.
    i2 = images[0,...,1] * 255.
    print(i1)
    print(i2)
    show_images([i1,i2], titles = ["frame 1", "frame 2"])

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
    test_model(FLAGS.model_name, FLAGS.data_dir)

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
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

