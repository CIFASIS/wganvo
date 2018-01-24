import tensorflow as tf
import matplotlib.pyplot as plt

def test_model():
    name = ""
    sess = tf.Session()
    saver = tf.train.import_meta_graph(name)
    saver.restore(sess, tf.train.latest_checkpoint('./'))
     

    



