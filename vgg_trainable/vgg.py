import tensorflow as tf

import numpy as np
from functools import reduce

#VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, width, height, vgg19_npy_path=None, trainable=True, dropout=0.5, activation_function="relu"):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.activation_function = activation_function
	self.width = width
	self.height = height

    def build(self, images, train_mode=None, pooling_type="max"):
        """
        load variable from npy to build the VGG
        :param images: [batch, height, width, 1] (usually a placeholder)
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        self.conv1_1 = self.conv_layer(images, 2, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.pooling(self.conv1_2, 'pool1', pooling_type=pooling_type)

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.pooling(self.conv2_2, 'pool2', pooling_type=pooling_type)

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.pooling(self.conv3_4, 'pool3', pooling_type=pooling_type)

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.pooling(self.conv4_4, 'pool4', pooling_type=pooling_type)

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.pooling(self.conv5_4, 'pool5', pooling_type=pooling_type)

        fc_in_size = ((self.width // (2 ** 5)) * (self.height // (2 ** 5))) * 512 # (las conv_layer mantienen el ancho y alto, y los max_pool lo reducen a la mitad. Hay 5 max pool)
        self.fc6 = self.fc_layer(self.pool5, fc_in_size, 4096, "fc6")
        self.relu6 = self.activation_function(self.fc6, act_funct=self.activation_function)#tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = self.activation_function(self.fc7, act_funct=self.activation_function)#tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 12, "fc8")

        #self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        return self.fc8


    def build_pruned_vgg(self, images, train_mode=None):
        """
        load variable from npy to build the VGG
        :param images: [batch, height, width, 1] (usually a placeholder)
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        self.conv1_1 = self.conv_layer(images, 2, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.pooling(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.pooling(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.pool3 = self.pooling(self.conv3_2, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.pool4 = self.pooling(self.conv4_2, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.pool5 = self.pooling(self.conv5_2, 'pool5')

        fc_in_size = ((self.width // (2 ** 5)) * (self.height // (2 ** 5))) * 512  # (las conv_layer mantienen el ancho y alto, y los max_pool lo reducen a la mitad. Hay 5 max pool)
        self.fc_in = tf.reshape(self.pool5, [-1, fc_in_size])
        if train_mode is not None:
            self.fc_in = tf.cond(train_mode, lambda: tf.nn.dropout(self.fc_in, self.dropout), lambda: self.fc_in)
        elif self.trainable:
            self.fc_in = tf.nn.dropout(self.fc_in, self.dropout)

        self.output = self.fc_layer(self.fc_in, fc_in_size, 12, "fc8")
        self.data_dict = None
        return self.output

    def build_non_deep_nn(self, images):
	self.conv1_1 = self.conv_layer(images, 2, 32, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 32, 32, "conv1_2")
	self.pool1 = self.max_pool(self.conv1_2, 'pool1')
	fc_in_size = ((self.width // 2) * (self.height // 2)) * 32
	print(fc_in_size)
	self.fc = self.fc_layer(self.pool1, fc_in_size, 12, "fc")
	self.data_dict = None
	return self.fc

	def pooling(self, bottom, name, pooling_type="max"):
        if pooling_type == "avg":
            return self.avg_pool(bottom, name)
        return self.max_pool(bottom, name)

    def avg_pool(self, bottom, name):
        print("Using avg pool")
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        print("Using max pool")
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def activation_function(self, features, act_function="relu"):
        if act_function == "leaky_relu":
            print("Using leaky relu")
            return tf.nn.leaky_relu(features)
        print("Using relu")
        return tf.nn.relu(features)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            act_funct = self.activation_function(bias, act_funct=self.activation_function)

            return act_funct

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
	initializer = tf.contrib.layers.xavier_initializer()
	initial_value = initializer([filter_size, filter_size, in_channels, out_channels])
        #initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")
	
	initial_value = initializer([out_channels])
        #initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
	initializer = tf.contrib.layers.xavier_initializer()
        #initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
	initial_value = initializer([in_size, out_size])
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        #initial_value = tf.truncated_normal([out_size], .0, .001)
	initial_value = initializer([out_size])
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
