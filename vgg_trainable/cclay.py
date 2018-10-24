import tensorflow as tf

import numpy as np
from functools import reduce
import math

# VGG_MEAN = [103.939, 116.779, 123.68]
slim = tf.contrib.slim

class CrossConvolutionalNet:

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
        :param images: [batch, height, width, channels] (usually a placeholder)
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        self.conv1_1 = self.conv_layer(images, 2, 96, "conv1_1",filter_size=5,stride=1)
        self.conv1_2 = self.conv_layer(self.conv1_1, 96, 96, "conv1_2",filter_size=5,stride=1)
        self.pool1 = self.pooling(self.conv1_2, 'pool1', pooling_type=pooling_type)

        self.conv2_1 = self.conv_layer(self.pool1, 96, 128, "conv2_1",filter_size=5,stride=1)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2",filter_size=5,stride=1)
        self.pool2 = self.pooling(self.conv2_2, 'pool2', pooling_type=pooling_type)

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1",filter_size=5,stride=1)
        self.pool3 = self.pooling(self.conv3_1, 'pool3', pooling_type=pooling_type)
        self.conv3_2 = self.conv_layer(self.pool3, 256, 256, "conv3_2",filter_size=5,stride=1)
        self.pool3_1 = self.pooling(self.conv3_2, 'pool3_1', pooling_type=pooling_type)
        # 256 * 6 * 8
        batch_size = int(images.shape[0])
        fc_in_size = ((self.width // (2 ** 4)) * (self.height // (2 ** 4))) * 256
        self.fc1 = self.fc_layer(self.pool3_1, fc_in_size, 1600, "fc1")
        z = tf.reshape(self.fc1, shape=[batch_size, -1])
        self.z_mean, self.z_stddev_log = tf.split(
            axis=1, num_or_size_splits=2, value=z)

        self.z_stddev = tf.exp(self.z_stddev_log)
        epsilon = tf.random_normal(
            self.z_mean.get_shape().as_list(), 0, 1, dtype=tf.float32)
        kernel = self.z_mean + tf.multiply(self.z_stddev, epsilon)
        width = int(math.sqrt(kernel.get_shape().as_list()[1] // 32))
        kernel = tf.reshape(kernel, [batch_size, width, width, 32])

        kernel = self.conv_layer(kernel, 32, 32, "convk_1", filter_size=5, stride=1)
        #kernel = self.conv_layer(kernel, 128, 128, "convk_2", filter_size=5, stride=1)

        ##############################################################################

        self.conv4_1 = self.conv_layer(images[...,0:1], 1, 64, "conv4_1",filter_size=5,stride=1)
        #self.conv4_2 = self.conv_layer(self.conv4_1, 64, 64, "conv4_2",filter_size=5,stride=1)
        self.pool4 = self.pooling(self.conv4_1, 'pool4', pooling_type=pooling_type, ksize=5)

        self.conv5_1 = self.conv_layer(self.pool4, 64, 32, "conv5_1", filter_size=5, stride=1)
        #self.conv5_2 = self.conv_layer(self.conv5_1, 64, 32, "conv5_2", filter_size=5, stride=1)
        encoded_image = self.pooling(self.conv5_1, 'pool5', pooling_type=pooling_type, ksize=2)
        ##############################################################################

        #kernels = tf.split(axis=3, num_or_size_splits=4, value=kernel)
        #kernel = kernels[0]
        kernel = tf.unstack(kernel, axis=0) # LISTA de kernels, esta lista tiene longitud batch_size
        encoded_image = tf.unstack(encoded_image, axis=0)
        assert len(encoded_image) == len(kernel)

        conved_image = []
        for j in xrange(len(encoded_image)):
            conved_image.append(self.crossConvHelper(
                encoded_image[j], kernel[j]))
        cross_conved_image = tf.concat(axis=0, values=conved_image)

        ###############################################################################

        dec = self.deconv(cross_conved_image, 64, kernel_size=3, stride=2)
        fc_in_size = int(dec.shape[1]) * int(dec.shape[2]) * int(dec.shape[3])
        self.fc6 = self.fc_layer(dec, fc_in_size, 512, "fc6")
        self.relu6 = self.activation_function_tensor(self.fc6,
                                                     act_function=self.activation_function)  # tf.nn.relu(self.fc6)
        if train_mode is not None:
            print("Train Mode placeholder")
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            print("Not Train Mode placeholder")
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc8 = self.fc_layer(self.relu6, 512, 7, "fc8")
        quaternions = self.fc8[:, 3:7]
        quaternions_norm = tf.norm(quaternions, axis=1)
        unit_quaternions = quaternions / tf.reshape(quaternions_norm, (-1, 1))
        self.fc8 = tf.concat([self.fc8[:, :3], unit_quaternions], 1)



        return self.fc8, self.z_mean, self.z_stddev, self.z_stddev_log

    def trainOp(self, loss, learning_rate):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        lrn_rate = tf.maximum(
            0.01,  # min_lr_rate.
            tf.train.exponential_decay(
                learning_rate, global_step, 10000, 0.5))
        tf.summary.scalar('learning rate', lrn_rate)
        optimizer = tf.train.AdamOptimizer(lrn_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def buildLoss(self, output, targets):
        # 1. reconstr_loss seems doesn't do better than l2 loss.
        # 2. Only works when using reduce_mean. reduce_sum doesn't work.
        # 3. It seems kl loss doesn't play an important role.
        self.loss = 0
        with tf.variable_scope('loss'):
            #l2_loss = tf.reduce_mean(tf.square(output - targets))
            #tf.summary.scalar('l2_loss', l2_loss)
            #self.loss += l2_loss
            # if self.params['reconstr_loss']:
            #     reconstr_loss = (-tf.reduce_mean(
            #         self.diffs[1] * (1e-10 + self.diff_output) +
            #         (1 - self.diffs[1]) * tf.log(1e-10 + 1 - self.diff_output)))
            #     reconstr_loss = tf.check_numerics(reconstr_loss, 'reconstr_loss')
            #     tf.summary.scalar('reconstr_loss', reconstr_loss)
            #     self.loss += reconstr_loss

            kl_loss = (0.5 * tf.reduce_mean(
                tf.square(self.z_mean) + tf.square(self.z_stddev) -
                2 * self.z_stddev_log - 1))
            tf.summary.scalar('kl_loss', kl_loss)
            #self.loss += kl_loss

            tf.summary.scalar('cnnloss', self.loss)
            return self.loss


    def crossConvHelper(self, encoded_image, kernel):
        """Cross Convolution.
          The encoded image and kernel are of the same shape. Namely
          [batch_size, image_size, image_size, channels]. They are split
          into [image_size, image_size] image squares [kernel_size, kernel_size]
          kernel squares. kernel squares are used to convolute image squares.
        """
        images = tf.expand_dims(encoded_image, 0)
        kernels = tf.expand_dims(kernel, 3)
        return tf.nn.depthwise_conv2d(images, kernels, [1, 1, 1, 1], 'SAME')

    def deconv(self, net, out_filters, kernel_size, stride):
        shape = net.get_shape().as_list()
        in_filters = shape[3]
        kernel_shape = [kernel_size, kernel_size, out_filters, in_filters]

        weights = tf.get_variable(
            name='weights',
            shape=kernel_shape,
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.01))

        out_height = shape[1] * stride
        out_width = shape[2] * stride
        batch_size = shape[0]

        output_shape = [batch_size, out_height, out_width, out_filters]
        net = tf.nn.conv2d_transpose(net, weights, output_shape,
                                     [1, stride, stride, 1], padding='SAME')
        return net


    def pooling(self, bottom, name, ksize=2, pooling_type="max"):
        if pooling_type == "avg":
            return self.avg_pool(bottom, name)
        return self.max_pool(bottom, ksize, name)

    def avg_pool(self, bottom, name):
        print("Using avg pool")
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, ksize, name):
        print("Using max pool")
        return tf.nn.max_pool(bottom, ksize=[1, ksize, ksize, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def activation_function_tensor(self, features, act_function="relu"):
        if act_function == "leaky_relu":
            print("Using leaky relu")
            return tf.nn.leaky_relu(features)
        print("Using relu")
        return tf.nn.relu(features)

    def conv_layer(self, bottom, in_channels, out_channels, name, filter_size=3, stride=1, padding='SAME',normalizer_fn=slim.batch_norm):
        with tf.variable_scope(name):
            #filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)

            #conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
            #bias = tf.nn.bias_add(conv, conv_biases)
            slim.conv2d(bottom,out_channels, [filter_size,filter_size], normalizer_fn=normalizer_fn, stride=stride, padding=padding)
            act_funct = self.activation_function_tensor(bias, act_function=self.activation_function)

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
        # initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = initializer([out_channels])
        # initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initializer = tf.contrib.layers.xavier_initializer()
        # initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        initial_value = initializer([in_size, out_size])
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        # initial_value = tf.truncated_normal([out_size], .0, .001)
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
