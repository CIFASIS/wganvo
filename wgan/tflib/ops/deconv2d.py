#
# This file is part of wganvo.
# This file is based on a file from https://github.com/igul222/improved_wgan_training (see original license below).
#
# Copyright (C) 2019 Javier Cremona (CIFASIS-CONICET)
# For more information see <https://github.com/CIFASIS/wganvo>
#
# wganvo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wganvo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with wganvo. If not, see <http://www.gnu.org/licenses/>.
#

# MIT License
#
# Copyright (c) 2017 Ishaan Gulrajani
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tflib as lib

import numpy as np
import tensorflow as tf

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Deconv2D(
    name, 
    input_dim, 
    output_dim, 
    filter_size, 
    inputs, 
    he_init=True,
    weightnorm=None,
    biases=True,
    gain=1.,
    mask_type=None,
    ):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.name_scope(name) as scope:

        if mask_type != None:
            raise Exception('Unsupported configuration')

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        stride = 2
        fan_in = input_dim * filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size**2

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))


        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )

        filter_values *= gain

        filters = lib.param(
            name+'.Filters',
            filter_values
        )

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,3)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,3]))
                filters = filters * tf.expand_dims(target_norms / norms, 1)


        inputs = tf.transpose(inputs, [0,2,3,1], name='NCHW_to_NHWC')

        input_shape = tf.shape(inputs)
        try: # tf pre-1.0 (top) vs 1.0 (bottom)
            output_shape = tf.pack([input_shape[0], 2*input_shape[1], 2*input_shape[2], output_dim])
        except Exception as e:
            output_shape = tf.stack([input_shape[0], 2*input_shape[1], 2*input_shape[2], output_dim])

        result = tf.nn.conv2d_transpose(
            value=inputs, 
            filter=filters,
            output_shape=output_shape, 
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )
            result = tf.nn.bias_add(result, _biases)

        result = tf.transpose(result, [0,3,1,2], name='NHWC_to_NCHW')


        return result
