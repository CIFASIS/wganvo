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

def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if axes != [0,2,3]:
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
    shape = mean.get_shape().as_list() # shape is [1,n,1,1]
    offset_m = lib.param(name+'.offset', np.zeros([n_labels,shape[1]], dtype='float32'))
    scale_m = lib.param(name+'.scale', np.ones([n_labels,shape[1]], dtype='float32'))
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(inputs, mean, var, offset[:,:,None,None], scale[:,:,None,None], 1e-5)
    return result