#
# This file is part of wganvo.
# This file is based on a file from https://github.com/igul222/improved_wgan_training (see original license below).
#
# Modifications copyright (C) 2019 Javier Cremona (CIFASIS-CONICET)
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

"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import os
from scipy.misc import imsave


def save_images(X, save_path):
    img = build_grid(X)
    imsave(save_path, img)


def build_grid(X):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99 * X).astype('uint8')
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, n_samples / rows
    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))
    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, X.shape[3]))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))
    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x
    return img


# Guarda dos grillas
def save_pair_images_grid(X, save_path, iteration, prefix='samples'):
    grid = build_grid(X)
    assert grid.ndim == 3 and grid.shape[2] == 2

    grid = grid.transpose(2, 0, 1)
    imsave_pair(grid, save_path, iteration, prefix)


# Guarda imgs individuales
def save_pair_images(X, save_path, iteration, prefix='samples'):
    # BCHW
    assert X.ndim == 4 and X.shape[1] == 2

    idx = 0 #randrange(X.shape[0])
    pair = X[idx]
    imsave_pair(pair, save_path, iteration, prefix)


def imsave_pair(pair, save_path, iteration, prefix):
    img_name = prefix + '_{}_{}.png'
    imsave(os.path.join(save_path, img_name.format(iteration, 0)), pair[0, ...])
    imsave(os.path.join(save_path, img_name.format(iteration, 1)), pair[1, ...])
