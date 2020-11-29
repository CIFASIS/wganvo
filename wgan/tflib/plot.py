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

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import collections
import time
import cPickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]


def tick():
    _iter[0] += 1

def get_global_iter():
    return _iter[0]


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def flush(log_dir):
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean(vals.values())))
        _since_beginning[name].update(vals)

        x_vals = np.sort(_since_beginning[name].keys())
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        file_name = name.replace(' ', '_') + '.jpg'
        plt.savefig(os.path.join(log_dir, file_name))

    print "iter {}\t{}".format(_iter[0], "\t".join(prints))
    _since_last_flush.clear()

    with open(os.path.join(log_dir, 'log.pkl'), 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
