#
# This file is part of wganvo.
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

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import argparse
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from input_data import read_data_sets, DataSet

def show(images):
    artist = plt.imshow(images[0], cmap='gray')
    for img in images:
        artist.set_data(img)
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.01)

def main():
    images,_,_,_,_ = read_data_sets(FLAGS.img_file)
    show(images[...,1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'img_file',
        type=str,
        help='Images file'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()


