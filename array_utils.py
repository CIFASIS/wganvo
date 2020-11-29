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

import numpy as np

def save_txt(name, array, fmt='%1.6f'):
    np.savetxt(name + '.txt', array, delimiter=' ', fmt=fmt)

def load(name):
    return np.loadtxt(name, delimiter=' ')
    
def save_as_list(name, array, fmt='%1.6f'):
    np.savetxt(name + '.txt', array.ravel(), delimiter=' ', fmt=fmt)
    
def list_to_array(list):
    return np.array(list)

def save_npy(name, arr):
    np.save(name, arr)

def load_npy(name):
    return np.load(name)