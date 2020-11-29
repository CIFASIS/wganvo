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

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
gparentdir = os.path.dirname(parentdir)
sys.path.insert(0, gparentdir)
import eval_utils
from plot_traj import read, plot
import argparse
import numpy as np

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('poses', type=str,
                    help='Poses file')
parser.add_argument("--mode", help="inv = invert the transformation",
                    default='ninv', choices=["inv", "ninv"])

args = parser.parse_args()

data = read(args.poses, delimiter=' ')

current = np.matrix(np.identity(4))
current = current[0:3, :]
num_examples = len(data)
# ts = np.empty((num_examples,12))
i = 0
inv = args.mode == 'inv'
ts = eval_utils.get_absolute_poses(data.reshape(-1, 3, 4), inv=inv)
ts = ts.reshape(-1, 12)
# for t in data:
# 	T = np.matrix(np.identity(4))
# 	T[0:3,:] = t.reshape(3,4)
# 	if args.mode == 'inv':
# 		transformation = np.linalg.inv(T)
# 	else:
# 		transformation = T
# 	current = current * transformation
# 	ts[i] = current.reshape(12)
# 	i += 1
np.savetxt('abs.poses.txt', ts, delimiter=' ')
