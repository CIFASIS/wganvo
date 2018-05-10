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
current = current[0:3,:]
num_examples = len(data)
ts = np.empty((num_examples,12))
i = 0
for t in data:
	T = np.matrix(np.identity(4))
	T[0:3,:] = t.reshape(3,4)
	if args.mode == 'inv':
		transformation = np.linalg.inv(T)
	else:
		transformation = T
	current = current * transformation
	ts[i] = current.reshape(12)
	i += 1
np.savetxt('abs.poses.txt', ts, delimiter=' ')




