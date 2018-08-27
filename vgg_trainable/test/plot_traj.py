from mpl_toolkits.mplot3d import axes3d
import numpy as np
import transformations
import matplotlib.pyplot as plt

def read(filename, delimiter=','):
	return np.genfromtxt(filename, delimiter=delimiter)


def plot(array):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d') # 111 means "1x1 grid, first subplot"
	p = ax.plot(array[:,0], array[:,1], array[:,2], label='target')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.legend()
	plt.show()
	

def main():
	data = read('vo.csv')
	data = data[1:len(data),2:8]

	current = np.array([0.,0.,0.])#.transpose()
	#current = np.matrix(np.identity(4))
	num_examples = len(data)
	ts = np.empty((num_examples,3))
	poses = np.empty((num_examples,12))
	i = 0
	for t in data:
		# Devuelve una matriz 4x4
		# t[3] = roll, t[4] = pitch, t[5] = yaw
		T = transformations.euler_matrix(t[3],t[4],t[5], 'sxyz')
		T[0:3,3] = t[0:3]
		current =  t[0:3] + current#np.linalg.inv(T) *current   #np.linalg.inv(T) * current
		ts[i] = current#[0:3,3].transpose()
		#poses[i] = current[0:3,:].reshape(12)		
		i += 1

	np.savetxt("poses.txt", poses, delimiter=" ")
	plot(ts)

if __name__ == "__main__":
	main()
