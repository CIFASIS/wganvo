import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import argparse
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from input_data import read_data_sets, DataSet

def show(imagesl, imagesr):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    artistl = ax1.imshow(imagesl[0], cmap='gray')
    ax2 = fig.add_subplot(122)
    artistr = ax2.imshow(imagesr[0], cmap='gray')
    for imgl, imgr in zip(imagesl, imagesr):
        artistl.set_data(imgl)
        artistr.set_data(imgl)
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.01)

def main():
    images,_,_,_ = read_data_sets(FLAGS.img_file)
    print(images.shape)
    show(images[...,0], images[...,2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'img_file',
        type=str,
        help='Images file'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()


