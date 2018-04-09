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
    images,_,_ = read_data_sets(FLAGS.img_file)
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


