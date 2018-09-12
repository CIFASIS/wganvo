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
    # TODO adaptar a stereo
    assert grid.ndim == 3 and grid.shape[2] == 2

    grid = grid.transpose(2, 0, 1)
    imsave_pair(grid[0, ...], grid[1, ...], save_path, iteration, prefix)


# Guarda imgs individuales
def save_pair_images(X, save_path, iteration, prefix='samples'):
    # BCHW
    assert X.ndim == 4 and X.shape[1] == 4

    idx = 0  # randrange(X.shape[0])
    samples = X[idx]
    imsave_pair(samples[0, ...], samples[1, ...], save_path, iteration, prefix + '_l')
    imsave_pair(samples[2, ...], samples[3, ...], save_path, iteration, prefix + '_r')


def imsave_pair(im1, im2, save_path, iteration, prefix):
    img_name = prefix + '_{}_{}.png'
    imsave(os.path.join(save_path, img_name.format(iteration, 0)), im1)
    imsave(os.path.join(save_path, img_name.format(iteration, 1)), im2)
