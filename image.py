################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
###############################################################################

import re
import numpy as np
from scipy.misc import imresize, imsave
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic

BAYER_STEREO = 'gbrg'
BAYER_MONO = 'rggb'


def load_image(image_path, model=None):
    """Loads and rectifies an image from file.

    Args:
        image_path (str): path to an image from the dataset.
        model (camera_model.CameraModel): if supplied, model will be used to undistort image.

    Returns:
        numpy.ndarray: demosaiced and optionally undistorted image

    """
    if model:
        camera = model.camera
    else:
        camera = re.search('(stereo|mono_(left|right|rear))', image_path).group(0)
    if camera == 'stereo':
        pattern = BAYER_STEREO
    else:
        pattern = BAYER_MONO

    img = Image.open(image_path)    
    img = demosaic(img, pattern)    
    if model:
        img = model.undistort(img)
    
    img = rgb_2_grey(img)
    return img

def crop_image(num_array, cropx, cropy):
    y = num_array.shape[0]
    x = num_array.shape[1]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return num_array[starty:starty + cropy, startx:startx+cropx]    

def scale_image(num_array, sizex, sizey):
    return imresize(num_array, (sizey,sizex))

def save_image(num_array, path):
    imsave(path, num_array)

def rgb_2_grey(img):
    return np.dot(img[...,:3],[0.299, 0.587, 0.114]).astype(img.dtype)