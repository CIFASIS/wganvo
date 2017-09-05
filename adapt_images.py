import argparse
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime as dt
from image import load_image, crop_image, scale_image, save_image
from camera_model import CameraModel

parser = argparse.ArgumentParser(description='Play back images from a given directory')

parser.add_argument('dir', type=str, help='Directory containing images.')
parser.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
parser.add_argument('--crop', nargs = 2, default=None, type=int, metavar=('WIDTH', 'HEIGHT'), help='(optional) If supplied, images will be cropped to WIDTH x HEIGHT')
parser.add_argument('--scale', nargs = 2, default=None, type=int, metavar=('WIDTH', 'HEIGHT'), help='(optional) If supplied, images will be scaled to WIDTH x HEIGHT')
#parser.add_argument('image_name', type=str, help='Image name.')
args = parser.parse_args()

camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0)

timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps'))
if not os.path.isfile(timestamps_path):
  timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps')
  if not os.path.isfile(timestamps_path):
      raise IOError("Could not find timestamps file")

model = None
if args.models_dir:
    model = CameraModel(args.models_dir, args.dir)

current_chunk = 0
timestamps_file = open(timestamps_path)

#im = args.image_name
if args.scale:
    scalex = args.scale[0]
    scaley = args.scale[1]
#dataset = np.array([])
for line in timestamps_file:
    tokens = line.split()
    #datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
    chunk = int(tokens[1])
    image_name = tokens[0]
    filename = os.path.join(args.dir, image_name + '.png')
    if not os.path.isfile(filename):
        if chunk != current_chunk:
            print("Chunk " + str(chunk) + " not found")
            current_chunk = chunk
        raise IOError("Rt exc")

    current_chunk = chunk

    img = load_image(filename, model)
    if args.crop:
        img = crop_image(img, args.crop[0], args.crop[1])
    if args.scale:
        img = scale_image(img, scalex, scaley)        
    #save_image(img, 'scaled_img.jpg')
    print "loading..." + image_name


