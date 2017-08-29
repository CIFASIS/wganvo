import argparse
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime as dt
from image import load_image
from camera_model import CameraModel

parser = argparse.ArgumentParser(description='Play back images from a given directory')

parser.add_argument('dir', type=str, help='Directory containing images.')
parser.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
parser.add_argument('--scale', type=float, default=1.0, help='(optional) factor by which to scale images before display')
parser.add_argument('image_name', type=str, help='Image name.')
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

im = args.image_name

tokens = im, 1
datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
chunk = int(tokens[1])

filename = os.path.join(args.dir, tokens[0] + '.png')
if not os.path.isfile(filename):
    if chunk != current_chunk:
        print("Chunk " + str(chunk) + " not found")
        current_chunk = chunk
    raise IOError("Rt exc")

current_chunk = chunk

img = load_image(filename, model)
plt.imshow(img)
plt.xlabel(datetime)
plt.xticks([])
plt.yticks([])
plt.pause(100.20)

