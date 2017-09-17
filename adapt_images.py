import argparse
import os
import re
#from datetime import datetime as dt
from image import load_image, crop_image, scale_image, savez_compressed
from camera_model import CameraModel
from transform import build_se3_transform, build_intrinsic_matrix
from array_utils import save_as_list, list_to_array
parser = argparse.ArgumentParser(description='Play back images from a given directory')

parser.add_argument('dir', type=str, help='Directory containing images.')
parser.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
parser.add_argument('--crop', nargs = 2, default=None, type=int, metavar=('WIDTH', 'HEIGHT'), help='(optional) If supplied, images will be cropped to WIDTH x HEIGHT')
parser.add_argument('--scale', nargs = 2, default=None, type=int, metavar=('WIDTH', 'HEIGHT'), help='(optional) If supplied, images will be scaled to WIDTH x HEIGHT')
parser.add_argument('--extrinsics_dir', type=str, help='(optional) Directory containing sensor extrinsics')
parser.add_argument('--output_dir', type=str, default = None, help='(optional) Output directory')
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

output_dir = os.curdir
if args.output_dir:
        output_dir = args.output_dir
if not os.path.isdir(output_dir):
    raise IOError(output_dir + "is not an existing folder")

#im = args.image_name
if args.scale:
    scalex = args.scale[0]
    scaley = args.scale[1]
#dataset = np.array([])
result_list = []
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
        continue

    current_chunk = chunk

    img = load_image(filename, model)
    resolution = [img.shape[1], img.shape[0]]
    if args.crop:
        img = crop_image(img, args.crop[0], args.crop[1])
    if args.scale:
        img = scale_image(img, scalex, scaley)
    result_list.append(list(img))    

path = os.path.normpath(args.dir)
folders = path.split(os.sep)
result = list_to_array(result_list)
print result.shape
compressed_file_path = os.path.join(output_dir, folders[-3])
savez_compressed(compressed_file_path, result)
focal_length = model.get_focal_length()
principal_point = model.get_principal_point()
if args.crop:
    resolution = args.crop
    principal_point = [x / 2. for x in args.crop]
if args.scale:
    focal_length = [float(args.scale[i])/resolution[i] * focal_length[i] for i in range(len(focal_length))]
    principal_point = [x / 2. for x in args.scale]

if(args.extrinsics_dir):
    extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
        extrinsic_matrix = build_se3_transform(extrinsics)
        intrinsic_matrix = build_intrinsic_matrix(focal_length, principal_point)        
        p_matrix = intrinsic_matrix * extrinsic_matrix[0:3]
        p_matrix_file_path = os.path.join(output_dir, 'p_matrix')
        save_as_list(p_matrix_file_path, p_matrix)
        
        
