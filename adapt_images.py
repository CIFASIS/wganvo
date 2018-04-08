import argparse
import os
import re
import csv
#from datetime import datetime as dt
from image import load_image, crop_image, scale_image, savez_compressed
from camera_model import CameraModel
from transform import build_se3_transform, build_intrinsic_matrix
from array_utils import save_as_list, list_to_array, save
import numpy as np


def process_image(img, crop=None, scale=None):
    return resize_image(img, crop, scale)


def resize_image(img, crop=None, scale=None):
    resolution = get_resolution(img)
    if crop:
        img = crop_image(img, crop[0], crop[1])
    if scale:
        img = scale_image(img, scale[0], scale[1])
    return img, resolution


def get_resolution(img):
    resolution = [img.shape[1], img.shape[0]]
    return resolution


def get_intrinsics_parameters(focal_length, principal_point, resolution, crop=None, scale=None):
    if crop:
        resolution = crop
        principal_point = [x / 2. for x in crop]  ## FIXME ver este mismo metodo en adapt_images_kitti
    if scale:
        focal_length = [float(scale[i])/resolution[i] * focal_length[i] for i in range(len(focal_length))]
        principal_point = [x / 2. for x in scale]
    return focal_length, principal_point


def main():
    args = get_arguments()

    camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0)

    timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps'))
    if not os.path.isfile(timestamps_path):
      timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps')
      if not os.path.isfile(timestamps_path):
          raise IOError("Could not find timestamps file")

    model = None
    if args.models_dir:
        model = CameraModel(args.models_dir, args.dir)

    output_dir = os.curdir
    if args.output_dir:
            output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        raise IOError(output_dir + "is not an existing folder")

    result_list = []
    count = 0
    dictionary = {}
    t_records = []
    p_records = []
    angles_records = []
    intrinsic_matrix = None

    with open(args.poses_file) as vo_file:
            vo_reader = csv.reader(vo_file)
            headers = next(vo_file)
            for row in vo_reader:
                src_image_name = row[0]
                dst_image_name = row[1]
                src_image_filename = os.path.join(args.dir, src_image_name + '.png')
                dst_image_filename = os.path.join(args.dir, dst_image_name + '.png')
                if not os.path.isfile(src_image_filename) or not os.path.isfile(dst_image_filename):
                    continue
                if dst_image_name not in dictionary:
                    img, orig_resolution = process_image(load_image(dst_image_filename, model), args.crop, args.scale)
                    dictionary[dst_image_name] = count
                    count = count + 1
                    result_list.append(list(img))
                if src_image_name not in dictionary:
                    img, orig_resolution = process_image(load_image(src_image_filename, model), args.crop, args.scale)
                    dictionary[src_image_name] = count
                    count = count + 1
                    result_list.append(list(img))

                focal_length, principal_point = get_intrinsics_parameters(model.get_focal_length(), model.get_principal_point(), orig_resolution, args.crop, args.scale)
                src_image_idx = dictionary[src_image_name]
                dst_image_idx = dictionary[dst_image_name]
                xyzrpy = [float(v) for v in row[2:8]]
                rel_pose = build_se3_transform(xyzrpy)
                t_matrix = rel_pose[0:3] # 3x4 matrix
                intrinsic_matrix = build_intrinsic_matrix(focal_length, principal_point)
                p_matrix = intrinsic_matrix * t_matrix
                t_records.append((t_matrix, src_image_idx, dst_image_idx))
                p_records.append((p_matrix, src_image_idx, dst_image_idx))
                angles_records.append((xyzrpy, src_image_idx, dst_image_idx))

    transf = np.array(t_records, dtype=[('T',('float64',(3,4))),('src_idx', 'int32'),('dst_idx', 'int32')])
    proy = np.array(p_records, dtype=[('P',('float64',(3,4))),('src_idx', 'int32'),('dst_idx', 'int32')])
    angles = np.array(angles_records, dtype=[('ang',('float64',6)),('src_idx', 'int32'),('dst_idx', 'int32')])
    # Solo lo guardo una vez porque es constante para todo el dataset (o deberia serlo)
    if intrinsic_matrix is not None:
        save(os.path.join(output_dir,"intrinsic_matrix"), intrinsic_matrix)
        save(os.path.join(output_dir,"intrinsic_parameters"), [focal_length, principal_point])
    #path = os.path.normpath(args.dir)
    #folders = path.split(os.sep)
    #compressed_file_path = os.path.join(output_dir, folders[-3])
    result = list_to_array(result_list)
    save(os.path.join(output_dir, 'images_shape'), result.shape, fmt='%i')
    print result.shape
    compressed_file_path = os.path.join(output_dir, 'images')
    savez_compressed(compressed_file_path, result)
    savez_compressed(os.path.join(output_dir, 't'),transf)
    savez_compressed(os.path.join(output_dir, 'p'),proy)
    savez_compressed(os.path.join(output_dir, 'angles'),angles)


def get_arguments():
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('dir', type=str, help='Directory containing images.')
    parser.add_argument('poses_file', type=str, help='File containing VO poses')
    parser.add_argument('--models_dir', type=str, default=None,
                        help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
    parser.add_argument('--crop', nargs=2, default=None, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help='(optional) If supplied, images will be cropped to WIDTH x HEIGHT')
    parser.add_argument('--scale', nargs=2, default=None, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help='(optional) If supplied, images will be scaled to WIDTH x HEIGHT')
    parser.add_argument('--output_dir', type=str, default=None, help='(optional) Output directory')
    # parser.add_argument('image_name', type=str, help='Image name.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()