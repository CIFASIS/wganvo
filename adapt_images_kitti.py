import adapt_images
import os
import argparse
import numpy as np
from image import non_demosaic_load, savez_compressed
from transform import build_intrinsic_matrix
import matplotlib.pyplot as plt
from array_utils import list_to_array, save_txt, save_npy
from triangulate import triangulatePoints, matcher
from outliers import mask_outliers
# from transformations import euler_from_matrix, translation_from_matrix


_CAM2INDEX = {'cam0': 0, 'cam1': 1, 'cam2': 2, 'cam3': 3}
_CAM2FOLDER = {'cam0': 'image_0', 'cam1': 'image_1', 'cam2': 'image_2', 'cam3': 'image_3'}
DEFAULT_CALIBRATION_FILENAME = 'calib.txt'


def get_arguments():
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('dir', type=str, help='Directory containing iamge sequence')
    parser.add_argument('poses_file', type=str, help='File containing absolute poses')
    # parser.add_argument('calib_file', type=str, help='File containing calibration parameters')
    # parser.add_argument("--cam", help="One of the four cameras",
    #                     default="cam0", choices=["cam0", "cam1", "cam2", "cam3"])
    parser.add_argument('--crop', nargs=2, default=None, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help='(optional) If supplied, images will be cropped to WIDTH x HEIGHT. It is the new resolution after cropping.')
    parser.add_argument('--scale', nargs=2, default=None, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help='(optional) If supplied, images will be scaled to WIDTH x HEIGHT')
    parser.add_argument('--output_dir', type=str, default=None, help='(optional) Output directory')
    parser.add_argument('--mirror', action='store_true', help='Flip the images (axis x)')
    parser.add_argument('--offset', type=int, default=1, help='Take pair of frames every n frames')
    parser.add_argument('--reverse', action='store_true', help='Reverse')
    # parser.add_argument('image_name', type=str, help='Image name.')
    args = parser.parse_args()
    return args


def to_homogeneous(m):
    identity = np.asmatrix(np.identity(4))
    identity[0:3, :] = m
    return identity


def vector_to_homogeneous(v):
    m = np.asmatrix(v.reshape(3, 4))
    return to_homogeneous(m)


def get_intrinsics_parameters(focal_length, principal_point, resolution, crop=None, scale=None):
    if crop:  # focal_length remains the same. We must adjust the principal_point according to the new resolution
        # FIXME no estoy seguro de que esto este bien hecho
        principal_point = [principal_point[i] - (resolution[i] - crop[i]) / 2. for i in range(len(principal_point))]
        resolution = crop
    if scale:
        focal_length = [float(scale[i]) / resolution[i] * focal_length[i] for i in range(len(focal_length))]
        principal_point = [float(scale[i]) / resolution[i] * principal_point[i] for i in range(len(principal_point))]
    return focal_length, principal_point


def show(images):
    artist = plt.imshow(images[0], cmap='gray')
    for img in images:
        artist.set_data(img)
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.01)


def get_calibration_matrix(calibration, cam):
    idx = _CAM2INDEX[cam]
    calibration_matrix = calibration[idx, 1:]
    return np.asmatrix(calibration_matrix.reshape(3, 4))


def get_image_folder_name(cam):
    return _CAM2FOLDER[cam]


def calculate_transformation(pose_a, pose_b):
    return np.linalg.inv(pose_a) * pose_b


def get_src_dst_index(idx_pose, offset=1, reverse=None):
    dst_index = idx_pose + offset
    src_index = idx_pose
    if reverse:
        return dst_index, src_index
    return src_index, dst_index


def main():
    args = get_arguments()
    is_mirror = args.mirror
    print(args)
    # if args.cam != "cam0":
    #     raise Exception("Only cam0 is supported")
    output_dir = os.curdir
    if args.output_dir:
        output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        raise IOError(output_dir + "is not an existing folder")

    calib_file = os.path.join(args.dir, DEFAULT_CALIBRATION_FILENAME)
    with open(calib_file) as calibration_file, open(args.poses_file) as poses_file:
        calib = np.genfromtxt(calibration_file, delimiter=' ')
        left_calibration_matrix = get_calibration_matrix(calib, "cam0")
        right_calibration_matrix = get_calibration_matrix(calib, "cam1")
        poses = np.loadtxt(poses_file, delimiter=' ')

    # Left images
    left_image_folder_name = get_image_folder_name("cam0")
    left_image_dir = os.path.join(args.dir, left_image_folder_name)
    left_image_filenames = sorted(
        [item for item in os.listdir(left_image_dir) if os.path.isfile(os.path.join(left_image_dir, item))])

    # Right images
    right_image_folder_name = get_image_folder_name("cam1")
    right_image_dir = os.path.join(args.dir, right_image_folder_name)
    right_image_filenames = sorted(
        [item for item in os.listdir(right_image_dir) if os.path.isfile(os.path.join(right_image_dir, item))])

    images_list = []
    crop = args.crop
    scale = args.scale
    N = 150
    cloud_points = np.empty((len(left_image_filenames), 3, N))

    for (i, (left_img_name, right_img_name)) in enumerate(zip(left_image_filenames, right_image_filenames)):
        left_img = non_demosaic_load(os.path.join(left_image_dir, left_img_name))
        right_img = non_demosaic_load(os.path.join(right_image_dir, right_img_name))
        assert left_img_name == right_img_name

        # X.shape -> 3xN
        X = triangulate(left_img, right_img, left_calibration_matrix, right_calibration_matrix, N)
        cloud_points[i] = X

        original_resolution = adapt_images.get_resolution(left_img)
        assert isinstance(left_img, np.ndarray) and left_img.dtype == np.uint8 and left_img.flags.contiguous
        modified_img, _ = adapt_images.process_image(left_img, crop=crop, scale=scale)
        if is_mirror:
            modified_img = np.fliplr(modified_img)
        assert isinstance(modified_img,
                          np.ndarray) and modified_img.dtype == np.uint8  # and modified_img.flags.contiguous
        images_list.append(modified_img)
    save_npy(os.path.join(output_dir, 'points'), cloud_points)
    print(original_resolution)
    compressed_images = list_to_array(images_list)
    print compressed_images.shape

    t_records = []
    # p_records = []
    offset = args.offset
    reverse = args.reverse

    print(left_calibration_matrix.reshape(-1))
    # new_focal_length, new_principal_point = get_intrinsics_parameters(
    #    [left_calibration_matrix[0, 0], left_calibration_matrix[1, 1]], [left_calibration_matrix[0, 2], left_calibration_matrix[1, 2]],
    #    original_resolution, crop=crop, scale=scale)
    # new_intrinsic_matrix = build_intrinsic_matrix(new_focal_length, new_principal_point)
    assert len(left_image_filenames) == len(poses)
    mirror = np.asmatrix(np.diag((-1, 1, 1)))
    # In this case mirror = mirror^(-1)
    mirror_inverse = mirror
    transformations = []
    for idx_pose in xrange(poses.shape[0] - offset):
        # dst_index = idx_pose
        # src_index = idx_pose + offset
        src_index, dst_index = get_src_dst_index(idx_pose, offset, reverse)
        src_pose = poses[src_index]
        dst_pose = poses[dst_index]
        src = vector_to_homogeneous(src_pose)
        dst = vector_to_homogeneous(dst_pose)
        transf_src_dst = calculate_transformation(src, dst)
        if is_mirror:
            transf_src_dst[0:3, 0:3] = mirror_inverse * transf_src_dst[0:3, 0:3] * mirror
        t_matrix = transf_src_dst[0:3, :]
        # FIXME no hay que solo premultiplicarlo por la intrinsic_matrix, el calculo es otro, ver la documentacion en el Drive
        # p_matrix = new_intrinsic_matrix * t_matrix
        t_records.append((t_matrix, src_index, dst_index))
        # p_records.append((p_matrix, src_index, dst_index))
        transformations.append(np.asarray(t_matrix).reshape(-1))

    transf = np.array(t_records, dtype=[('T', ('float32', (3, 4))), ('src_idx', 'int32'), ('dst_idx', 'int32')])
    # proy = np.array(p_records, dtype=[('P', ('float32', (3, 4))), ('src_idx', 'int32'), ('dst_idx', 'int32')])
    savez_compressed(os.path.join(output_dir, 't'), transf)
    # savez_compressed(os.path.join(output_dir, 'p'), proy)
    # save(os.path.join(output_dir, "intrinsic_matrix"), new_intrinsic_matrix, fmt='%.18e')
    # save(os.path.join(output_dir, "intrinsic_parameters"), [new_focal_length, new_principal_point], fmt='%.18e')
    save_txt(os.path.join(output_dir, 'images_shape'), compressed_images.shape, fmt='%i')
    compressed_images_path = os.path.join(output_dir, 'images')
    savez_compressed(compressed_images_path, compressed_images)
    ts = list_to_array(transformations)
    # print(euler_from_matrix(transf_src_dst))
    # print(translation_from_matrix(transf_src_dst))
    save_txt(os.path.join(output_dir, 'transformations'), ts, fmt='%.18e')

# Triangulate points
# Input:
#       left_img: image
#       right_img: image
#       P1: projection matrix 3x4 (left)
#       P2: projection matrix 3x4 (right)
#       N: select N points
# Output:
#       X: array of 3D points (3xN)
def triangulate(left_img, right_img, P1, P2, N):
    pts_l, pts_r = matcher(left_img, right_img)

    # X's points are in camera coordinates
    X = triangulatePoints(P1, P2, pts_l, pts_r)
    dim = X.shape[0]
    # X.shape -> (4xN)
    for i in range(3):
        # Each axis (X,Y,Z) is filtered by mask_outliers
        mask = mask_outliers(X[i], 1000)
        X = X[:, mask]
        X = X.reshape((dim, -1))

    # Randomly select N points
    replace = X.shape[1] <= N
    if replace:
        print(X.shape[1])
    random_selection = np.random.choice(X.shape[1], N, replace=replace)
    X = X[:3, random_selection]
    return X


if __name__ == "__main__":
    main()
