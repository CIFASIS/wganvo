import tensorflow as tf
import numpy as np
from scipy import linalg
import argparse
import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from input_data import read_data_sets, DataSet
from eval_utils import infer_relative_poses, get_absolute_poses, plot_frames_vs_abs_distance

# import transformations

DEFAULT_INTRINSIC_FILE_NAME = "intrinsic_matrix.txt"


def test_model(model_name, data_dir, output_dir, batch_size):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_name + ".meta")
    # print(model_name)
    # inverse_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)
    saver.restore(sess, model_name)  # tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    outputs = graph.get_tensor_by_name("outputs:0")
    targets_placeholder = graph.get_tensor_by_name("targets_placeholder:0")
    images_placeholder = graph.get_tensor_by_name("images_placeholder:0")
    train_mode = graph.get_tensor_by_name("train_mode:0")  # FIXME Podria arrojar exception
    images, targets, _, groups, _ = read_data_sets(data_dir)
    dataset = DataSet(images, targets, groups, fake_data=False)
    relative_poses_prediction, relative_poses_target = infer_relative_poses(sess, dataset, batch_size,
                                                                            images_placeholder,
                                                                            outputs,
                                                                            targets_placeholder, train_mode)
    frames, abs_distance = plot_frames_vs_abs_distance(relative_poses_prediction, relative_poses_target, dataset,
                                                       output_dir, save_txt=True, plot=True)
    points = np.array(zip(frames, abs_distance))
    np.savetxt(os.path.join(output_dir, "frames_vs_abs_distance.txt"), points)
    np.savetxt(os.path.join(output_dir, "relative_poses_prediction.txt"), relative_poses_prediction.reshape(-1, 12),
               delimiter=' ')
    np.savetxt(os.path.join(output_dir, "relative_poses_target.txt"), relative_poses_target.reshape(-1, 12),
               delimiter=' ')
    absolute_poses_prediction = get_absolute_poses(relative_poses_prediction)
    absolute_poses_target = get_absolute_poses(relative_poses_target)
    np.savetxt(os.path.join(output_dir, "absolute_poses_prediction.txt"),
               absolute_poses_prediction.reshape(-1, 12), delimiter=' ')
    np.savetxt(os.path.join(output_dir, "absolute_poses_target.txt"), absolute_poses_target.reshape(-1, 12),
               delimiter=' ')


def main(_):
    # intrinsic_matrix = np.matrix(np.loadtxt(FLAGS.intrinsics_path, delimiter=' '))
    test_model(FLAGS.model_name, FLAGS.data_dir, FLAGS.output_dir, FLAGS.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_name',
        type=str,
        help='Model name'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory to put the data'
    )
    # parser.add_argument(
    #     '--intrinsics_path',
    #     type=str,
    #     default=os.path.join(os.getcwd(), DEFAULT_INTRINSIC_FILE_NAME),
    #     help='Intrinsic matrix path'
    # )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.getcwd(),
        help='Output dir'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
