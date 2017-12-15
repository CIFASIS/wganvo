import numpy

import collections
import os
from six.moves import xrange

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 128
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
LABELS_SIZE = 12
DEFAULT_MAIN_KEY = 'arr_0'
P_FILENAME = "p.npz"
DEFAULT_LABEL_KEY = "P"

class DataSet(object):
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype="float32",
               reshape=False,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    #seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    #numpy.random.seed(seed1 if seed is None else seed2)
    #dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in ("uint8", "float32"):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]
      print images.shape
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
#      if dtype == "float32":
        # Convert from [0, 255] -> [0.0, 1.0].
#        images = images.astype(numpy.float32, copy=False)
#        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]



def read_data_sets(data_dir, fake_data):
    listdir = [os.path.join(data_dir, item) for item in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, item)) and "centre_" in item]
    train_images, train_labels = _get_images_and_labels(listdir)

    validation_images, validation_labels = _inputs("/home/jcremona/tesina/workspace.back1/2014-05-06-12-54-54_stereo_centre_01")
    test_images, test_labels = _inputs("/home/jcremona/tesina/workspace.back1/2014-05-06-13-17-51_stereo_centre_01")
    train = DataSet(train_images, train_labels, dtype="uint8")
    validation = DataSet(validation_images, validation_labels, dtype="uint8")
    test = DataSet(test_images, test_labels, dtype="uint8")
    return Datasets(train=train, validation=validation, test=test)

def _get_images_and_labels(listdir):
    total_num_examples = 0
    labels = []
    frames_idx_map = {}
    for dir in listdir:
        labels_filename = os.path.join(dir, P_FILENAME)
        raw_labels = numpy.load(labels_filename)[DEFAULT_MAIN_KEY]
        num_examples = raw_labels.size
        total_num_examples += num_examples
        idxs = []
        for i in range(num_examples):
            single_raw_label = raw_labels[i]
            src_idx = single_raw_label['src_idx']
            dst_idx = single_raw_label['dst_idx']
            idxs.append((src_idx, dst_idx))
            label = single_raw_label[DEFAULT_LABEL_KEY].reshape(LABELS_SIZE)
            labels.append(label)
        if dir in frames_idx_map:
            raise ValueError("Duplicate directory: " + dir)
        frames_idx_map[dir] = idxs
    assert len(labels) == total_num_examples
    images = numpy.empty((total_num_examples, IMAGE_HEIGHT, IMAGE_WIDTH, 2))
    iter = 0
    for dir in listdir:
        images_filename = os.path.join(dir, "images.npz")
        dataset = numpy.load(images_filename)[DEFAULT_MAIN_KEY]
        if dir not in frames_idx_map:
            raise ValueError(dir + " directory")
        for (src_idx, dst_idx) in frames_idx_map[dir]:
            images[iter,...,0] = dataset[src_idx] * (1.0 / 255.0)
            images[iter,...,1] = dataset[dst_idx] * (1.0 / 255.0)
            iter += 1
    assert total_num_examples == iter
    return images, numpy.array(labels)


def _inputs(dir):
    main_key = 'arr_0'
    images_filename = os.path.join(dir,"images.npz")
    labels_filename = os.path.join(dir,"p.npz")
    dataset = numpy.load(images_filename)[main_key]
    raw_labels = numpy.load(labels_filename)[main_key]
    num_examples = raw_labels.size
    images = numpy.empty((num_examples, IMAGE_HEIGHT, IMAGE_WIDTH, 2))
    #images = []
    labels = []
    for i in range(num_examples):
        single_raw_label = raw_labels[i]
        src_idx = single_raw_label['src_idx']
        dst_idx = single_raw_label['dst_idx']
        label = single_raw_label['P'].reshape(LABELS_SIZE)
        labels.append(label)
        frame_1 = dataset[src_idx]
        frame_2 = dataset[dst_idx]
        images[i,...,0] = frame_1
        images[i,...,1] = frame_2
        #images.append((frame_1, frame_2))
    print images.dtype, images.dtype
    ### images,
    return images, numpy.array(labels)
