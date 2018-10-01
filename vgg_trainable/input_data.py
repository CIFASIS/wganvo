import numpy
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import collections
from sklearn.model_selection import GroupKFold
from six.moves import xrange
import transformations

Datasets = collections.namedtuple('Datasets', ['train', 'cross_validation_splits', 'test'])
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 2
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
LABELS_SIZE = 6
DEFAULT_MAIN_KEY = 'arr_0'
P_FILENAME = "t.npz"
DEFAULT_LABEL_KEY = "T"

class DataSet(object):
  def __init__(self,
               images,
               labels,
               groups=None,
               fake_data=False,
               one_hot=False,
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
    #if dtype not in ("uint8", "float32"):
    #  raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
    #                  dtype)
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
    self._groups = groups
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._targets_mean = numpy.mean(labels, axis=0)
    self._targets_std = numpy.std(labels, axis=0) 


  @property
  def targets_mean(self):
    return self._targets_mean

  @property
  def targets_std(self):
    return self._targets_std

  @property
  def images(self):
    return self._images

  @property
  def groups(self):
      return self._groups

  @property
  def images_norm(self):
    return self._images * (1.0 / 255.0)

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property # Deprecar
  def epochs_completed(self):
    return self._epochs_completed

  def reset_epoch(self):
    self._index_in_epoch = 0

  def next_batch(self, batch_size, fake_data=False, shuffle=True, standardize_targets=False):
    im, lb = self._next_batch(batch_size, fake_data, shuffle)
    if standardize_targets:
        lb = (lb - self._targets_mean) / self._targets_std
    return im * (1.0 / 255.0), lb

  def _next_batch(self, batch_size, fake_data, shuffle):
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
      if self.groups is not None:
        self._groups = self.groups[perm0]
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
        if self.groups is not None:
          self._groups = self.groups[perm]
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



def get_list_of_subdirectories(data_dir):
    return [os.path.join(data_dir, item) for item in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, item))]

def read_data_sets(data_dir, kfold=None, rot_tolerance=0.):
    list_dir = get_list_of_subdirectories(data_dir)#[os.path.join(train_data_dir, item) for item in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, item))]
    #test_list_dir = get_list_of_subdirectories(test_data_dir)
    #if validation_data_dir is not None:
    #    validation_list_dir = get_list_of_subdirectories(validation_data_dir)
    #    validation_images, validation_labels = _get_images_and_labels(validation_list_dir)
    #else:
    # FIXME obtener desde train data
    #    validation_images, validation_labels = _inputs(
    #        "/home/cremona/workspace/train/2014-05-06-12-54-54_stereo_centre_01")
    return _get_images_and_labels(list_dir, kfold=kfold, rot_tolerance=rot_tolerance)
    #test_images, test_labels, _ = _get_images_and_labels(test_list_dir)


    #test_images, test_labels = _inputs("/home/javo/Descargas/Backup/workspace/2014-05-06-13-17-51_stereo_centre_01")
    #train = DataSet(train_images, train_labels, fake_data=fake_data)
    #cross_validation_splits = splits #DataSet(validation_images, validation_labels, fake_data=fake_data)
    #test = DataSet(test_images, test_labels, fake_data=fake_data)
    #return Datasets(train=train, cross_validation_splits=cross_validation_splits, test=test)

def _get_images_and_labels(list_of_subdir, images_dtype="uint8", labels_dtype="float32", kfold=None, rot_tolerance=0.):
    total_num_examples = 0
    labels = []
    frames_idx_map = {}

    # Process targets
    for dir in list_of_subdir:
        labels_filename = os.path.join(dir, P_FILENAME)
        raw_labels = numpy.load(labels_filename)[DEFAULT_MAIN_KEY]
        num_examples = raw_labels.size
        total_num_examples += num_examples
        idxs = []
        for i in range(num_examples):
            single_raw_label = raw_labels[i]
            src_idx = single_raw_label['src_idx']
            dst_idx = single_raw_label['dst_idx']
            rt = single_raw_label[DEFAULT_LABEL_KEY]#.reshape(LABELS_SIZE)
            rt = numpy.asmatrix(rt)
            #ax, ay, az = transformations.euler_from_matrix(rt[0:3,0:3])
            ax, ay, az = transformations.euler_from_matrix(numpy.vstack((rt,[0,0,0,1.])))
            label = numpy.array([rt[0,3],rt[1,3],rt[2,3], ax, ay, az])
            labels.append(label)
            idxs.append((src_idx, dst_idx))

        if dir in frames_idx_map:
            raise ValueError("Duplicate directory: " + dir)
        frames_idx_map[dir] = idxs
    assert rot_tolerance or len(labels) == total_num_examples

    # Process images
    images = numpy.empty((len(labels), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=images_dtype)
    groups = numpy.empty(len(labels))
    group_idx = 0
    iter = 0
    for dir in list_of_subdir:
        images_filename = os.path.join(dir, "images.npz")
        dataset = numpy.load(images_filename)[DEFAULT_MAIN_KEY]
        assert dataset.dtype == images_dtype
        if dir not in frames_idx_map:
                raise ValueError(dir + " directory")
        # group_number = (group_idx % kfold)
        group_idx += 1
        for (src_idx, dst_idx) in frames_idx_map[dir]:
                images[iter,...,0] = dataset[src_idx]# * (1.0 / 255.0)
                images[iter,...,1] = dataset[dst_idx]# * (1.0 / 255.0)
                # Images from the same dir must be in the same fold. See GroupKFold from sklearn.
                groups[iter] = group_idx
                iter += 1
    assert len(labels) == iter
    im = images
    lb = numpy.array(labels, dtype=labels_dtype)
    splits = None
    if kfold is not None:
        gkf = GroupKFold(n_splits=kfold)
        splits = gkf.split(images, labels, groups = (groups % kfold))
    return im, lb, splits, groups

if __name__ == '__main__':
    pass
