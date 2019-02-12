import numpy as np
import time
import h5py

def make_generator(path, n_files, batch_size):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        files = range(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        with h5py.File(path, 'r') as hf:
            ds = hf[hf.keys()[0]]
            print(ds.shape)
            for n, i in enumerate(files):
                image = ds[i] #scipy.misc.imread("{}/{}.png".format(path, str(i+1).zfill(len(str(n_files)))))
                images[n % batch_size] = image #.transpose(2,0,1)
                if n > 0 and n % batch_size == 0:
                    yield (images,)
    return get_epoch

def load(batch_size, data_dir):
    return make_generator(data_dir, 1281167, batch_size)

if __name__ == '__main__':
    train_gen, valid_gen = load(64,'/home/shared/datasets/imagenet/imagenet.64x64.hdf5')
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
