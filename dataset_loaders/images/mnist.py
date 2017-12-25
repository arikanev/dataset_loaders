"""Mnist dataset."""
import numpy as np
import os
import time
import pickle as pkl
import errno
from PIL import Image
from dataset_loaders.parallel_loader import ThreadedDataset
from timeit import default_timer as timer


class MnistDataset(ThreadedDataset):
    """The mnist handwritten digit dataset

    The dataset should be downloaded from [1] into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.

     References
    ----------
    [1] Mnist dataset pickle file:
        your_username_here@elisa1.iro.umontreal.ca:/data/lisa/data/mnist/mnist_seg/

    """

    name = 'mnist'

    # optional arguments
    data_shape = (28, 28, 3)
    mean = [0, 0, 0]
    std = [1, 1, 1]
    max_files = 50000

    GTclasses = range(2)
    mapping_type = 'mnist'

    non_void_nclasses = 2
    _void_labels = []
    GTclasses = range(2)
    #GTclasses = GTclasses + [-1]

    _mask_labels = {
        0: 'background',
        1: 'digit',
        }

    _cmap = {
        0: (0),           # background
        1: (255),           # digit
        }

    _filenames = None

    def __init__(self, which_set='train', *args, **kwargs):
        """Construct the ThreadedDataset.

        it also creates/copies the dataset in self.path if not already there
     
          mnist data is in 3 directories train, test, valid)
        
        """

        self.which_set = 'val' if which_set == 'valid' else which_set

        # set file paths
        if which_set == 'train':
            self.image_path = os.path.join(self.path, 'train_images')
            self.mask_path = os.path.join(self.path, 'train_masks')

        elif which_set == 'test':
            self.image_path = os.path.join(self.path, 'test_images')
            self.mask_path = os.path.join(self.path, 'test_masks')

        else:
            self.image_path = os.path.join(self.path, 'val_images')
            self.mask_path = os.path.join(self.path, 'val_masks')

        super(MnistDataset, self).__init__(*args, **kwargs)
    


    @property
    def filenames(self):
        """Get file names for this set."""
        if self._filenames is None:
            filenames = []

            for i in range(len(os.listdir(self.image_path))):
                filenames.append(str(i).zfill(5) + '.png')

            self._filenames = filenames

            print('MnistDataset: ' + self.which_set + ' ' + str(len(filenames)) + ' files')

        return self._filenames

    def get_names(self):
        """Return a dict of mnist filenames."""

        return {'default': self.filenames}

    def load_sequence(self, sequence):
        """Load a sequence of images/frames.

        Auxiliary function that loads a sequence of mnist images with
        the corresponding ground truth mask and the filenames.
        Returns a dict with the images in [0, 1], and their filenames.
        """
        X = []
        Y = []
        F = []

        for prefix, image in sequence:

            # open mnist image, convert to numpy array
            curr_mnist_im = np.array(Image.open(os.path.join(self.image_path, image))).astype('float32')
            # append image to X
            X.append(curr_mnist_im)

            # append image fname to F
            F.append(image)

            # open mnist mask, convert to numpy array
            curr_mnist_mask = np.array(Image.open(os.path.join(self.mask_path, image))).astype('int32')
            curr_mnist_mask = curr_mnist_mask / 255
            # append mask to Y
            Y.append(curr_mnist_mask)

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test():
    """Test."""
    trainiter = MnistDataset(
        which_set='train',
        batch_size=10,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
             'crop_size': (28, 28)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=True)

    validiter = MnistDataset(
        which_set='valid',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
             'crop_size': (28, 28)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=False)

    train_nsamples = trainiter.nsamples
    nbatches = trainiter.nbatches
    print("Train %d" % (train_nsamples))

    valid_nsamples = validiter.nsamples
    print("Valid %d" % (valid_nsamples))

    # Simulate training
    max_epochs = 2
    start_training = time.time()
    for epoch in range(max_epochs):
        start_epoch = time.time()
        for mb in range(nbatches):
            start_batch = time.time()
            trainiter.next()
            print("Minibatch {}: {} seg".format(mb, (time.time() -
                                                     start_batch)))
        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))


def run_tests():
    """Run tests."""
    test()


if __name__ == '__main__':
    run_tests()
