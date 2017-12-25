"""Mnist-m dataset."""
import numpy as np
import os
import time
from PIL import Image
from dataset_loaders.parallel_loader import ThreadedDataset


class MnistMDataset(ThreadedDataset):
    """The mnist-m handwritten digit dataset

    The dataset should be downloaded from [1]

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.

     References
    ----------
    [1] Mnist-m dataset file:
        your_username_here@elisa1.iro.umontreal.ca:/data/lisa/data/mnistm/images/

    """

    name = 'mnistm'

    # optional arguments
    data_shape = (28, 28, 3)
    mean = [0, 0, 0]
    std = [1, 1, 1]
    max_files = 50000

    mapping_type = 'mnist'

    n_classes = 2

    non_void_nclasses = 2
    _void_labels = []
    GTclasses = range(2)
    # GTclasses = GTclasses + [-1]

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

          mnistm_data is in 3 directories: train, test, val;
          train contains 50000 images
          test contains 9000 images
          val contains 9000 images

        """
        self.which_set = 'val' if which_set == 'valid' else which_set

        # set file paths
        if which_set == 'train':
            self.im_path = os.path.join(self.path, 'train_images')
            self.mask_path = os.path.join(self.path, 'train_masks')
        elif which_set == 'test':
            self.im_path = os.path.join(self.path, 'test_images')
            self.mask_path = os.path.join(self.path, 'test_masks')
        else:
            self.im_path = os.path.join(self.path, 'val_images')
            self.mask_path = os.path.join(self.path, 'val_masks')

        super(MnistMDataset, self).__init__(*args, **kwargs)

    @property
    def filenames(self):
        """Get file names for this set."""
        if self._filenames is None:
            filenames = []

            for i in range(len(os.listdir(self.im_path))):
                filenames.append(str(i).zfill(5) + '.png')

            print('MnistMDataset: ' + self.which_set +
                  ' ' + str(len(filenames)) + ' files')

        return filenames

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
            curr_mnistm_im = Image.open(os.path.join(self.im_path, image))
            curr_mnistm_im = np.array(curr_mnistm_im).astype('float32')

            # append image to X
            X.append(curr_mnistm_im)

            # append image fname to F
            F.append(image)

            # open mnist mask, convert to numpy array
            curr_mnistm_mask = Image.open(os.path.join(self.mask_path, image))
            curr_mnistm_mask = np.array(curr_mnistm_mask).astype('int32')
            curr_mnistm_mask = curr_mnistm_mask / 255

            # append mask to Y
            Y.append(curr_mnistm_mask)

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test():
    """Test."""
    trainiter = MnistMDataset(
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

    validiter = MnistMDataset(
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
