import os
import h5py
from glob import glob

def clean_path(path):
    """ rm * """
    paths = glob(path + '/*')
    for file in paths:
        os.remove(file)
    
    print 'Cleaned directory:', path

def make_spoko_mnist():
    """ Convert mnist data to more familiar structure """
    # Get data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Think signals
    # Training
    t_signals = mnist.train.images
    t_labels  = mnist.train.labels

    # Validation
    v_signals = mnist.validation.images
    v_labels  = mnist.validation.labels

    # ?
    savepath = 'data/mnist.storage'

    with h5py.File(savepath, 'w') as db:
        # Validation
        v_group = db.create_group('validation')

        v_group.create_dataset('signals', data = v_signals)
        v_group.create_dataset('labels',  data = v_labels)

        # Training
        t_group = db.create_group('training')

        t_group.create_dataset('signals', data = t_signals)
        t_group.create_dataset('labels',  data = t_labels)

    print "Saved mnist data to:", savepath

