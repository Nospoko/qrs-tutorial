import numpy as np

class Dataset(object):
    """ Access data """
    def __init__(self, datapath):
        """ Loads all of the data """
        # Load training ...
        train_path = datapath + '/training.npy'
        self.training_set = np.load(train_path)[()]

        # ... validation
        valid_path = datapath + '/validation.npy'
        self.validation_set = np.load(valid_path)[()]

        # ... and test
        test_path = datapath + '/test.npy'
        self.test_set = np.load(test_path)[()]

        # Prepare randomized access to training data
        self.nof_examples = len(self.training_set['signals'])
        print 'Dataset with {} trianing examples'.format(self.nof_examples)

        self.ids = np.arange(self.nof_examples)
        np.random.shuffle(self.ids)

        # Prepare iterators
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        """ Iterate """
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.nof_examples:
        # Finished epoch
            self.epochs_completed += 1
            print 'Data epochs done:', self.epochs_completed

            # Shuffle the data accessing iterators (again)
            np.random.shuffle(self.ids)

            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size

        end = self.index_in_epoch

        # Get random row numbers
        ids = self.ids[start : end]

        signals = self.training_set['signals'][ids]
        labels = self.training_set['labels'][ids]

        # Get rid of the dirac solutions
        labels = labels[:, 1, :]

        return signals, labels

            
