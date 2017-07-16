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

        # Prepare randomized access to training data ...
        self.nof_examples = len(self.training_set['signals'])
        print 'Dataset with {} trianing examples'.format(self.nof_examples)

        # ... and validation ...
        self.nof_validation = len(self.validation_set['signals'])
        print 'Dataset with {} validation examples'.format(self.nof_validation)

        # ... and test
        self.nof_test = len(self.test_set['signals'])
        print 'Dataset with {} test examples'.format(self.nof_test)

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

    def validation_batch(self, batch_size):
        """ Just a random selection """
        # Prepare random indices
        every = range(self.nof_validation)
        ids = np.random.choice(every, batch_size)

        # Cut out from the full set
        signals = self.validation_set['signals'][ids]
        labels = self.validation_set['labels'][ids]

        # Get rid of the dirac solutions
        labels = labels[:, 1, :]

        return signals, labels

    def test_batch(self, batch_size):
        """ Just a random selection """
        # Prepare random indices
        every = range(self.nof_test)
        ids = np.random.choice(every, batch_size)

        # Cut out from the full set
        signals = self.test_set['signals'][ids]
        labels = self.test_set['labels'][ids]

        # Get rid of the dirac solutions
        labels = labels[:, 1, :]

        return signals, labels

            
