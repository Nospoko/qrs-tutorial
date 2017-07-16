import matplotlib as mpl
mpl.use('Agg')
import wfdb as wf
import numpy as np
from utils import storage as us
from utils import plotters as up
from datasets import mitdb as dm
from models import abstract as ma
from matplotlib import pyplot as plt

def training():
    """ Of some model """
    # Define the problem
    in_width = 100
    out_width = 100
    params = [in_width, out_width]

    # Prepare the data
    dataset = create_dataset(params)

    # Define the architecture
    name = 'first_model'
    model = mb.BasicModel(name, params)

    # Perform training ...
    model.train(dataset)

    # ... and validation
    v_signals, v_labels = dataset.validation_set()
    score = model.consume(v_signals, v_labels)
    print 'Final score:', score

def main():
    # dm.create_datasets()

    dataset = us.Dataset('data')

    params = [300]
    netname = 'foo'

    # Create model
    model = ma.FirstTry(netname, params)

    # How much data will it see
    model.set_epochs(1000)
    model.set_batch_size(512)

    # Learning tactic
    lrs = [0.003, 0.0007, 0.0001]
    model.set_learning_rates(lrs)

    model.train(dataset)

    # Extract the validation dataset
    howmany = 1000
    si, la = dataset.test_batch(howmany)

    # Calculate loss rescaled 'per-example'
    loss = model.consume(si, la)
    loss = loss * model.batch_size / howmany
    
    print 'Validation loss', loss

    # Show some results
    score = model.process(si)
    its = range(500)
    its = its[::17]
    for it in its:
        plt.plot(score[it])
        plt.plot(la[it])
        savepath = 'tmp/result-{}.png'.format(it)
        plt.savefig(savepath)
        plt.clf()

if __name__ == '__main__':
    main()
