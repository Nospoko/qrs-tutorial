import matplotlib as mpl
mpl.use('Agg')
import wfdb as wf
import numpy as np
from utils import storage as us
from utils import plotters as up
from datasets import mitdb as dm
from models import abstract as ma
from matplotlib import pyplot as plt

def main():
    # dm.create_datasets()
    dataset = us.Dataset('data')

    params = [300]
    netname = 'foo'

    # Create model
    model = ma.FirstTry(netname, params)

    # How much data will it see
    model.set_epochs(500)
    model.set_batch_size(32)

    # Learning tactic
    lrs = [0.003, 0.0007, 0.0001]
    model.set_learning_rates(lrs)

    model.train(dataset)

    # Extract the validation dataset
    howmany = 1000
    si, la = dataset.test_batch(howmany)

    # Calculate loss
    loss = model.consume(si, la)
    
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

    # Also extract the final filter
    filtr = model.get_tensor('conv_out/weights:0')
    filtr = filtr.reshape([len(filtr)])

    # To png
    plt.plot(filtr)
    plt.savefig('tmp/filter.png')
    plt.clf()

if __name__ == '__main__':
    main()
