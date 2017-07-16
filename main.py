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
    model = ma.FirstTry(netname, params)
    model.set_epochs(1000)
    lrs = [0.003, 0.0007, 0.0001]
    model.set_learning_rates(lrs)

    model.train(dataset)

    # Extract the validation dataset
    si = dataset.validation_set['signals']
    la = dataset.validation_set['labels']
    la = la[:, 1, :]
    la = la

    loss = model.consume(si, la)
    print 'Validation loss', loss

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
