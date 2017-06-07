import wfdb as wf
import numpy as np
from datasets import mitdb as dm
from matplotlib import pyplot as plt

def show_path(path):
    """ As a plot """
    # Read in the data
    record = wf.rdsamp(path)
    annotation = wf.rdann(path, 'atr')
    data = record.p_signals
    cha = data[:, 0]
    print 'Channel type:', record.signame[0]
    times = np.arange(len(cha), dtype = float)
    times /= record.fs
    plt.plot(times, cha)
    plt.xlabel('Time [s]')
    plt.show()

def show_annotations(path):
    """ Exemplary code """
    record = wf.rdsamp(path)
    annotation = wf.rdann(path, 'atr')

    # Get data and annotations for the first 2000 samples
    howmany = 2000
    channel = record.p_signals[:howmany, 0]

    # Extract all of the annotation related infromation
    where = annotation.annsamp < howmany
    samp = annotation.annsamp[where]

    # Convert to numpy.array to get fancy indexing access
    types = np.array(annotation.anntype)
    types = types[where]

    times = np.arange(howmany, dtype = 'float') / record.fs
    plt.plot(times, channel)

    # Prepare qrs information for the plot
    qrs_times = times[samp]

    # Scale to show markers at the top 
    qrs_values = np.ones_like(qrs_times)
    qrs_values *= channel.max() * 1.4

    plt.plot(qrs_times, qrs_values, 'ro')

    # Also show annotation code
    # And their words
    for it, sam in enumerate(samp):
        # Get the annotation position
        xa = times[sam]
        ya = channel.max() * 1.1

        # Use just the first letter 
        a_txt = types[it]
        plt.annotate(a_txt, xy = (xa, ya))

    plt.xlim([0, 4])
    plt.xlabel('Time [s]')
    plt.show()
