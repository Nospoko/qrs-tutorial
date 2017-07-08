import wfdb as wf
import numpy as np
from scipy import signal as ss
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

def show_objective():
    """ For the model """
    # Choose a record
    records = dm.get_records()
    path = records[17]
    record = wf.rdsamp(path)
    ann = wf.rdann(path, 'atr')

    chid = 0
    print 'Channel:', record.signame[chid]

    cha = record.p_signals[:, chid]

    # These were found manually
    sta = 184000
    end = sta + 1000
    times = np.arange(end-sta, dtype = 'float')
    times /= record.fs

    # Extract the annotations for that fragment
    where = (sta < ann.annsamp) & (ann.annsamp < end)
    samples = ann.annsamp[where] - sta
    print samples

    # Prepare dirac-comb type of labels
    qrs_values = np.zeros_like(times)
    qrs_values[samples] = 1

    # Prepare gaussian-comb type of labels
    kernel = ss.hamming(36)
    qrs_gauss = np.convolve(kernel,
                            qrs_values,
                            mode = 'same')

    # Make the plots
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(times, cha[sta : end])

    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax2.plot(times,
             qrs_values,
             'C1',
             lw = 4,
             alpha = 0.888)
    ax3 = fig.add_subplot(3,1,3, sharex=ax1)
    ax3.plot(times,
             qrs_gauss,
             'C3',
             lw = 4,
             alpha = 0.888)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.xlabel('Time [s]')
    plt.xlim([0, 2.5])
    plt.show()

def show_objective_part2():
    """ For the model """
    # Choose a record
    records = dm.get_records()
    path = records[13]
    record = wf.rdsamp(path)
    ann = wf.rdann(path, 'atr')

    chid = 0
    print 'File:', path
    print 'Channel:', record.signame[chid]

    cha = record.p_signals[:, chid]

    # These were found manually
    sta = 184000
    end = sta + 1000
    times = np.arange(end-sta, dtype = 'float')
    times /= record.fs

    # Extract the annotations for that fragment
    where = (sta < ann.annsamp) & (ann.annsamp < end)
    samples = ann.annsamp[where] - sta
    print samples

    # Prepare dirac-comb type of labels
    qrs_values = np.zeros_like(times)
    qrs_values[samples] = 1

    # Prepare gaussian-comb type of labels
    kernel = ss.hamming(36)
    qrs_gauss = np.convolve(kernel,
                            qrs_values,
                            mode = 'same')

    # Make the plots
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(times, cha[sta : end])
    ax1.set_title('Input', loc = 'left')

    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.plot(times,
             qrs_gauss,
             'C3',
             lw = 4,
             alpha = 0.888)
    ax2.set_title('Output', loc = 'left')
    ax1.grid()
    ax2.grid()
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xlabel('Time [s]')
    plt.xlim([0, 2.5])
    plt.show()
