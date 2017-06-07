import wfdb as wf
import numpy as np
from datasets import mitdb as dm

# Load paths of available data files
records = dm.get_records()
print 'There are {} record files'.format(len(records))

# Select one of them
path = records[0]
print 'Loading file:', path

# Number of samples to read in
howmany = 1000

# Read in the data ...
record = wf.rdsamp(records[0], sampto = howmany)

# .. and annotations
annotation = wf.rdann(records[0], 'atr', sampto = howmany)

# Print some data
print 'Sampling frequency used for this record:', record.fs
print 'Shape of loaded data array:', record.p_signals.shape
print 'Number of loaded annotations:', len(annotation.annids)
print 'Third annotation is of type:', annotation.anntype[2]
