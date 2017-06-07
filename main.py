import wfdb as wf
import numpy as np
from datasets import mitdb as dm

# Load paths of available data files
records = dm.get_records()
print 'There are {} record files'.format(len(records))

# Select one of them
path = records[0]
print 'Loading file:', path

# Read in the data ...
record = wf.rdsamp(records[0])

# .. and annotations
annotation = wf.rdann(records[0], 'atr')

from matplotlib import pyplot as plt

# Select one of the channels (there are two)
chid = 0
data = record.p_signals
channel = data[:, chid]

print 'ECG channel type:', record.signame[chid]

# Plot only the first 2000 samples
howmany = 2000

# Calculate time values in seconds
times = np.arange(howmany, dtype = 'float') / record.fs
plt.plot(times, channel[ : howmany])
plt.xlabel('Time [s]')
plt.xlim([0, 5])
plt.show()



