import wfdb as wf
import numpy as np
from utils import plotters as up
from datasets import mitdb as dm
from matplotlib import pyplot as plt

def make_dataset(records, savepath):
    """ Inside an array """
    # Prepare containers
    signals, labels = [], []

    # Iterate files
    for path in records:
        record = wf.rdsamp(path)
        annotations = wf.rdann(path, 'atr')

        # Extract pure signals
        data = record.p_signals

        # Convert each channel into labeled fragments
        signal, label = convert_data(data, annotations)

        # Cumulate
        signals.append(signal)
        labels.append(label)

    # Convert to numpy.array
    signals = np.array(signals)
    labels = np.array(labels)

    # Write to disk
    np.save(savepath, {'signals' : signals,
                       'labels'  : labels }
        

