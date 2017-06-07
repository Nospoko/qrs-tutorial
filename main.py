import wfdb as wf
import numpy as np
from utils import plotters as up
from datasets import mitdb as dm
from matplotlib import pyplot as plt

# Load paths of available data files
records = dm.get_records()
print 'There are {} record files'.format(len(records))

# Read in the data
path = records[11]
print 'Using file:', path
up.show_annotations(path)
