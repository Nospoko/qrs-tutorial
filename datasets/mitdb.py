import os
import h5py
import numpy as np
import pandas as pd
from glob import glob
from utils import download as ud
from matplotlib import pyplot as plt

def get_records():
    """ Get paths for data in data/mit/ directory """
    # Download if doesn't exist
    if not os.path.isdir('data/mitdb'):
        print 'Downloading the mitdb ecg database, please wait'
        ud.download_mitdb()
        print 'Download finished'

    # There are 3 files for each record
    # *.atr is one of them
    paths = glob('data/mitdb/*.atr')

    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    paths.sort()

    return paths

def good_annotations():
    """ Const function with good annotations """
    # For now it seems those are most popular
    good_annotations = [1, 2, 3, 4,
                        5, 6, 7, 8,
                        9, 10, 11, 12,
                        13, 16, 31, 38]

    return good_annotations

def make_hdf(savepath, params = [1024, 2]):
    """ Sick """
    # Prepare dataset defining parameters
    in_shape = params[0]
    r_factor = params[1]
    params = [in_shape, r_factor]

    sigsize = in_shape
    labsize = in_shape * r_factor

    paths = get_records()

    with h5py.File(savepath, 'w') as db:
        sigshape = [1000, sigsize]
        sigmax = [None, sigsize]
        sigset = db.create_dataset('training/inputs',
                                    shape = sigshape,
                                    maxshape = sigmax)

        labshape = [1000, labsize]
        labmax = [None, labsize]
        labset = db.create_dataset('training/outputs',
                                    shape = labshape,
                                    maxshape = labmax)

        end = 0
        for path in paths:
            print 'Making data from:', path
            # inputs, outputs = make_qrs_set(path, params)
            inputs, outputs = make_set(path, params)

            sta = end
            end += inputs.shape[0]

            sigset.resize([end, sigsize])
            sigset[sta : end] = inputs

            labset.resize([end, labsize])
            labset[sta : end] = outputs

    print 'Dataset saved into:', savepath

