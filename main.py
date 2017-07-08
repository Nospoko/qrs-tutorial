import wfdb as wf
import numpy as np
from utils import plotters as up
from datasets import mitdb as dm
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
    dm.create_datasets()

if __name__ == '__main__':
    main()
