
import numpy as np
import matplotlib
from matplotlib import pyplot
import tensorflow as tf

from sklearn.utils import shuffle
from skimage import exposure

import pickle

def load_pickled_data(file, cols): 
    """
    Loads pickled training and test data.
    
    Parameters
    ----------
    file    : Name of the pickle file.
              
    columns : list of strings
              List of columns in pickled data we're interested in.

    Returns
    -------
    A tuple of datasets for given columns.    
    """
    
    with open(file, mode='rb') as f: 
        dataset = pickle.load(f)
        
    return tuple(map(lambda c: dataset[c], cols))
    
    



