from __future__ import division

import os
import numpy as np
import pandas as pd


def read_feature(filename, mode='pandas', time=True, skip_cols=(0, None)):
    """Read features from CSV.

    This is not a general purpose i/o function. It is written to work well
        with frame-based features, and with this module's `write_features()'
        in particular.

    Args:
        filename (list or str): file name. If list, will use
            os.path.join to join dir names and filename. CSV extension will be
            added if not already included
        mode (str): choose between 'pandas' and 'numpy'.
            Pandas is faster when reading large files.
        time (bool): set True to split data in a column of frame times and
            2d-array of frame data.
    """
    if skip_cols = 'auto':
        dir_name = os.path.dirname(filename)
        subdir_name = os.path.basename(dir_name)
        
        skip_cols = skip_columns(subdir_name)

    # if filename is a list, use os.path.join to join
    if type(filename) is list:
        filename = os.path.join(*filename)
    if not (filename.endswith('.csv') or filename.endswith('.txt')):
        filename += '.csv'

    # pick csv reader
    if mode == 'numpy':
        data = np.genfromtxt(filename, delimiter=',')
    elif mode == 'pandas':
        data = pd.read_csv(filename, delimiter=',', header=None).values

    # split time column and frames
    if time:
        t = data[:, 0]
        x = data[:, 1+skip_cols[0]:1+skip_cols[1]]
        feature = (t, x)
    else:
        feature = data[:, skip_cols[0]:skip_cols[1]]
        
    return feature


def write_feature(data, filename):
    """Write frame-based features to CSV.

    Args:
        data (nd-array or list): feature matrix or list of feature matrices
            if list, feature matrices will be concatenated
            (1d-arrays will be reshaped into column vectors).
        filename (list or str): file name. If list, will use
            os.path.join to join dirs and filename. CSV extension will be
            added if not already included.

    Usage:
        >>> # simplest case
        >>> # write an array of ones to temp.csv
        >>> X = np.ones((100, 30))
        >>> write_csv(X, 'temp.csv')

        >>> # with lists
        >>> # write indexed array of ones to data/ones/0.csv
        >>> t = np.arange(100)
        >>> X = np.ones((100, 30))
        >>> feature_name, id = 'ones', str(0)
        >>> write_csv([t, X], ['data', feature_name, id])
    """

    # if data is a list of nd-arrays, hstack as 2d-arrays
    if type(data) is list:
        for i in np.where([len(x.shape) == 1 for x in data]):
            data[i] = data[i][:, np.newaxis]
        data = np.hstack(data)
    elif len(data.shape) == 1:
        data = data[:, np.newaxis]

    # if filename is a list, use os.path.join to join
    if type(filename) is list:
        filename = os.path.join(*filename)
    if not (filename.endswith('.csv') or filename.endswith('.txt')):
        filename += '.csv'

    dataframe = pd.DataFrame(data)
    dataframe.to_csv(filename, header=False, index=False)


def skip_columns(feature_name):
    """Set automatic column ignore behavior in read_feature().
    """
    dict_first_col = {'loudness':   1,
                      'sharpness':  1,
                      'bands':      1,
                      'melody':     1,
                      'hpcp':       1,
                      'mfcc':       2}
    dict_last_col = {'mfcc':       14}

    first_col = dict_first_col.get(feature_name, 0)
    last_col = dict_last_col.get(feature_name, None)

    return first_col, last_col
