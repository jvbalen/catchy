from __future__ import division

import os
import numpy as np
import pandas as pd


# what to do if skip_cols == 'auto'
column_range = {'loudness': (1, None),
                'sharpness': (1, None),
                'roughness': (1, None),
                'bands': (1, None),
                'melody': (1, None),
                'hpcp': (1, None),
                'mfcc': (2, 14)}


def read_feature(filename, mode='pandas', time=False, skip_cols=(0, None)):
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
    # if filename is a list, use os.path.join to join
    if type(filename) is list:
        filename = os.path.join(*filename)
    if not (filename.endswith('.csv') or filename.endswith('.txt')):
        filename += '.csv'

    # if no skip_cols 
    if skip_cols == 'auto':
        dir_name = os.path.dirname(filename)
        subdir_name = os.path.basename(dir_name)

        skip_cols = skip_columns(subdir_name)

    # pick csv reader
    if mode == 'numpy':
        data = np.genfromtxt(filename, delimiter=',')
    elif mode == 'pandas':
        data = pd.read_csv(filename, delimiter=',', header=None).values

    feature = data[:, skip_cols[0]:skip_cols[1]]

    # if time=True, split first and following columns
    if time:
        t = feature[:, 0]
        x = feature[:, 1:]
        feature = (t, x)
        
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


def skip_columns(feature_name, default_range=(0,None)):
    """Set automatic column ignore behavior in read_feature().
    """
    first_col, last_col = column_range.get(feature_name, default_range)

    return first_col, last_col


def dataset_from_dir(audio_dir, separator='-'):
    """Make a dictionary of song section paths grouped by song id
    from audio files in a particular directory.
    Assumes files are labeled 'songid-sectionid.wav', where the
    dash is the separator specified in the separator parameter.
    Extension can be 'wav' or 'mp3'.

    Args:
        audio_dir (str): path to audio dir.
        separator (str): character or string that separates song
            id and section id in the audio file names.
    
    Returns:
        segment_dict (dict): dictionary of song segments, containing
        all segment paths (without extension) as a list, grouped by
        song id.
    """
    segment_dict = {}
    for file_path in os.listdir(audio_dir):

        if file_path.endswith('.wav') or file_path.endswith('.mp3'):
            filename = os.path.basename(file_path).split('.')[0]
            song_id = filename.split(separator)[0]

            if song_id in segment_dict:
                segment_dict[song_id].append(filename)
            else:
                segment_dict[song_id] = [filename]

    return segment_dict
