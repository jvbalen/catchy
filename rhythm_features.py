
from __future__ import division, print_function

import os
import numpy as np

import utils


onsets_dir = ''
beats_dir = ''


def compute_and_write(data_dir, track_list=None, features=None):
    """Compute frame-based features for all audio files in a folder.

    Args:
        data_dir (str): where to write features
        track_list (str or None): list of file ids. Set to None to infer from
            files in ioi_dir and chroma_dir.
        features (dict): dictionary with (unique) feature names as keys and 
            tuples as values, each containing a feature extraction function and a
            parameter dictionary.
            Feature extraction functions can be any function that returns one
                or more 1d or 2d-arrays that share their first dimension.

    Required global variables:
        beats_dir (str): where to find beat data
        onsets_dir (str): where to find onset data
    """
    if track_list is None:
        track_list = [filename.split('.')[0] for filename in os.listdir(ioi_dir)]
    
    if features is None:
        features = {'ioihist': (get_ioi_hist, {})}

    for track_id in track_list:

        print("Computing features for track {}...".format(track_id))

        for feature in features:

            # run feature function
            func, params = features[feature]
            X = func(track_id, **params)

            # normalize (!) and flatten
            X = X.flatten() / np.sum(X)

            # write
            utils.write_feature(X, [data_dir, feature, track_id])


def get_ioi_hist(track_id, min_length = -7, max_length = 0, step=1):
    """Compute a IOI histogram, with bins logarithmically spaced between
            `min_length` (def: -7) and `max_length` (0), with step `step`.
    """
    t, ioi = get_norm_ioi(track_id)

    log_ioi = np.log2(ioi)

    halfstep = step / 2.0
    nbins = (max_length - min_length) / step + 1
    binedges = np.linspace(minpitch - halfstep, maxpitch + halfstep, nbins + 1)

    ioi_hist, _ = np.histogram(log_ioi, binedges)
    ioi_hist = ioi_hist / np.sum(ioi_hist)

    return ioi_hist


def get_beats(track_id):
    """Read beat data from file beats_dir + track_id + '.csv'.
    File should contain a time column followed by one column of
        beat intervals.
    """
    beats_file = os.path.join(beats_dir, track_id + '.csv')
    t, beat_intervals = utils.read_feature(beats_file, time=True)
    return t, beat_intervals


def get_onsets(track_id):
    """Read ioi data from file onsets_dir + track_id + '.csv'.
    File should contain a time column followed by one column of
        inter-onset intervals.
    """
    onsets_file = os.path.join(onsets_dir, track_id + '.csv')
    t, ioi = utils.read_feature(onsets_file, time=True)
    return t, ioi


# TODO
def get_norm_ioi(track_id):
    pass


if __name__ == '__main__':
    compute_and_write(sys.argv[1], sys.argv[2])