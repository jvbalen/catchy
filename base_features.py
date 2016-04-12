from __future__ import division, print_function

import os
import numpy as np
import pandas as pd
import sys

""" This module provides an interface to several existing audio feature time
    series extractors.

    Requires Librosa to be installed, and optional Vamp plug-ins.
"""

def write_features(audio_dir, data_dir, features=None):
    """Compute frame-based features for all audio files in a folder.

    Args:
        audio_dir (str): where to find audio files
        data_dir (str): where to write features
        features (dict): dictionary with feature extraction functions, indexed
            by feature name.
            Feature extraction functions should return a time 1d-array of 
            frame times and a 2d-array of feature frames.
            Feature name will be used as the subdirectory to
            which feature CSVs are written.)
        """
    import librosa
    import vamp
    
    if features is None:
        features = {'mfcc': get_mfcc, 'hpcp': get_hpcp, 'melody': get_melody}

    filenames = os.listdir(audio_dir)
    for filename in filenames:

        if filename.endswith('.wav') or filename.endswith('.mp3'):
            print("Computing features for file {}...".format(filename))

            x, sr = librosa.load(os.path.join(audio_dir, filename), mono=True)

            for feature in features:

                func = features[feature]
                t, X = func(x, sr)

                track_id = filename.split('.')[-2]
                write_csv(t, X, [data_dir, feature, track_id])


def get_mfcc(x, sr, n_mfcc=20):
    """Compute MFCC features from raw audio, using librosa.
    Librosa must be installed.
    
    Args:
        x (1d-array) audio signal, mono
        sr (int): sample rate
        n_mfcc (int): number of coefficients to retain

    Returns:
        2d-array: MFCC features
    """
    mfcc_all = librosa.feature.mfcc(x, sr)
    n_coeff, n_frames = mfcc_all.shape
    t = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=512)

    return t, mfcc_all[:n_mfcc].T


def get_hpcp(x, sr, n_bins=12, f_min=55, f_ref=440.0, min_magn=-100):
    """Compute HPCP features from raw audio using the HPCP Vamp plugin.
    Vamp, vamp python module and plug-in must be installed.
    
    Args:
        x (1d-array): audio signal, mono
        sr (int): sample rate
        n_bins (int): number of chroma bins
        f_min (float): minimum frequency
        f_ref (float): A4 tuning frequency
        min_magn (float): minimum magnitude for peak detection, in dB
        
    Returns:
        1d-array: time vector
        2d-array: HPCP features
    """

    plugin = 'vamp-hpcp-mtg:MTG-HPCP'
    params = {'LF': f_min, 'nbins': n_bins, 'reff0': f_ref,
              'peakMagThreshold': min_magn}
    
    data = vamp.collect(x, sr, plugin, parameters=params)
    vamp_hop, hpcp = data['matrix']
    
    t = float(vamp_hop) * (8 + np.arange(len(hpcp)))
    
    return t, hpcp


def get_melody(x, sr, f_min=55, f_max=1760, min_salience=0.0):
    """Extract main melody from raw audio using the Melodia Vamp plugin.
    Vamp, vamp python module and plug-in must be installed.
    
    Args:
        x (np.array): audio signal, mono
        sr (int): sample rate
        f_min (float): minimum frequency
        f_max (float): maximum frequency
        
    Return:
        1d-array: time vector
        1d-array: main melody (in cents)
    """
    plugin = 'mtg-melodia:melodia'
    params = {'minfqr': f_min, 'maxfqr': f_max,
              'minpeaksalience': min_salience}
    
    data = vamp.collect(x, sr, plugin, parameters=params)
    vamp_hop, melody = data['vector']
    
    melody_cents = 1200*np.log2(melody/55.0)
    melody_cents[melody<=0] = None
    melody_cents = melody_cents.reshape((-1,1))
    
    t = float(vamp_hop) * (8 + np.arange(len(melody)))
    
    return t, melody_cents


def write_csv(t, X, filename):
    """Write frame-based features to CSV.

    Args:
        t (1d-array): frame times
        X (2d-array): feature matrix (frames are rows)
        filename (list or str): file name. If list, will use
            os.path.join to join dirs and filename. CSV extension will be
            added if not already included.
    """
    if type(filename) is list:
        filename = os.path.join(*filename)
    if not (filename.endswith('.csv') or filename.endswith('.txt')):
        filename += '.csv'
    
    data = pd.DataFrame(np.hstack([t.reshape((-1,1)), X]))
    data.to_csv(filename, header=False, index=False)


if __name__ == '__main__':
    write_features(sys.argv[1], sys.argv[2])
