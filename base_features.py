from __future__ import division, print_function

import os
import numpy as np
import pandas as pd
import sys

import librosa
import vamp

import utils

""" This module provides an interface to several existing audio feature time
    series extractors.

    Requires Librosa to be installed, and optional Vamp plug-ins.
"""

def compute_and_write(audio_dir, data_dir, features=None):
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
                utils.write_feature([t, X], [data_dir, feature, track_id])


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


def get_melody(x, sr, f_min=55, f_max=1760, min_salience=0.0, unvoiced=True):
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
    vamp_hop, f0 = data['vector']
    
    if unvoiced:
        f0 = abs(f0)
        f0[f0 == 0] = None
    else:
        f0[f0 <= 0] = None

    hz2midi = lambda f: 69 + 12 * np.log2(abs(f) / 440)
    
    melody = hz2midi(f0)
    melody = melody[:, np.newaxis]
    
    t = float(vamp_hop) * (8 + np.arange(len(melody)))
    
    return t, melody


if __name__ == '__main__':
    compute_and_write(sys.argv[1], sys.argv[2])
