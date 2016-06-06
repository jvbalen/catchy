
from __future__ import division, print_function

import os
import numpy as np
from scipy import einsum
from scipy.sparse import csr_matrix
import scipy.signal as dsp


import utils

melody_dir = ''
chroma_dir = ''


def compute_and_write(data_dir, track_list=None, features=None):
    """Compute frame-based features for all audio files in a folder.

    Args:
        data_dir (str): where to write features
        track_list (str or None): list of file ids. Set to None to infer from
            files in melody_dir and chroma_dir (the intersection is used).
        features (dict): dictionary with (unique) feature names as keys and 
            tuples as values, each containing a feature extraction function and a
            parameter dictionary.
            Feature extraction functions can be any function that returns one
                or more 1d or 2d-arrays that share their first dimension.

    Required global variables:
        melody_dir (str): where to find melody data
        chroma_dir (str): where to find chroma data
    """
    
    if track_list is None:
        melody_ids = [filename.split('.')[0] for filename in os.listdir(melody_dir)]
        chroma_ids = [filename.split('.')[0] for filename in os.listdir(chroma_dir)]

        track_list = list(set(melody_ids + chroma_ids))
    
    if features is None:
        features = {'pitchhist': (get_pitchhist, {}),
                    'pitchhist2': (get_pitchhist2, {}),
                    'pitchhist3': (get_pitchhist3, {}),
                    'pitchhist3_int': (get_pitchhist3, {'intervals': True, 'diagfactor': 1, 'sqrt': False}),
                    'chromahist2': (get_chromahist2, {}),
                    'chromahist3': (get_chromahist3, {}),
                    'chromahist3_int': (get_chromahist3, {'intervals': True}),
                    'harmonisation': (get_harmonisation, {}),
                    'harmonisation_int': (get_harmonisation, {'intervals': True}) }

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


def set_feature_dirs(new_melody_dir, new_chroma_dir):
    if not new_melody_dir is None:
        global melody_dir
        melody_dir = new_melody_dir
    if not new_chroma_dir is None:
        global chroma_dir
        chroma_dir = new_chroma_dir


## -----------------------  features


def get_pitchhist(track_id, minpitch=33, maxpitch=93, bpo=12):
    t, melody = get_melody(track_id)

    step = 12.0 / bpo
    halfstep = step / 2.0
    nbins = (maxpitch - minpitch) / step + 1
    binedges = np.linspace(minpitch - halfstep, maxpitch + halfstep, nbins + 1)

    pitchhist, _ = np.histogram(melody, binedges)
    pitchhist = pitchhist / np.sum(pitchhist)
    bincenters = (binedges[0:-1] + binedges[1:]) / 2
    return pitchhist  # bincenters


def get_pitchhist2(track_id, win=0.5, diagfactor=1, norm=True, sqrt=False, intervals=False):
    t, melstm, melmat = two_melody_matrices(track_id, win=win)
    pitchbihist = co_occurrence([melstm, melmat], mode='dot')
    if diagfactor < 1:
        pitchbihist = scalediag(pitchbihist, diagfactor)
    if norm:
        pitchbihist = pitchbihist * 1.0 / np.sum(pitchbihist)
    if sqrt:
        pitchbihist = np.sqrt(pitchbihist)
    if intervals:
        pitchbihist = to_intervals(pitchbihist)
    return pitchbihist


def get_pitchhist3(track_id, win=0.5, diagfactor=1, norm=True, sqrt=False, intervals=False):
    t, melstm, melmat, melfwd = three_melody_matrices(track_id, win)
    pitchtrihist = co_occurrence([melstm, melmat, melfwd], mode='dot')
    if diagfactor < 1:
        pitchtrihist = scalediag(pitchtrihist, diagfactor)
    if norm:
        pitchtrihist = pitchtrihist * 1.0 / np.sum(pitchtrihist)
    if sqrt:
        pitchtrihist = np.sqrt(pitchtrihist)
    if intervals:
        pitchtrihist = to_intervals(pitchtrihist) 
    return pitchtrihist


def get_chromahist(track_id, norm=True):
    tchr, chroma, tmel, melody = aligned_pitch_features(track_id)
    chromahist = np.mean(chroma, axis=0)
    if norm:
        chromahist = chromahist * 1.0 / np.sum(chromahist)
    return chromahist


def get_chromahist2(track_id, mode='dot', diagfactor=1, norm=True, sqrt=False, intervals=False):
    tchr, chroma, tmel, melody = aligned_pitch_features(track_id)
    chromacorr = co_occurrence([chroma], mode=mode)
    if diagfactor < 1:
        chromacorr = scalediag(chromacorr, diagfactor)
    if norm:
        chromacorr = chromacorr * 1.0 / np.sum(chromacorr)
    if sqrt:
        chromacorr = np.sqrt(chromacorr)
    if intervals:
        chromacorr = to_intervals(chromacorr)
    return chromacorr


def get_chromahist3(track_id, mode='dot', diagfactor=1, norm=True, sqrt=False, intervals=False):
    tchr, chroma, tmel, melody = aligned_pitch_features(track_id)
    chromatrihist = co_occurrence([chroma, chroma, chroma], mode=mode)
    if diagfactor < 1:
        chromatrihist = scalediag(chromatrihist, diagfactor)
    if norm:
        chromatrihist = chromatrihist * 1.0 / np.sum(chromatrihist)
    if sqrt:
        chromatrihist = np.sqrt(chromatrihist)
    if intervals:
        chromatrihist = to_intervals(chromatrihist)
    return chromatrihist


def get_harmonisation(track_id, mode='dot', diagfactor=1, norm=True, sqrt=False, intervals=False):
    tchr, chroma, tmel, melody = aligned_pitch_features(track_id)
    dt = tchr[1] - tchr[0]
    chroma = chroma[2:, :] * dt     # cropping is for exact matlab correspondence
    t, melmat = one_melody_matrix(track_id)
    melmat = np.array(melmat)
    melmat = melmat[2:, :]          # cropping is for exact matlab correspondence
    harmonisation = co_occurrence([melmat, chroma], mode=mode)
    if diagfactor < 1:
        harmonisation = scalediag(harmonisation, diagfactor)
    if norm:
        harmonisation = harmonisation * 1.0 / np.sum(harmonisation)
    if sqrt:
        harmonisation = np.sqrt(harmonisation)
    if intervals:
        harmonisation = to_intervals(harmonisation)
    return harmonisation


## -------------------------------- base features


def get_chroma(track_id):
    """Read chroma data from file chroma_dir + track_id + '.csv'.
    File should contain a time column followed by one column per chroma
        dimension.
    """
    chroma_file = os.path.join(chroma_dir, track_id + '.csv')
    t, chroma = utils.read_feature(chroma_file, time=True)
    return t, chroma


def get_melody(track_id):
    """Read melody data from file melody_dir + track_id + '.csv'.
    File should contain melody data in two columns: (time, melody)
        with melody in midi note number (float or int).
        Frames in which no pitch is present can be set to 0, None or np.nan.
    """
    melody_file = os.path.join(melody_dir, track_id + '.csv')
    t, melody = utils.read_feature(melody_file, time=True)
    return t, melody.flatten()


## -------------------------------- time series handling


def three_melody_matrices(track_id, win=4.0):

    t, melstm, melmat = two_melody_matrices(track_id)
    dt = t[1] - t[0]
    nkern = np.round(win / dt)
    kern1 = np.ones((nkern, 1))
    kern2 = np.zeros((nkern + 1, 1))
    kern = np.vstack((kern1, kern2))
    kern *= 1.0 / nkern
    melfwd = dsp.convolve2d(melmat, kern, mode='same')
    return t, melstm, melmat, melfwd


def two_melody_matrices(track_id, win=4.0):

    t, melmat = one_melody_matrix(track_id)
    dt = t[1] - t[0]
    nkern = np.round(win / dt)
    kern1 = np.zeros((nkern + 1, 1))
    kern2 = np.ones((nkern, 1))
    kern = np.vstack((kern1, kern2))
    kern *= 1.0 / nkern
    melstm = dsp.convolve2d(melmat, kern, mode='same')
    return t, melstm, melmat


def one_melody_matrix(track_id):

    tchr, chroma, t, melody = aligned_pitch_features(track_id)
    melody = np.round(melody)
    pitched = melody > 0
    pitchclass = np.remainder(melody - 69, 12)
    framerate = 1.0/(t[1]-t[0])

    nmel = len(melody)

    vals = np.ones(nmel)[pitched]
    vals *= 1.0 / framerate
    rows = np.arange(nmel)[pitched]
    cols = pitchclass[pitched]
    melmat = csr_matrix((vals, (rows, cols)), shape=(nmel, 12))
    return t, melmat.todense()


def aligned_pitch_features(track_id):
    tchr, chroma = get_chroma(track_id)
    tmel, melody = get_melody(track_id)
    tchr, chroma, tmel, melody = align_features(tchr, chroma, tmel, melody)
    return tchr, chroma, tmel, melody


def to_intervals(X):

    def _roll_rows(x):
        """ Circularly shift ('roll') rows i in array by -i, recursively.
        If 2d-array: circularly shift each row i to the left, i times so that
            X(i, j-i) = X(i, j)
        If 3d-array (or 4d, 5d..):
            X(i, j-i, k-j) = X(i, j, k)
        """
        if len(x.shape) > 2:
            x = np.array([_roll_rows(xi) for xi in x])
        elif len(x.shape) == 1:
            raise ValueError('Method requires nd-array with n >= 2.')
        x_rolled = np.array([np.roll(xi, -i, axis=0) for i, xi in enumerate(x)])
        return x_rolled

    X_rolled = _roll_rows(X)

    X_inv = np.sum(X_rolled, axis=0)

    return X_inv


## ------------------------- feature alignment


def align_features(tx, x, ty, y):
    """ 'Conservative' alignment of time series tx, x and tx, y:
        First, both time series are cropped to the range where they are both
        defined. The time series with highest resolution is then downsampled
        to obtain the same number of elements for both. This is done by
        retaining the samples closest to those of the other time series.
        For time values of the downsampled time series, the nearest time
        values of the reference series are used.
    """
    dtx = tx[1]-tx[0]
    dty = ty[1]-ty[0]
    xbegins = x[0] <= y[0]
    xends = x[-1] >= y[-1]
    if dtx >= dty:
        # find subset of indices of y closest to x
        iresample = np.round((tx-ty[0])/dty).astype(int)
        icrop = (0 <= iresample) * (iresample < y.size)
        y = y[iresample[icrop]]
        x = x[icrop]
        tx = tx[icrop]
        ty = tx
    elif dty > dtx:
        # find subset of indices of x closest to y
        iresample = np.round((ty-tx[0])/dtx).astype(int)
        icrop = (0 <= iresample) * (iresample < x.size)
        x = x[iresample[icrop]]
        y = y[icrop]
        ty = ty[icrop]
        tx = ty
    return tx, x, ty, y


## ------------------ co-occurrence


def co_occurrence(mats, mode='dot', verbose=False):
    import numpy as np
    
    # feature co_occurrence within one multi-dimensional features
    if len(mats) == 1:
        x = mats[0]
        if mode == 'dot':
            cont = x.T.dot(x)  # equivalent of x' * y
        elif mode == 'corr':
            cont = np.corrcoef(x, rowvar=0)
        elif mode == 'sparse':
            x = np.sparse.csr_matrix(x)
            cont = x.T.dot(x)
            cont = cont.todense()
        else:
            raise NameError('mode does not exist')

    # feature co_occurrence across two multidimensional features
    if len(mats) == 2:
        x, y = mats[:]
        if mode == 'dot':
            cont = x.T.dot(y)
        elif mode == 'corr':
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)
            cont = x.T.dot(y)
        elif mode == 'sparse':
            x = np.sparse.csr_matrix(x)
            y = np.sparse.csr_matrix(y)
            cont = x.T.dot(y)
            cont = cont.todense()
        else:
            raise NameError('mode does not exist')

    # feature co_occurrence across three multidimensional features
    if len(mats) == 3:
        x, y, z = mats[:]
        if mode == 'dot':
            cont = einsum('ni,nj,nk', x, y, z)
        elif mode == 'corr':
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)
            z = (z - np.mean(z, axis=0)) / np.std(z, axis=0)
            cont = einsum('ni,nj,nk', x, y, z)
        else:
            raise NameError('mode does not exist')

    return np.array(cont)  # we don't want returns of type matrix


def scalediag(x, diagfactor):
    # Potential extension: scale not only z[i,i,i] but all of x[i,i,:] and x[:,i,i]
    # More elegant seeing how the trigrams are computed (from the middle pitch out).

    diag = getdiag(x)
    shp = x.shape
    x = np.array(x).flatten()
    x[diag] = diagfactor * x[diag]
    return x.reshape(shp)


def getdiag(x):

    shp = x.shape
    ndim = len(shp)
    diag = np.zeros(shp)
    if ndim == 1:
        diag = 0
    elif ndim == 2:
        i = np.arange(min(shp))
        ii = (i, i)
        diag = np.ravel_multi_index(ii, shp)
    elif ndim == 3:
        i = np.arange(min(shp))
        iii = (i, i, i)
        diag = np.ravel_multi_index(iii, shp)
    return diag


## ------------------- Main


if __name__ == '__main__':
    compute_and_write(sys.argv[1], sys.argv[2])