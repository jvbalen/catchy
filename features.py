# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:39:39 2014
"""

__author__ = 'Jan'

import numpy as np
from scipy.sparse import csr_matrix
import scipy.signal as dsp


# base features


def get_chroma(filename):
    chroma_dir = '/Users/Jan/Documents/Work/Cogitch/Goldsmiths/data/hpcp-512-2048/'
    chroma_ext = '_vamp_vamp-hpcp-mtg_MTG-HPCP_HPCP.csv'
    chroma_file = chroma_dir + filename + chroma_ext
    t, chroma = readfeature(chroma_file)
    return t, chroma


def get_melody(filename, unvoiced=True):

    hz2midi = lambda f: 69 + 12 * np.log2(abs(f) / 440)

    melody_dir = '/Users/Jan/Documents/Work/Cogitch/Goldsmiths/data/melodia-128-2048/'
    melody_ext = '_vamp_mtg-melodia_melodia_melody.csv'
    melody_file = melody_dir + filename + melody_ext

    t, x = readfeature(melody_file)
    f0 = x[:, 0]
    if unvoiced:
        f0 = abs(f0)
    else:
        f0[f0 < 0] = 0
    pitched = f0 > 0
    melody = f0
    melody[pitched] = hz2midi(f0[pitched])
    return t, melody


def get_aligned_features(filename):
    tchr, chroma = get_chroma(filename)
    tmel, melody = get_melody(filename)
    tchr, chroma, tmel, melody = alignfeatures(tchr, chroma, tmel, melody)
    return tchr, chroma, tmel, melody


def get_melmat(filename):

    tchr, chroma, t, melody = get_aligned_features(filename)
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


def get_melstm(filename, win=4.0):

    t, melmat = get_melmat(filename)
    dt = t[1] - t[0]
    nkern = np.round(win / dt)
    kern1 = np.zeros((nkern + 1, 1))
    kern2 = np.ones((nkern, 1))
    kern = np.vstack((kern1, kern2))
    kern *= 1.0 / nkern
    melstm = dsp.convolve2d(melmat, kern, mode='same')
    return t, melstm, melmat


def get_melfwd(filename, win=4.0):

    t, melmat = get_melmat(filename)
    dt = t[1] - t[0]
    nkern = np.round(win / dt)
    kern1 = np.ones((nkern, 1))
    kern2 = np.zeros((nkern + 1, 1))
    kern = np.vstack((kern1, kern2))
    kern *= 1.0 / nkern
    melfwd = dsp.convolve2d(melmat, kern, mode='same')
    return t, melfwd, melmat


def get_chromastm(filename, win=4.0):
    t, chroma = get_chroma(filename)
    dt = t[1] - t[0]
    nkern = np.round(win / dt)
    kern1 = np.zeros((nkern + 1, 1))
    kern2 = np.ones((nkern, 1))
    kern = np.vstack((kern1, kern2))
    kern *= 1.0 / nkern
    chromastm = dsp.convolve2d(chroma, kern, mode='same')
    return t, chromastm, chroma


# Summary features


def get_pitchhist(filename, minpitch, maxpitch, bpo=12):
    t, melody = get_melody(filename)

    step = 12.0 / bpo
    halfstep = step / 2.0
    nbins = (maxpitch - minpitch) / step + 1
    binedges = np.linspace(minpitch - halfstep, maxpitch + halfstep, nbins + 1)

    pitchhist = np.histogram(melody, binedges)
    bincenters = (binedges[0:-1] + binedges[1:]) / 2
    return pitchhist[0], bincenters


def get_chromahist(filename, norm=True):
    tchr, chroma, tmel, melody = get_aligned_features(filename)
    chromahist = np.mean(chroma, axis=0)
    if norm:
        chromahist = chromahist * 1.0 / np.sum(chromahist)
    return chromahist


def get_pitchbihist(filename, win=0.5, diagfactor=1, sqrt=False):
    t, melstm, melmat = get_melstm(filename, win=win)
    pitchbihist = co_occurrence([melstm, melmat], mode='dot')
    if diagfactor < 1:
        pitchbihist = scalediag(pitchbihist, diagfactor)
    if sqrt:
        pitchbihist = np.sqrt(pitchbihist)
    return pitchbihist


def get_chromacorr(filename, diagfactor=1, mode='corr', sqrt=False):
    tchr, chroma, tmel, melody = get_aligned_features(filename)
    chromacorr = co_occurrence([chroma], mode=mode)
    if diagfactor < 1:
        chromacorr = scalediag(chromacorr, diagfactor)
    if sqrt:
        chromacorr = np.sqrt(chromacorr)
    return chromacorr


# def getchromabihist(filename, win=0.5, diagfactor=0.5):
#     t, chromastm, chroma = get_chromastm(filename, win=win)
#     chromabihist = co_occurrence([chromastm, chroma], mode='dot')
#     if diagfactor < 1:
#         chromabihist = scalediag(chromabihist, diagfactor)
#     return chromabihist


def get_harmonisation(filename, diagfactor=0):
    tchr, chroma, tmel, melody = get_aligned_features(filename)
    dt = tchr[1] - tchr[0]
    chroma = chroma[2:, :] * dt     # cropping is for exact matlab correspondence
    t, melmat = get_melmat(filename)
    melmat = np.array(melmat)
    melmat = melmat[2:, :]          # cropping is for exact matlab correspondence
    harmonisation = co_occurrence([melmat, chroma], mode='dot')
    if diagfactor < 1:
        harmonisation = scalediag(harmonisation, diagfactor)
    return harmonisation


def get_pitchtrihist(filename, win=0.5, diagfactor=0, norm=True):
    t, melstm, melmat = get_melstm(filename, win)
    t, melfwd, melmat = get_melfwd(filename, win)
    pitchtrihist = co_occurrence([melstm, melmat, melfwd], mode='dot')
    if diagfactor < 1:
        pitchtrihist = scalediag(pitchtrihist, diagfactor)
    if norm:
        pitchtrihist = pitchtrihist * 1.0 / np.sum(pitchtrihist)
    return pitchtrihist


# def get_chromatrihist(filename, win=0.5, diagfactor=0, norm=True):
#     t, melstm, melmat = get_melstm(filename, win)
#     tchr, chroma, tmel, melody = get_aligned_features(filename)
#     chromatrihist = co_occurrence([melstm, melmat, chroma], mode='dot')
#     if diagfactor < 1:
#         chromatrihist = scalediag(chromatrihist, diagfactor)
#     if norm:
#         chromatrihist = chromatrihist * 1.0 / np.sum(chromatrihist)
#     return chromatrihist


def get_chromacorr3(filename, win=0.5, diagfactor=0, mode='corr', norm=True):
    tchr, chroma, tmel, melody = get_aligned_features(filename)
    chromatrihist = co_occurrence([chroma, chroma, chroma], mode=mode)
    if diagfactor < 1:
        chromatrihist = scalediag(chromatrihist, diagfactor)
    if norm:
        chromatrihist = chromatrihist * 1.0 / np.sum(chromatrihist)
    return chromatrihist


# Support functions


def readfilelist(filedir, filelist, listindex, ext):
    import csv
    filelistfile = open(filelist)
    listreader = csv.reader(filelistfile)
    listentries = list(listreader)[listindex]
    return filedir + listentries[0] + ext


def readfeature(filename, mode='pandas'):
    import numpy as np
    import pandas as pd

    if mode == 'numpy':
        data = np.genfromtxt(filename, delimiter=',')
    elif mode == 'pandas':
        data = pd.read_csv(filename, delimiter=',').values
    t = data[:, 0]
    x = data[:, 1:]
    return t, x


def alignfeatures(tx, x, ty, y):
    """ 'Conservative' alignment of time series tx, x and tx, y:
            First, both time series are cropped to the range where they are both defined.
            The time series with highest resolution is then downsampled to obtain the same number of elements for both.
            This is done by retaining the samples closest to those of the other time series.
            For time values of the downsampled time series, the nearest time values of the reference series are used.
    """
    import numpy as np
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


def co_occurrence(mats, mode='dot', verbose=False):
    import numpy as np
    from scipy import einsum
    if len(mats) == 1:
        x = mats[0]
        if verbose:
            print 'type of x in co-occurrence computation: ' + str(type(x))
        if mode == 'dot':
            cont = x.T.dot(x)  # equivalent of x' * y
        elif mode == 'normdot':
            x /= np.linalg.norm(x, axis=0)
            cont = x.T.dot(x)
        elif mode == 'posdot':
            x -= np.mean(x, axis=0)
            x /= np.std(x, axis=0)
            x[x < 0] = 0        # without this line, equivalent to 'corr'
            cont = x.T.dot(x)
        elif mode == 'corr':
            cont = np.corrcoef(x, rowvar=0)
        elif mode == 'sparse':
            x = np.sparse.csr_matrix(x)
            cont = x.T.dot(x)
            cont = cont.todense()
        else:
            raise NameError('mode does not exist')
    if len(mats) == 2:
        x, y = mats[:]
        if mode == 'dot':
            cont = x.T.dot(y)  # equivalent of x' * y
        elif mode == 'sparse':
            x = np.sparse.csr_matrix(x)
            y = np.sparse.csr_matrix(y)
            cont = x.T.dot(y)
            cont = cont.todense()
        else:
            raise NameError('mode does not exist')
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
    return cont


def scalediag(x, diagfactor):
    # TO DO: scale not only z[i,i,i] but all of x[i,i,:] and x[:,i,i]
    # More elegant seeing how the trigrams are computed (from the middle pitch out).
    import numpy as np
    diag = getdiag(x)
    shp = x.shape
    x = np.array(x).flatten()
    x[diag] = diagfactor * x[diag]
    return x.reshape(shp)


def getdiag(x):
    import numpy as np

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