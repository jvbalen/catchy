from __future__ import division, print_function

import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.neighbors as nn

import utils


data_dir = ''


def compute(segment_dict, features):
    """
    Args:
        segment_dict (dict): dictionary of song segments, containing a list of
            segment ids (values) for a set of unique song identifiers (keys).
    """

    data_dict = {}

    # compute features
    for feature in features:
        print('computing ' + feature + '...')
        feature_name, first_order_aggregates, second_order_aggregates = parse_feature(feature)

        corpus_features = []
        for song_id in segment_dict.keys():
            song_features = []
            for segment in segment_dict[song_id]:
                raw_features = utils.read_feature([data_dir, feature_name, segment], skip_cols='auto')
                segment_features = first_order(raw_features, first_order_aggregates, verbose=False)
                song_features.append(segment_features)
            if 'song' in second_order_aggregates:
                song_features = second_order(song_features, second_order_aggregates, verbose=False)
            corpus_features.extend(song_features)
        if 'corpus' in second_order_aggregates:
            # print('        in: len(corpus_features) = {}, corpus_features[0] = {}'.format(len(corpus_features), corpus_features[0]))
            corpus_features = second_order(corpus_features, second_order_aggregates, verbose=False)
        # print('        out: len(corpus_features) = {}, corpus_features[0] = {}'.format(len(corpus_features), corpus_features[0]))
        data_dict[feature] = np.squeeze(corpus_features)
    
    # add segment ids
    song_ids = []
    segments = []
    for song_id in segment_dict.keys():
        for segment in segment_dict[song_id]:
            song_ids.append(song_id)
            segments.append(segment)
    data_dict['song.id'] = np.array(song_ids)
    data_dict['segment.id'] = np.array(segments)
    
    # convert to dataframe            
    return pd.DataFrame(data_dict)


# FEATURE COMPUTATION

def parse_feature(feature):
    """ Parse feature string into
            (feature name, [1st order aggregates], [2nd order aggregates]).

        'Grammar':
        - feature name and aggregates are separated by dots, e.g. 'mfcc.entropy'
        - feature name is first and contains no dots
        - first order and second order aggregates are separated by one of 2 keywords:
            'corpus' or 'song'

        Ex.:
        >>> parse_features('loudness.mean.song.pdf.log')
        ('loudness', ['mean'], ['song', 'pdf', 'log'])
    """
    s = np.array(feature.split('.'))
    split_points = (s == 'corpus') | (s == 'song')
    split_points = np.nonzero(split_points)[0] if any(split_points) else [len(s)]
    return s[0], s[1:split_points[0]].tolist(), s[split_points[-1]:].tolist()


def first_order(feature, aggregates, verbose=False):
    if not type(aggregates) is list:
        aggregates = [aggregates]
    for aggregate in aggregates:
        if verbose:
            print('        first order computation: ' + aggregate)
        if aggregate == 'log':
            feature = np.log(feature)
        elif aggregate == 'sqrt':
            feature = np.sqrt(feature)
        elif aggregate == 'minlog':
            feature = np.log(1 - feature)
        elif aggregate == 'minsqrt':
            feature = np.sqrt(1 - feature)
        elif aggregate == 'mean':
            # feature = np.mean(feature, axis=0)
            feature = np.nanmean(feature, axis=0)
        elif aggregate == 'var':
            feature = np.var(feature, axis=0)
        elif aggregate == 'std':
            # feature = np.std(feature, axis=0)
            feature = np.nanstd(feature, axis=0)
        elif aggregate == 'stdmean':
            feature = np.hstack([np.mean(feature, axis=0), np.std(feature, axis=0)])
        elif aggregate == 'cov':
            feature = np.flatten(np.cov(feature, axis=0))
        elif aggregate == 'totvar':
            feature = np.array([np.mean(np.var(feature, axis=0))])
        elif aggregate == 'totstd':
            feature = np.array([np.mean(np.std(feature, axis=0))])
        elif aggregate == 'entropy':
            feature = feature.flatten()
            feature = np.array([stats.entropy(feature)])
        elif aggregate == 'normentropy':
            feature = feature.flatten()
            feature = np.array([stats.entropy(feature) / np.log(feature.size)])
        elif aggregate == 'information':
            feature = - np.log(feature)

    return feature


def second_order(features, aggregates, verbose=False):
    if not type(aggregates) is list:
        aggregates = [aggregates]

    features = np.asarray(features)
    for aggregate in aggregates:
        if verbose and not (aggregate == 'song' or aggregate == 'corpus'):
            print('        second order computation ({}): {}'.format(aggregates[0], aggregate))
        if aggregate == 'log':
            features = np.log(features)
        elif aggregate == 'sqrt':
            features = np.sqrt(features)
        elif aggregate == 'square':
            features = np.array(features)**2
        elif aggregate == 'minlog':
            features = np.log(1 - np.array(features))
        elif aggregate == 'minsqrt':
            features = np.sqrt(1 - np.array(features))
        elif aggregate == 'logit':
            features = np.log(np.array(features)) - np.log(1 - np.array(features))
            
        elif aggregate == 'kld':
            m = np.sum(features, axis=0)
            m /= np.sum(m)
            features = [stats.entropy(f.flatten(), m.flatten()) for f in features]
        elif aggregate == 'tau':
            m = np.sum(features, axis=0)
            m /= np.sum(m)
            features = [stats.kendalltau(f.flatten(), m.flatten())[0] for f in features]
        elif aggregate == 'dot':
            m = np.sum(features, axis=0)
            features = [np.dot(f.flatten(), m.flatten()) for f in features]
        elif aggregate == 'corr':
            m = np.sum(features, axis=0)
            features = [np.correlate(f.flatten(), m.flatten()) for f in features]
        elif aggregate == 'crossentropy' or aggregate == 'information':
            m = np.sum(features, axis=0)
            m = m.flatten()/np.sum(m)
            features = [-np.nansum(np.log(m) * f.flatten()/np.sum(f)) for f in features]

        elif aggregate == 'pdf':
            n, d = features.shape
            finite_rows = np.all(np.isfinite(features), axis=1)
            features = features[finite_rows]
            s = np.std(features)
            bw_factor = n**(-1./(5))*s if d == 1 and s > 0.0 else 1.0
            kde = nn.KernelDensity(bandwidth=bw_factor)
            kde.fit(features)
            scores = kde.score_samples(features)
            features = np.zeros((n,))
            features[finite_rows] = np.exp(scores)
        elif aggregate == 'indeppdf':
            # above for independent dimensions: fit each dim and add log scores
            kde = nn.KernelDensity(bandwidth=1.0)
            scores = np.zeros(len(features))
            for feat_dim in features.T:
                feat_dim = feat_dim.reshape([-1, 1])
                kde.fit(feat_dim)
                scores += kde.score_samples(feat_dim)
            features = np.exp(scores)

        elif aggregate == 'cdf':
            f0 = np.min(features)
            kde = stats.gaussian_kde(features)
            features = [kde.integrate_box(f0, f) for f in features]
        elif aggregate == 'rank':
            features = (np.argsort(np.argsort(features)) + 0.5) * (1.0 / len(features))

    # features = [np.squeeze(f) for f in features]
    return features
