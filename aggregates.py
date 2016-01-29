# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.neighbors as nn
import csv

import load as ld

# FEATURE COMPUTATION


def skip_columns(feature):
    """pythonic switch case to set column ignore behavior in csv files."""
    firstcol = {'loudness': 1, 'sharpness': 1, 'bands': 1, 'pitch': 1, 'hpcp': 1,
                'mfcc': 2}.get(feature, 0)
    lastcol = {'mfcc': 14}.get(feature, None)
    return firstcol, lastcol


def parse_feature(feature):
    """ Parse feature string into
            (feature name, [1st order aggregates], [2nd order aggregates]).

        'Grammar':
        - feature name and aggregates are separated by dots, e.g. 'mfcc.entropy'
        - feature name is first and contains no dots
        - first order and second order aggregates are separated by one of 2 keywords:
            'corpus' or 'song'

        feature name should point to a file named segmentid_featurename.csv,
            e.g. 10-1_hpcp.csv

        Ex.:
        >>> parse_features('loudness.mean.song.pdf.log')
        ('loudness', ['mean'], ['song', 'pdf', 'log'])
    """
    s = np.array(feature.split('.'))
    split_points = (s == 'corpus') | (s == 'song')
    split_points = np.nonzero(split_points)[0] if any(split_points) else [len(s)]
    return s[0], s[1:split_points[0]].tolist(), s[split_points[-1]:].tolist()


def read_feature(segment, ft_name, base_dir = '/Users/Jan/Documents/Work/Cogitch/Goldsmiths/data/all_features/'):
    filename = base_dir + segment + '_' + ft_name + '.csv'
    with open(filename) as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        feature = np.asarray(list(reader))
    firstcol, lastcol = skip_columns(ft_name)
    feature = feature[:, firstcol:lastcol]
    return feature


def first_order(feature, aggregates, verbose=False):
    if not type(aggregates) is list:
        aggregates = [aggregates]
    for aggregate in aggregates:
        if verbose:
            print '        first order computation: ' + aggregate
        if aggregate == 'log':
            feature = np.log(feature)
        elif aggregate == 'sqrt':
            feature = np.sqrt(feature)
        elif aggregate == 'minlog':
            feature = np.log(1 - feature)
        elif aggregate == 'minsqrt':
            feature = np.sqrt(1 - feature)
        elif aggregate == 'mean':
            feature = np.mean(feature, axis=0)
        elif aggregate == 'var':
            feature = np.var(feature, axis=0)
        elif aggregate == 'std':
            feature = np.std(feature, axis=0)
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
        if verbose:
            print '        second order computation: ' + aggregate
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
        elif aggregate == 'crossentropy':
            m = np.sum(features, axis=0)
            m = m.flatten()/np.sum(m)
            features = [-np.dot(f.flatten()/np.sum(f), np.log(m)) for f in features]

        elif aggregate == 'pdf':
            n, d = features.shape
            bw_factor = n**(-1./(4+d)) * np.std(features) if d == 1 else 1.0
            kde = nn.KernelDensity(bandwidth=bw_factor)
            kde.fit(features)
            scores = kde.score_samples(features)
            features = np.exp(scores)  # score = log density, therefore return exp(score)
            # print np.mean(features), np.std(features)
            # print np.min(features), np.median(features), np.max(features)
        elif aggregate == 'normpdf':
            n, d = features.shape
            bw_factor = n**(-1.0/(4+d)) * np.std(features) if d == 1 else 1.0
            kde = nn.KernelDensity(bandwidth=bw_factor)
            kde.fit(features)
            sample = kde.sample(n)
            scores = kde.score_samples(features)
            features = np.exp(scores)  # score = log density, therefore return exp(score)
            max_features = np.max(features)
            max_sample = np.max(np.exp(kde.score_samples(sample)))
            # print '        ' + str(max_features) + str(max_sample)
            features = features / np.max([max_sample, max_features])
        # elif aggregate == 'dimpdf':
        #     # performs badly:
        #     # - pca should probably be computed wrt larger corpus.
        #     # - results for both bw_factor = 1 and scott's bw n^(-1/(4+d))
        #     #   seem to depend only on number of number of sections in song..
        #     n, d = features.shape
        #     bw_factor = n**(-1./4)  # scott's factor for KDE bandwidths
        #     for i in range(d):
        #         kde = nn.KernelDensity(bandwidth=bw_factor)
        #         kde.fit(features[:, i])
        #         features[:, i] = np.exp(kde.score_samples(features[:, i]))
        #     features = np.sum(features, axis=0)
        elif aggregate == 'meandist':
            import scipy.spatial.distance as dist
            # m = [np.mean(features, axis=0)]
            dist_matrix = dist.squareform(dist.pdist(features))
            # spread_matrix = dist.cdist(features, m)
            features = np.mean(dist_matrix, axis=1)  # / np.mean(spread_matrix)

        elif aggregate == 'cdf':
            f0 = np.min(features)
            kde = stats.gaussian_kde(features)
            features = [kde.integrate_box(f0, f) for f in features]
        elif aggregate == 'rank':
            features = np.argsort(features) * (1.0 / len(features))

        elif aggregate == 'pca3':
            import sklearn.decomposition as deco
            pca = deco.PCA(3)
            features = (features - np.mean(features, 0)) / np.std(features, 0)
            features = pca.fit(features).transform(features)
    # features = [np.squeeze(f) for f in features]
    return features


def get(song_data, features):
    data_dict = {}
    for feature in features:
        print '    ' + feature
        feature_name, first_order_aggregates, second_order_aggregates = parse_feature(feature)

        corpus_features = []
        for song in song_data['segment.files']:
            song_features = []
            for segment in song:
                raw_features = read_feature(segment, feature_name)
                segment_features = first_order(raw_features, first_order_aggregates, verbose=False)
                song_features.append(segment_features)
            if 'song' in second_order_aggregates:
                song_features = second_order(song_features, second_order_aggregates, verbose=False)
            corpus_features.extend(song_features)
        if 'corpus' in second_order_aggregates:
            # print '        in: len(corpus_features) = {}, corpus_features[0] = {}'.format(len(corpus_features), corpus_features[0])
            corpus_features = second_order(corpus_features, second_order_aggregates, verbose=True)
        print '        out: len(corpus_features) = {}, corpus_features[0] = {}'.format(len(corpus_features), corpus_features[0])
        data_dict[feature] = np.squeeze(corpus_features)
    
    segment_id = []
    segment_dr = []
    segment_annotated = []
    segment_notannotated = []
    for i, song in enumerate(song_data['segment.files']):
            for j, segment in enumerate(song):
                segment_id.append(segment)
                segment_dr.append(song_data['segment.dr'].values[i].values[j])
                segment_annotated.append(song_data['segment.annotated'].values[i])
                segment_notannotated.append(song_data['segment.notannotated'].values[i])
    data_dict['segment.id'] = np.array(segment_id)
    data_dict['segment.dr'] = np.array(segment_dr)
    data_dict['segment.annotated'] = np.array(segment_annotated)
    data_dict['segment.notannotated'] = np.array(segment_notannotated)
                
    return pd.DataFrame(data_dict)


def write_demo(filename='data/aggregates'):

    song_data = ld.get_big_dataset()

    features = ['loudness.mean',
                'loudness.mean.song.pdf.log',
                'loudness.mean.corpus.normpdf.minsqrt',

                'loudness.std',
                'loudness.std.song.pdf.log',
                'loudness.std.corpus.normpdf.minsqrt',

                'sharpness.mean',
                'sharpness.mean.song.pdf.log',
                'sharpness.mean.corpus.normpdf.minsqrt',

                'roughness.mean.log',
                'roughness.mean.log.song.pdf.log',
                'roughness.mean.log.corpus.normpdf.minsqrt',

                'mfcc.totvar.log',
                'mfcc.totvar.log.song.pdf.log',
                'mfcc.totvar.log.corpus.normpdf.minsqrt',

                'mfcc.mean.corpus.pca3.normpdf',
                'mfcc.mean.song.meandist.log',

                'pitch.mean',
                'pitch.mean.song.pdf.log',
                'pitch.mean.corpus.normpdf.minsqrt',

                'pitch.std.log',
                'pitch.std.log.song.pdf.log',
                'pitch.std.log.corpus.normpdf.minsqrt',


                'harmony3.normentropy.minlog',
                'harmony3.normentropy.minlog.song.pdf.log',
                'harmony3.normentropy.minlog.corpus.normpdf.minsqrt',

                'harmony-interval2.song.tau',
                'harmony-interval2.corpus.tau',

                'harmony-interval1.mean.corpus.pca3.normpdf',
                'harmony-interval1.mean.song.meandist.log',

                'harmonisation.normentropy.minlog',
                'harmonisation.normentropy.minlog.song.pdf.log',
                'harmonisation.normentropy.minlog.corpus.normpdf.minsqrt',


                'melody-interval1.mean.corpus.pca3.normpdf',
                'melody-interval1.mean.song.meandist.log',

                'melody3.normentropy.minlog',
                'melody3.normentropy.minlog.song.pdf.log',
                'melody3.normentropy.minlog.corpus.normpdf.minsqrt',

                'melody-interval2.song.tau',
                'melody-interval2.corpus.tau',
                ]

    print 'Computing dataframe for ' + str(len(features)) + ' features...'
    all_data = get(song_data, features)
    all_data['song.id'] = [seg_id.split('-')[0] for seg_id in all_data['segment.id']]
    print 'Done.'

    big_data = all_data
    small_data = all_data[all_data['segment.notannotated']==2]

    filename = filename.split('.')[0]
    big_data.to_csv(filename + '_big.csv')
    small_data.to_csv(filename + '_small.csv')


def test_kde():
    song_data = ld.get_big_dataset()
    features = ['loudness.mean.corpus.pdf', 'mfcc.mean.corpus.pdf']
    # features = ['mfcc.mean.corpus.pdf', 'harmony-interval1.mean.corpus.normpdf', 'melody-interval1.mean.corpus.normpdf']
    print get(song_data, features).head()


if __name__ == '__main__':
    test_kde()