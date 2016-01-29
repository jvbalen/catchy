# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


tracklist_file = '/Users/Jan/Documents/Work/Cogitch/Audio/HOOKEDNL/tracklist.csv'
tracklist = pd.read_csv(tracklist_file)


def get_tracklist():
    return tracklist


def inseconds(time):
    parts = time.split(':')
    if len(parts) > 3 or len(time) == 0:
        raise Exception('Error: too many colons.')
    if len(time) == 0:
        raise Exception('Error: no numbers to parse.')
    return sum([60**i * float(parts[-1-i]) for i in range(len(parts))])


def has_audiofile(uri):
    matches = tracklist['filename'][tracklist['uri'] == uri].values
    return matches[0] is not np.nan


def get_song_data():

    dr_file = 'data/driftrates/drift-rates.csv'
    dr_data = pd.read_csv(dr_file)

    songs = []
    for i, uri in enumerate(dr_data['track.uri'].unique()):
        seg_ids = dr_data['segment.position'][dr_data['track.uri'] == uri]  #.values
        seg_dr = dr_data['correct'][dr_data['track.uri'] == uri]  #.values
        nfiles = len(seg_ids) if has_audiofile(uri) else 0
        seg_files = [str(i+1) + '-' + str(j+1) for j in range(nfiles)]
        song = {
            'track.uri': uri,
            'segment.ids': seg_ids,
            'segment.t': [inseconds(seg) for seg in seg_ids],
            'segment.dr': seg_dr,
            'segment.files': seg_files}
        songs.append(song)

    n_segs = len(songs)
    song_data = pd.DataFrame(songs)

    db_data = pd.read_csv('data/sections.csv')
    db_data['sec_list'] = [sections.split('|')[:-1] for sections in db_data['Sections']]
    uris = [track_uri.split(':')[-1] for track_uri in song_data['track.uri']]

    song_data['segment.total'] = [len(db_data['sec_list'][db_data['URI'] == uri].values[0]) for uri in uris]
    song_data['segment.annotated'] = [len(song_data['segment.files'][i]) for i in range(n_segs)]
    song_data['segment.notannotated'] = np.array(song_data['segment.total']) - np.array(song_data['segment.annotated'])

    return song_data


def get_small_dataset():
    song_data = get_song_data()
    return song_data[np.asarray(song_data['segment.notannotated']) == 2]


def get_big_dataset(min_annotated=2):
    song_data = get_song_data()
    return song_data[song_data['segment.annotated'] > min_annotated]