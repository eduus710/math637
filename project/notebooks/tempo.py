#
# dirty script to generate beat/tempo statistics for every clips_tids
#
# walks an FMA clip directory and creates a dataframe of 
# tempo stats - dataframe is compatible for merge with 
# FMA features dataset
# 
# dataframe is exported to CSV
#
# NOTE: a couple of MP3s are garbage, or I don't have the right 
# encoder - there will be a few NAs in the dataframe that need
# to be handled when merging with tracks or features dataframes
#
# NOTE: 3 hours on my desktop for all clips (FMA large)

import os
from pathlib import Path

import multiprocessing
from tqdm import tqdm

import pandas as pd
import numpy as np
import scipy.stats as stats
import librosa

import warnings
warnings.filterwarnings("ignore")

import utils

# set this to appropriate FMA clip directory

AUDIO_DIR=Path('./fma/data/fma_large')

def get_fs_tids(audio_dir):
    tids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            tids.extend(int(file[:-4]) for file in files)
    return tids

def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

# helper to generate Pandas multi-index compatible with Features
tempo_columns = [('tempo','mean', '01'),
                 ('tempo','std', '01'),
                 ('tempo','skew', '01'),
                 ('tempo','kurtosis', '01'),
                 ('tempo','median', '01'),
                 ('tempo','min', '01'),
                 ('tempo','max', '01'),
                ]
def columns():
    return pd.MultiIndex.from_tuples(tempo_columns, 
                                     names=('feature', 'statistics', 'number'))

# compute tempo stats for 1 track    
# return a Pandas series (will become a row in output dataset)
def compute_features(tid):

    features = pd.Series(index=columns(), dtype=np.float32, name=tid)
    
    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values)
        features[name, 'std'] = np.std(values)
        features[name, 'skew'] = stats.skew(values)
        features[name, 'kurtosis'] = stats.kurtosis(values)
        features[name, 'median'] = np.median(values)
        features[name, 'min'] = np.min(values)
        features[name, 'max'] = np.max(values)

    try:
        filepath = get_audio_path(AUDIO_DIR, tid)
        y, sr = librosa.load(filepath, sr=None, mono=True) 
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        f = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        feature_stats('tempo', f)
    
    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))
        
    return features
    

if __name__ == "__main__":

    # load Features to get track indexes to build dataframe
    (features, tracks) = utils.load_features()
    features.info()

    # output dataframe
    tempo = pd.DataFrame(index=tracks.index,
                         columns=columns(), 
                         dtype=np.float32)

    # walk directories for clip ids
    # pool processing for speed
    clips_tids = get_fs_tids(AUDIO_DIR)      
    pool = multiprocessing.Pool()
    it = pool.imap_unordered(compute_features, clips_tids)

    for i, row in enumerate(tqdm(it, total=len(clips_tids))):
        tempo.loc[row.name] = row
    
    tempo.to_csv('tempo.csv', float_format='%.{}e'.format(9))
    


