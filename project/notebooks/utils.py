# utility code for MATH637 project

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix, plot_confusion_matrix, f1_score)

def load_features():
    features = pd.read_csv('../fma/features.csv', index_col=0, header=[0,1,2])
    tracks = pd.read_csv('../fma/tracks.csv', index_col=0, header=[0,1])
    np.testing.assert_array_equal(features.index, tracks.index)
    return(features, tracks)

def load_genres():
    genres =  pd.read_csv('../fma/genres.csv', index_col=0, header=[0])
    return genres

def load_tempo(set):
    tempo =  pd.read_csv(f'./tempo_{set}.csv', index_col=0, header=[0,1,2])
    return tempo

def stratify_genres(tracks, features, genres, n=100):
    subset = tracks[('track','genre_top')].isin(genres)
    tdf = tracks[subset]
    fdf = features[subset]
    tdf = tdf.groupby(('track','genre_top'), group_keys=False).apply(lambda x: x.sample(n))
    fdf = fdf[fdf.index.isin(tdf.index)]
    tdf.sort_index(inplace=True)
    fdf.sort_index(inplace=True)
    return(tdf, fdf)

def run_estimator(df, Y, estimator, ax, scale=True):
    if scale:
        X = MinMaxScaler().fit_transform(df)
    else:
        X = df
    X_est = estimator.fit_transform(X)
    ax = sns.scatterplot( x = X_est[:,0], y = X_est[:,1], hue=Y, ax=ax)
    
def estimate_features(df, Y, feature_sets, title, estimator, scale=True):   
    nrows = (len(feature_sets)+1)//2
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(17,5*nrows))

    for (set,ax) in zip(feature_sets,axs.flat):
        run_estimator(df[set], 
                      Y,
                      estimator,
                      ax,
                     scale)
        ax.set_title(f'{title} - {set}')

    if nrows*2 > len(feature_sets):
        if nrows > 1:
            fig.delaxes(axs[nrows-1,1])
        else:
            fig.delaxes(axs[1])
    plt.tight_layout()
    
def estimate_all_features(df, Y, title, estimator, scale=True):   
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(17,5))

    run_estimator(df, 
                  Y,
                  estimator,
                  axs[0],
                 scale)
    axs[0].set_title(f'{title} - all features')

    fig.delaxes(axs[1])
    plt.tight_layout()

def track_genre_tops(track_genres_string, genres):
    track_genres = [int(x) for x in track_genres_string[1:-1].split(',')]
    return track_genres

# predict with a classifier and generate reports
def run_classifier_and_report(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_confusion_matrix(classifier, X_test, y_test,ax=ax,xticks_rotation='vertical',values_format='0.2f', normalize='true')
