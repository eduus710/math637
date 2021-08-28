import pandas as pd
from collections import Counter

def top_genres(genres_string, genres):
    ''' Extract unique root genres from a list of genres
    '''    
    # check for empty genre string
    if not genres_string.strip('][').strip():
        return(None, None, 0, None)

    # track genres as list
    gl = [int(g) for g in genres_string.strip('][').split(',')]
        
    # track genres to list of top genres
    tl = [genres.loc[g]['top_level'] for g in gl]
    # unique tops and counts; sort by count
    ti = sorted(Counter(tl).items(),  key=lambda item: item[1], reverse=True)
    
    # summarize: counts
    top_count = len(ti)
    genre_count = len(gl)
    top1_count = (ti[0][1])

    # categorize: majority, hybrid, multi
    if top1_count * 2 > genre_count:
        top_type = 'major'
    elif top_count == 2:
        top_type = 'hybrid'
    else:
        top_type = 'multi'

    # top genres 1-3
    top_genre1 = top_genre2 = top_genre3 = ''
    if top_count > 0:
        top_genre1 = genres.loc[ti[0][0]]['title']
    if top_count > 1:
        top_genre2 = genres.loc[ti[1][0]]['title']
    if top_count > 2:
        top_genre3 = genres.loc[ti[2][0]]['title']
    
    return(top_genre1, top_genre2, top_genre3, top_count, top_type)

def add_top_genres(tracks, genres):
    ''' add top_genre columns to tracks dataframe
    '''
    
    tops = tracks[('track','genres')].apply(lambda r: top_genres(r, genres))
    tracks[[('track','top_genre1'),
            ('track','top_genre2'),
            ('track','top_genre3'),
            ('track','top_genre_count'),
            ('track','top_genre_type')]]= pd.DataFrame(tops.tolist(), index=tops.index)