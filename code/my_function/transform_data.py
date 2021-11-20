from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix


def split_dict(raw_data):
    """
    change [word, count/freq] into dict for each instance
    """
    raw_data['list_tweet'] = raw_data['tweet'].apply(lambda x: x[2:-2].split('), ('))
    raw_data['word'] = raw_data['list_tweet'].apply(lambda x: [int(np.asarray(a.split(','), dtype='float32').tolist()[0]) for a in x])
    raw_data['freq'] = raw_data['list_tweet'].apply(lambda x: [np.asarray(a.split(','), dtype='float32').tolist()[1] for a in x])
    raw_data['zip'] = raw_data.apply(lambda x: Counter(dict(zip(x['word'], x['freq']))), axis=1)


def get_coo_matrix(data):
    """
    get the sparse matrix of features
    """
    cols = []
    rows = []
    full_values = []
    for i in range(data.shape[0]):
        if i % 1000 == 0:
            print("Instance ---- ", i)
        cols += list(data['zip'].iloc[i].keys())
        rows += [i] * len(data['zip'].iloc[i].keys())
        sorted_data = [da[1] for da in sorted(data['zip'].iloc[i].items(), key=lambda d: d[0])]
        full_values += sorted_data
    coo_M = coo_matrix((full_values, (rows, cols)))
    return coo_M
