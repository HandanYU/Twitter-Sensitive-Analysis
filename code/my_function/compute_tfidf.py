import numpy as np
from collections import Counter
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from .transform_data import split_dict,get_coo_matrix

def count_word_class(label, full_data):
    """
    count the numbers of words occur in each class

    INPUT:
    label : label_name, string
    full_data : dataset including all labels feature data, dataframe

    RETURN: dict
    """
    cur_data = full_data[full_data.sentiment == label]
    word_counter = Counter({})
    for i in range(cur_data.shape[0]):
        word_counter += Counter(cur_data.word.iloc[i])

    return word_counter

def get_word_count(data):
    """
    get each word's count in the all instances

    RETURN:
    each word's count in each class, and each word's count in the all instances
    """
    neg = count_word_class('neg',data)
    pos = count_word_class('pos',data)
    neu = count_word_class('neu',data)
    full_word_count = dict(neg+pos+neu)
    return neg, pos, neu, full_word_count

def get_word_distribution(neg, pos, neu, full_word_count):
    """
    get each word's frequency in the all instances under the certain class

    RETURN:
    each word's frequency
    """
    neg_p = {}
    pos_p = {}
    neu_p = {}
    for word, count in full_word_count.items():
        neg_p[word] = neg.get(word,0)/count
        pos_p[word] = pos.get(word,0)/count
        neu_p[word] = neu.get(word,0)/count
    return [neg_p, pos_p, neu_p]
def compute_entropy(full_word_count,distribution_list):
    """
    compute each word's entropy
    """
    info_entr = {}
    off_word = []
    for k in full_word_count.keys():
        S = 0 # record the information entropy of word k
        for i in range(len(distribution_list)):
            pi = distribution_list[i].get(k,0) # the probability of word k in class i
            if pi != 0:  # since np.log2(x) where x>0, in order to ganruantee that, let pi a small number.
                S -= pi *np.log2(pi)
        # when the information entropy of word k is similar to the maximum of information entropy, we should assume to ignore this key words.
        if abs(S-np.log2(len(distribution_list))) < 1e-5:
            off_word.append(k)
        elif abs(S) < 1e-5:
            S = 1e-5
        info_entr[k] = S
    return info_entr, off_word
def comput_info_tfdif(data,info_entr_dict,full_word_count):
    """
    update tfdif

    tfdif /= information entropy
    """
    zip_new = []
    for i in range(data.shape[0]):
        cur_dict = {}
        for j in data['zip'].iloc[i].keys():
            if j in info_entr_dict:
                nk = full_word_count[j]
                fj = -((nk-1)/nk*np.log2((nk-1)/nk)+1/nk*np.log2(1/nk))
                bottom_value = info_entr_dict[j] + fj
                cur_dict[j] = data['zip'].iloc[i].get(j,0) * np.log2(1/bottom_value+1)
        zip_new.append(cur_dict)
    data['zip'] = zip_new

def getTfdifMatrix(tfdif_df, count_df):
    split_dict(tfdif_df)
    split_dict(count_df)
    print("---------FINISH SPLIT---------")
    neg, pos, neu, full_word_count = get_word_count(count_df)
    distribution_list = get_word_distribution(neg, pos, neu, full_word_count)
    info_entr, off_word = compute_entropy(full_word_count,distribution_list)
    print('------FINSHISH COMPUTE ENTROPY----------')
    for i in off_word:
        del info_entr[i]
    comput_info_tfdif(tfdif_df,info_entr,full_word_count)
    print(tfdif_df.shape)
    print('---------------FINISH UPDATE TFDIF------------')
    coo_M = get_coo_matrix(tfdif_df)
    print('------------FINISH GET COO MATRIX-------------')
    return coo_M, off_word, info_entr,full_word_count