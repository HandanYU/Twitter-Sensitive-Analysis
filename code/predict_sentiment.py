import pandas as pd
from my_function.compute_tfidf import comput_info_tfdif
from my_function.transform_data import split_dict, get_coo_matrix
import numpy as np
from scipy.sparse import coo_matrix
import pickle


test_data = pd.read_csv('../data/test_tfidf.csv')
compute_info_tfidf_data = np.load('../result_data/compute_info_tfidf_data.npz')
off_word, info_entr, full_word_count = compute_info_tfidf_data['off_word'].tolist(), compute_info_tfidf_data['info_entr'].item(),compute_info_tfidf_data['full_word_count'].item()

# split dataset
split_dict(test_data)
# improved tfidf feature selection
comput_info_tfdif(test_data, info_entr, full_word_count)
coo_M_test = get_coo_matrix(test_data)
test_X = coo_M_test.todense()
test_X = np.delete(test_X, off_word, axis=1)
test_X = coo_matrix(test_X)
# best_k_selector = pickle.load(open('../model/update_feature_selector.sav','rb'))
# test_X = best_k_selector.transform(test_X)
# predict
classifier = 'Softmax'
res = pd.DataFrame(columns = ['tweet_id','sentiment'])
res['tweet_id'] = test_data['tweet_id']
model = pickle.load(open('../model/{}_scalar_data_update_data_model.sav'.format(classifier), 'rb'))
res['sentiment'] = model.predict(test_X.todense()).tolist() 
res.to_csv('../result_data/{}_result.csv'.format(classifier),index=False)

