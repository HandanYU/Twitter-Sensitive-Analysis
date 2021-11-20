import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.feature_selection import SelectKBest, chi2
import pickle
from .compute_tfidf import getTfdifMatrix,comput_info_tfdif
from .transform_data import split_dict, get_coo_matrix


class featureSelection:
    def __init__(self, train_data, train_count, test_data):
        self.train_data = train_data
        self.train_count = train_count
        self.test_data = test_data

    # def select_best(self, train_X, train_y, test_X, k=2000, tfidf_type = 'update'):
    #     """
    #     select best k features using chi-square

    #     :param train_X: sparse matrix of training features
    #     :param train_y: training tweets sentiment labels
    #     :param test_X: sparse matrix of testing features
    #     :param k: the number of best features
    #     :param tfidf_type: the type of TFIDF feature selection, the default is 'update' which is used to predict test data. The other option is 'traditional'
    #     :return: a sparse matrix of k best training features and a sparse matrix of k best testing features
    #     """
    #     x2 = SelectKBest(chi2, k)
    #     X_train_x2 = x2.fit_transform(train_X, train_y)
    #     X_test_x2 = x2.transform(test_X)
    #     pickle.dump(x2, open('../model/{}_feature_selector.sav'.format(tfidf_type), 'wb'))
    #     return X_train_x2, X_test_x2

    def traditional_tfidf(self):
        """
        obtain training features and testing features after traditional TFIDF and chi-square feature selection

        :return: a tuple of sparse matrixs of training features and testing features
        """
        train_data = self.train_data.copy()
        test_data = self.test_data.copy()

        # transform data
        print('----------BEGIN SPLIT TRAIN DATA----------')
        split_dict(train_data)
        print('----------BEGIN SPLIT TEST DATA----------')
        split_dict(test_data)
        # transform TFIDF values into sparse matrix,
        # in order to available for training model
        print('----------BEGIN TRANSFORM TRAIN DATA----------')
        train_X = get_coo_matrix(train_data)
        print('----------BEGIN TRANSFORM TEST DATA----------')
        test_X = get_coo_matrix(test_data)
        print('-------BEGIN SELECT KBEST-----------')
        train_y = train_data.sentiment
        # X_train_x2, X_test_x2 = self.select_best(train_X, train_y, test_X, 2000,'traditional')
        return train_X, test_X

    def update_tfidf(self):
        """
        obtain training features and testing features after updated TFIDF with information entropy and chi-square feature selection

        :return: a tuple of sparse matrixs of training features and testing features
        """
        train_data = self.train_data.copy()
        train_count = self.train_count.copy()
        test_data = self.test_data.copy()

        coo_M_train, off_word, info_entr, full_word_count = getTfdifMatrix(train_data, train_count)

        np.savez('../result_data/compute_info_tfidf_data.npz',off_word=off_word,info_entr=info_entr,full_word_count=full_word_count)

        train_X = coo_M_train.todense()
        train_X = np.delete(train_X, off_word, axis=1)

        split_dict(test_data)
        comput_info_tfdif(test_data, info_entr, full_word_count)
        coo_M_test = get_coo_matrix(test_data)
        test_X = coo_M_test.todense()
        test_X = np.delete(test_X, off_word, axis=1)
        train_X, train_y = coo_matrix(train_X), train_data.sentiment
        test_X, test_y = coo_matrix(test_X), test_data.sentiment
        # print('------BEGIN SELECT KBEST-----------')
        # X_train_x2, X_test_x2 = self.select_best(train_X, train_y, test_X, 2000, 'update')
        return train_X, test_X

    def feature_select(self, method = 'update'):
        """
        implement feature selection

        :param method: the method of feature selection (traditional TFIDF or updated TFIDF with information entropy).
                       The default is 'update' which is used to predict test dataset. The other option is 'traditional'.

        :return:
        """
        if method == 'traditional':
            return self.traditional_tfidf()
        if method == 'update':
            return self.update_tfidf()


if __name__ == '__main__':
    train_data = pd.read_csv('../data/train_tfidf.csv')
    test_data = pd.read_csv('../data/dev_tfidf.csv')
    train_count = pd.read_csv('../data/train_count.csv')

    feature_selector = featureSelection(train_data, train_count, test_data)
    tradiction_X_train, tradiction_X_test = feature_selector.feature_select('traditional')
    update_X_train, update_X_test = feature_selector.feature_select('update')
