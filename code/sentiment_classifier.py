import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler

from my_function.analyze_gender import compute_acc_gender, create_gender_dict, identify_gender
from my_function.feature_selection import featureSelection
from my_function.compute_tfidf import split_dict

# read datasets
train_data = pd.read_csv('../data/train_tfidf.csv')
dev_data = pd.read_csv('../data/dev_tfidf.csv')
train_count = pd.read_csv('../data/train_count.csv')
# feature select
feature_selector = featureSelection(train_data, train_count, dev_data)
tradiction_X_train, tradiction_X_test = feature_selector.feature_select('traditional')
update_X_train, update_X_test = feature_selector.feature_select('update')

# scalar data
print('--------BEGIN TO SCALAR TRADITIONAL DATA-----------')
scalar = MaxAbsScaler()
scalar_old_train_X = scalar.fit_transform(tradiction_X_train)
scalar_old_test_X = scalar.transform(tradiction_X_test)

print('--------BEGIN TO SCALAR UPDATE DATA-----------')
scalar1 = MaxAbsScaler()
scalar_new_train_X = scalar1.fit_transform(update_X_train)
scalar_new_test_X = scalar1.transform(update_X_test)

# train model
y_train = train_count.sentiment
y_test = dev_data.sentiment
split_dict(dev_data)
identify_gender('../data/gender_word_list.npz', dev_data, '../data/vocab.txt')

models = {
    'NB': MultinomialNB(alpha=1e-5),
          'Softmax': LogisticRegression(C=10),
          'KNN': KNeighborsClassifier(n_neighbors=5)
          }  # classifier lists
data_sets = {
    'no_scalar_data': {'traditional_data': (tradiction_X_train, tradiction_X_test),
                       'update_data': (update_X_train, update_X_test)},
    'scalar_data': {'traditional_data': (scalar_old_train_X, scalar_old_test_X),
                    'update_data': (scalar_new_train_X, scalar_new_test_X)}
}
np.savez('../result_data/data_sets.npz',data_sets=data_sets)
for model in models:
    for scalar in data_sets:
        for data in data_sets[scalar]:
            X_train, X_test = data_sets[scalar][data]
            m = models[model]

            # draw learning curve
            # train_sizes, train_scores, valid_scores = learning_curve(m,X_train,y_train,
            #                                                          scoring = 'accuracy',cv = StratifiedKFold(10),
            #                                                          train_sizes=np.linspace(0.1,1.0,5))
            # plt.figure()
            # plt.xlabel('training size')
            # plt.ylabel('accuracy')
            # plt.plot(train_sizes,np.mean(train_scores,axis = 1),'o-',color='r',label = 'Training score')
            # plt.plot(train_sizes,np.mean(valid_scores,axis = 1),'o-',color='g',label = 'Cross-validation score')
            # plt.legend(loc='best')
            # plt.title('{}\'s learning curve'.format(model))
            # plt.show()

            print('------------FITTING---------')
            m = m.fit(X_train, y_train)
            # save current model
            pickle.dump(m, open('../model/{}_{}_{}_model.sav'.format(model,scalar, data), 'wb'))
            print('----------PREDICT-----------')
            pre = m.predict(X_test)

            # analyse gender bias under scalared and update TFIDF feature selection classifiers
            if scalar == 'scalar_data' and data == 'update_data':
                dev_data['{}_pre'.format(model)] = pre
                df_dict = create_gender_dict(dev_data)
                for df in df_dict:
                    print('{}\'s accuracy is {}'.format(df, compute_acc_gender(df_dict[df], '{}_pre'.format(model))))
            # evaluate
            print('-----------EVALUATE-----------')
            acc = accuracy_score(pre, y_test)
            f1 = f1_score(pre, y_test, average='macro')
            print('{}\n{}_{} dataset'.format(model, scalar, data))
            print('acc:{}\nf1-score:{}'.format(acc, f1))
