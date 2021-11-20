from nltk.corpus import opinion_lexicon
from my_function.transform_data import split_dict
import pandas as pd
import itertools

# build baseline according to opinion_lexicon and compute its accuracy

if __name__ == '__main__':
    baseline_data = pd.read_csv('../data/dev_count.csv')
    word_dict = {}
    with open('../data/vocab.txt', "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            word_dict[int(line.split('\t')[1])] = line.split('\t')[0]

    # transform data
    split_dict(baseline_data)

    # obtain positive and negative words from opinion_lexicon
    pos_list=set(opinion_lexicon.positive())
    neg_list=set(opinion_lexicon.negative())

    # obtain positive, negative, and neutral words' id
    neg_id_list = [i for i in word_dict.keys() if word_dict[i] in neg_list]
    pos_id_list = [i for i in word_dict.keys() if word_dict[i] in pos_list]
    neu_id_list = [i for i in word_dict.keys() if (word_dict[i] not in neg_list) and (word_dict[i] not in pos_list)]


    sent_word_list = {'neg':neg_id_list,'pos':pos_id_list,'neu':neu_id_list}
    opp_dict={'neg':'pos','pos':'neg','neu':'neu'}
    
    # analyse sentiment according to sentiment word appeared in opinion_lexicon
    baseline_data['notsent'] = baseline_data.apply(lambda x: [[i]*int(x['zip'][i]) for i in x['zip'].keys() if i in sent_word_list[opp_dict[x['sentiment']]]],axis = 1)
    baseline_data['sent'] = baseline_data.apply(lambda x: [[i]*int(x['zip'][i]) for i in x['zip'].keys() if (i in sent_word_list[x['sentiment']]) and (x['sentiment'] in ['neg','pos'])],axis = 1)
    baseline_data['notsent'] = baseline_data['notsent'].apply(lambda x:list(itertools.chain.from_iterable(x)) )
    baseline_data['sent'] = baseline_data['sent'].apply(lambda x:list(itertools.chain.from_iterable(x)) )
    baseline_data['judge'] = baseline_data.apply(lambda x: len(x['sent'])>=len(x['notsent']),axis=1)

    # the accuracy of opinion_lexicon baseline
    print(sum(baseline_data['judge'])/baseline_data.shape[0])