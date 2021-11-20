import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def get_word_dict(path):
    """
    obtain the word_dict like {word_id: word...}

    :param path: the path of word_dict text file
    :return: a dictionary like {word_id: word...}
    """
    word_dict = {}
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            word_dict[int(line.split('\t')[1])] = line.split('\t')[0]
    return word_dict


def get_gender_word(path):
    """
    obtain the gender words according to opinion lexicon

    :param path: the path of npz file for gender words
    :return: a list of male words, like ['boy','king',..] and a list of female words, like ['girl','mom',...]
    """
    gender_word = np.load(path)
    male = gender_word['male']
    female = gender_word['female']
    return male, female


# def change_gender_word(word,gender,male_id_list,female_id_list):
#     """
#
#     :param word:
#     :param gender:
#     :param male_id_list:
#     :param female_id_list:
#     :return:
#     """
#     words = word.copy()
#     for i in range(len(words)):
#         if gender == 'male':
#             if words[i] in male_id_list:
#                 words[i] = max(female_id_list, key=lambda v: female_id_list.count(v))
#         if gender == 'female':
#             if words[i] in female_id_list:
#                 words[i] = max(male_id_list, key=lambda v: male_id_list.count(v))
#     return words
def identify_gender(gender_dict_path, split_data, word_dict_path):
    """
    identify the gender of each tweets according to the keywords conveying gender information

    :param gender_dict_path: the path of gender words
    :param split_data: the dataset after transforming
    :param word_dict_path: the path of words dict
    :return: add gender information into split_data
    """
    word_dict = get_word_dict(word_dict_path)
    male, female = get_gender_word(gender_dict_path)
    male_id_list = [k for k in word_dict if word_dict[k] in male]
    female_id_list = [k for k in word_dict if word_dict[k] in female]
    split_data['gender'] = split_data.apply(lambda x: 'male' if sum([w for w in x.word if w in male_id_list]) >= 1 else 'no', axis=1)
    split_data['gender'] = split_data.apply(lambda x: 'female' if sum([w for w in x.word if w in female_id_list]) >= 1 else x.gender, axis=1)


def create_gender_dict(df):
    """
    create a gender dict

    :param df: the dataframe with gender information
    :return: a dictionary for gender types with their corresponding parts of tweets instances
    """
    male_df = df[df.gender == 'male']
    female_df = df[df.gender == 'female']
    no_df = df[df.gender == 'no']
    df_dict = {'male': male_df, 'female': female_df, 'no': no_df}
    return df_dict


def compute_acc_gender(df, pre_col):
    """
    compute the accuracy of classifier under a certain gender group tweets
    and also present the confusion matrix of classifier under a certain gender group tweets

    :param df: a dataframe of a certain gender group tweets
    :param pre_col: the column name of a certain classifier prediction
    :return: the accuracy of classifier under a certain gender group tweets
    """
    print(pre_col)
    print(classification_report(df.sentiment, df[pre_col]))
    C2 = confusion_matrix(df.sentiment, df[pre_col], labels=['neg', 'neu', 'pos'])
    print(C2)
    df[pre_col + '_acc'] = df.apply(lambda x: x.sentiment == x[pre_col], axis=1)
    return sum(df[pre_col + '_acc']) / df.shape[0]
