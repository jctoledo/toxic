import argparse
import logging
import math
import string

import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


####### INITIALIZATION ########

LOG_FILENAME = 'toxic.log'
open(LOG_FILENAME, 'w').close()
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


SWEAR_WORDS = set()
PUNCTUATIONS = ['#','!','@','$','*','~']
CLASS_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def compute_swear_word_proportion_df(adf, colname):
    rm = pd.Series()
    for index, row in adf.iterrows():
        sp = compute_swear_word_proportion_str(row[colname])
        rm.set_value(index, sp)
    return rm

def compute_swear_word_proportion_str(astr):
    rm = 0.0
    if astr is None or len(astr) == 0:
        rm = 0.0
    words = astr.split()
    if len(words) > 0:
        s = sum(remove_punctuation(w.lower()) in SWEAR_WORDS for w in words)
        prop = s / len(astr)
        rm = prop
    return math.log10(rm + 1e-6)


def compute_punctuation_proportion_str(astr):
    '''
    Compute the proportion of the string that is a "punctuation" mark
    some characters will be ignored
    '''
    if astr is None or len(astr) == 0:
        return 0.0
    zum = 0.0
    for c in astr:
        if c in PUNCTUATIONS:
            zum += 1
    prop = zum/len(astr)
    return prop


def compute_punctuation_proportion_str2(astr):
    punctz = set()
    if astr is None or len(astr) == 0:
        return 0.0
    exclusions = ['.','\n','\t', ',', ' ', '\'']
    zum = 0
    for c in astr:
        if c in punctz:
            zum += 1
        else:
            if not c.isalnum() :
                zum += 1
                punctz.add(c)
    prop = zum / len(astr)
    return prop


def compute_punctuation_proportion_df(adf, colname):
    rm = pd.Series()
    for index,row in adf.iterrows():
        pp = compute_punctuation_proportion_str2(row[colname])
        rm.set_value(index, pp)
    return rm


def get_all_comments(train, test, colname):
    return pd.concat([train[colname], test[colname]])

def read_swear_words(afp):
    rm = set()
    fin = open(afp, 'r')
    for l in fin:
        rm.add(l.strip().lower())
    fin.close()
    return rm

def remove_punctuation(astr):
    for c in string.punctuation:
        astr = astr.replace(c,"")
    return astr


def create_word_vectorizer():
    wv = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=10000)
    return wv


def create_char_vectorizer():
    cv = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=50000)
    return cv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--traindata", help="Path to location of training data")
    parser.add_argument("-tst", "--testdata", help="Path to location of testing data")
    parser.add_argument("-swd","--swearwords", help="Path to location of swear words dictionary")
    parser.add_argument('--tune', dest='tune', action='store_true')
    parser.add_argument('--use_params', dest='use_params', action='store_true')
    parser.set_defaults(tune=False,use_params=False)
    args = parser.parse_args()
    logging.info("Starting ...")
    SWEAR_WORDS = read_swear_words(args.swearwords)
    #load the data
    train = pd.read_csv(args.traindata).fillna(' ')
    test = pd.read_csv(args.testdata).fillna(' ')

    #concatenate text
    all_text = get_all_comments(train,test,'comment_text')

    logging.info("Computing punctuation proportions...")

    train['punctuation_proportion'] = compute_punctuation_proportion_df(train, 'comment_text')
    test['punctuation_proportion'] = compute_punctuation_proportion_df(test, 'comment_text')

    logging.info("Finished computing punctuation proportions...")
    logging.info("Started computing swearing proportions ...")

    train['swear_proportion'] = compute_swear_word_proportion_df(train, 'comment_text')
    test['swear_proportion'] = compute_swear_word_proportion_df(test, 'comment_text')
    logging.info("Finished computing swearing proportions")


    logging.info("Starting word vectorizer")
    # word vectorizer
    word_vectorizer = create_word_vectorizer()
    word_vectorizer.fit(all_text)

    train_word_features = word_vectorizer.transform(train['comment_text'])
    test_word_features = word_vectorizer.transform(test['comment_text'])

    logging.info("Finished word vectorizer ...")

    logging.info("Started char vectorizer")
    # char vectorizer
    char_vectorizer = create_char_vectorizer()
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train['comment_text'])
    test_char_features = char_vectorizer.transform(test['comment_text'])

    logging.info("finished char vectorizer...")
    #collect all features
    # collect all features

    train_pp = scipy.sparse.csr_matrix(train['punctuation_proportion']).transpose()
    test_pp = scipy.sparse.csr_matrix(test['punctuation_proportion']).transpose()

    train_sp = scipy.sparse.csr_matrix(train['swear_proportion']).transpose()
    test_sp = scipy.sparse.csr_matrix(test['swear_proportion']).transpose()

    train_features = hstack([train_char_features, train_word_features, train_pp, train_sp])
    test_features = hstack([test_char_features, test_word_features, test_pp, test_sp])
    logging.info("Preparing to train...")

    tuned_params = {
        'toxic':{'C':1000, 'max_iter':400},
        'severe_toxic': {'C': 1000, 'max_iter': 400},
        'obscene': {'C': 1000, 'max_iter': 400},
        'insult': {'C': 1000, 'max_iter': 400},
        'identity_hate': {'C': 1000, 'max_iter': 400},
        'threat' : {'C':1000, 'max_iter':400}
    }

    if args.tune == False:
        scores = []
        submission = pd.DataFrame.from_dict({'id': test['id']})
        for class_name in CLASS_NAMES:
            try:
                train_target = train[class_name]

                classifier = LogisticRegression(solver='sag')
                if args.use_params:
                    mi = tuned_params[class_name]['max_iter']
                    c = tuned_params[class_name]['C']
                    classifier = LogisticRegression(solver='sag',max_iter=mi,C=c)


                cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
                scores.append(cv_score)
                print('CV score for class {} is {}'.format(class_name, cv_score))
                logging.info('CV score for class {} is {}'.format(class_name, cv_score))
                classifier.fit(train_features, train_target)
                submission[class_name] = classifier.predict_proba(test_features)[:, 1]
            except ValueError as e:
                print(class_name + " => " + str(e))
        print('Total CV score is {}'.format(np.mean(scores)))
        logging.info('Total CV score is {}'.format(np.mean(scores)))
        from random import randint
        r = randint(0, 100000)
        fout = "submission_"+str(r)+".csv"
        submission.to_csv(fout, index=False)
    else:
        fout = open("best_params.txt", "w")
        Cs = [ 100, 1000]
        maxiters = [100,500,2000]
        for class_name in CLASS_NAMES:
            try:
                train_target = train[class_name]
                classifier = LogisticRegression(solver='sag')

                grid = GridSearchCV(estimator=classifier, param_grid=dict(C=Cs,max_iter=maxiters), scoring='roc_auc',n_jobs=2)
                grid_result = grid.fit(train_features, train_target)
                perf_line = "Best for class name : %s -> %f using %s" % (class_name,grid_result.best_score_, grid_result.best_params_)
                logging.info("----")
                logging.info(perf_line)
                logging.info("----")
                fout.write(perf_line+"\n")

            except ValueError as e:
                print(class_name + " => " + str(e))

        fout.close()
    pass

