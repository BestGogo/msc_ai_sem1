import time
import numpy as np
import copy

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import random
seed = 1

class ItemSelector(BaseEstimator, TransformerMixin):
    """
    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    param: key: hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class ColumnExtractor(object):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        sliced = X[:, self.cols]
        return sliced

    def fit(self, X, y=None):
        return self
    
def returnNumpyArray(x):
    return np.array([t for t in x]).reshape(-1, 1)

def returnNumpyMatrix(x):
    return x

def naiveBayesClassifier(training_data_X, training_data_y, vocab, word_vocab, nb_type):
    if nb_type == 'tf': 
        pos_vectors = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
        text_vectors = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

        classifier = MultinomialNB()

        feature_cat_1 = FeatureUnion([
                ('POS', Pipeline([
                    ('selector', ItemSelector(key='POS')),
                    ('vectorizer', pos_vectors),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text_norm')),
                    ('vectorizer', text_vectors),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('gf', Pipeline([
                    ('selector', ItemSelector(key='gf')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('fa', Pipeline([
                    ('selector', ItemSelector(key='fa')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('diag_act', Pipeline([
                    ('selector', ItemSelector(key='diag_act')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
            ])

        feature_cat_2 = FeatureUnion([
                ('word_count', Pipeline([
                    ('selector', ItemSelector(key='word_count')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                    ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                ])),
                ('f_measure', Pipeline([
                    ('selector', ItemSelector(key='f_measure')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                    ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                ])),
            ])

        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('pipeline', Pipeline([
                    ('features', feature_cat_1),
                    ('scaler', MaxAbsScaler()),
                ])),
                ('pipeline2', Pipeline([
                    ('features', feature_cat_2),
                    ('scaler', MinMaxScaler()),
                ])),
            ])),
            ('clf', classifier),
        ])

    text_clf.fit(training_data_X, training_data_y)
    return text_clf

def SVMClassifier(training_data_X, training_data_y, vocab, word_vocab, svm_type):
    if svm_type == 'bool': 
        pos_vectors = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
        text_vectors = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), lowercase=True)

        classifier = LinearSVC(max_iter=100000, C=0.001, penalty='l2', loss='squared_hinge')
        parameters = [
                {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge']},
                {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False]}
            ]
        fclassifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

        feature_cat_1 = FeatureUnion([
                ('POS', Pipeline([
                    ('selector', ItemSelector(key='POS')),
                    ('vectorizer', pos_vectors),
                    ('binarize', Binarizer()),
                ])),
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text_norm')),
                    ('vectorizer', text_vectors),
                    ('binarize', Binarizer()),
                ])),
                ('gf', Pipeline([
                    ('selector', ItemSelector(key='gf')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('binarize', Binarizer()),
                ])),
                ('fa', Pipeline([
                    ('selector', ItemSelector(key='fa')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('binarize', Binarizer()),
                ])),
                ('diag_act', Pipeline([
                    ('selector', ItemSelector(key='diag_act')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('binarize', Binarizer()),
                ])),
            ])

        feature_cat_2 = FeatureUnion([
                ('word_count', Pipeline([
                    ('selector', ItemSelector(key='word_count')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                    ('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                ])),
                ('f_measure', Pipeline([
                    ('selector', ItemSelector(key='f_measure')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                    ('binarize', Binarizer(threshold=50)),
                ])),
            ])
        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('pipeline', Pipeline([
                    ('features', feature_cat_1),
                ])),
                ('pipeline2', Pipeline([
                    ('features', feature_cat_2),
                ])),
            ])),
            ('clf', fclassifier),
        ])
    elif svm_type == 'discrete': 
        pos_vectors = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
        text_vectors = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

        classifier = LinearSVC(max_iter=100000)
        parameters = [
                {'C': [0.1, 1, 2, 3, 4, 5]},
            ]
        fclassifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

        feature_cat_1 = FeatureUnion([
                ('POS', Pipeline([
                    ('selector', ItemSelector(key='POS')),
                    ('vectorizer', pos_vectors),
                    ('scaler', Normalizer(norm='l2')),
                ])),
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text_norm')),
                    ('vectorizer', text_vectors),
                    ('scaler', Normalizer(norm='l2')),
                ])),
                ('gf', Pipeline([
                    ('selector', ItemSelector(key='gf')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('scaler', Normalizer(norm='l2')),
                ])),
                ('fa', Pipeline([
                    ('selector', ItemSelector(key='fa')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('scaler', Normalizer(norm='l2')),
                ])),
                ('diag_act', Pipeline([
                    ('selector', ItemSelector(key='diag_act')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('scaler', Normalizer(norm='l2')),
                ])),
            ])

        feature_cat_2 = FeatureUnion([
                ('word_count', Pipeline([
                    ('selector', ItemSelector(key='word_count')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                    ('scaler', Normalizer(norm='l2')),
                ])),
                ('f_measure', Pipeline([
                    ('selector', ItemSelector(key='f_measure')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                    ('scaler', Normalizer(norm='l2')),
                ])),
            ])
        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('pipeline', Pipeline([
                    ('features', feature_cat_1),
                ])),
                ('pipeline2', Pipeline([
                    ('features', feature_cat_2),
                ])),
            ])),
            ('clf', fclassifier),
        ])
    elif svm_type == 'svc': 
        pos_vectors = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
        text_vectors = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

        classifier = SVC(C=1, kernel='linear')
        parameters = [
                {'kernel': ['linear'], 'C': [0.1, 1, 2]},
                {'kernel': ['rbf'], 'C': [0.1, 1, 2], 'gamma': ['auto', 'scale']},
                {'kernel': ['poly'], 'C': [0.1, 1, 2], 'gamma': ['auto', 'scale'], 'degree': [2, 3, 4]},
                {'kernel': ['sigmoid'], 'C': [0.1, 1, 2], 'gamma': ['auto', 'scale']},
            ]
        fclassifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

        feature_cat_1 = FeatureUnion([
                ('POS', Pipeline([
                    ('selector', ItemSelector(key='POS')),
                    ('vectorizer', pos_vectors),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text_norm')),
                    ('vectorizer', text_vectors),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('gf', Pipeline([
                    ('selector', ItemSelector(key='gf')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
                ('diag_act', Pipeline([
                    ('selector', ItemSelector(key='diag_act')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                ])),
            ])

        feature_cat_2 = FeatureUnion([
                ('word_count', Pipeline([
                    ('selector', ItemSelector(key='word_count')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                ])),
                ('f_measure', Pipeline([
                    ('selector', ItemSelector(key='f_measure')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                ])),
            ])
        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('pipeline', Pipeline([
                    ('features', feature_cat_1),
                    ('scaler', MaxAbsScaler()),
                ])),
                ('pipeline2', Pipeline([
                    ('features', feature_cat_2),
                    ('scaler', MinMaxScaler()),
                ])),
            ])),
            ('clf', fclassifier),
        ])
    
    text_clf.fit(training_data_X, training_data_y)
    
    return text_clf

    
def SVMRClassifier(training_data_X, training_data_y, vocab, word_vocab, svmr_type):
    if svmr_type == 'linearsvr':
        pos_vectors = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
        text_vectors = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), lowercase=True)

        classifier = LinearSVR(max_iter=100000)
        parameters = [
                {'C': [0.1, 1, 10], 'epsilon': [0, 0.1, 1], 'loss':('epsilon_insensitive', 'squared_epsilon_insensitive')}
            ]
        fclassifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

        feature_cat_1 = FeatureUnion([
                ('POS', Pipeline([
                    ('selector', ItemSelector(key='POS')),
                    ('vectorizer', pos_vectors),
                    ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                ])),
                ('text', Pipeline([
                    ('selector', ItemSelector(key='text_norm')),
                    ('vectorizer', text_vectors),
                    ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                ])),
                ('gf', Pipeline([
                    ('selector', ItemSelector(key='gf')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                ])),
                ('fa', Pipeline([
                    ('selector', ItemSelector(key='fa')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                ])),
                ('diag_act', Pipeline([
                    ('selector', ItemSelector(key='diag_act')),
                    ('toarray', FunctionTransformer(returnNumpyMatrix, validate = False)),
                    ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                ])),
            ])

        feature_cat_2 = FeatureUnion([
                ('word_count', Pipeline([
                    ('selector', ItemSelector(key='word_count')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                ])),
                ('f_measure', Pipeline([
                    ('selector', ItemSelector(key='f_measure')),
                    ('toarray', FunctionTransformer(returnNumpyArray, validate = False)),
                ])),
            ])

        text_clf = Pipeline([
            ('features', FeatureUnion([
                ('pipeline', Pipeline([
                    ('features', feature_cat_1),
                ])),
                ('pipeline2', Pipeline([
                    ('features', feature_cat_2),
                    ('scaler', MinMaxScaler()),
                ])),
            ])),
            ('clf', fclassifier),
        ])
    text_clf.fit(training_data_X, training_data_y)
    return text_clf



