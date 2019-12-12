import time
import numpy as np
import copy

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neural_network import MLPClassifier
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import model_selection
import random
from numpy.random import choice
from scipy.stats import beta
from scipy.stats import expon

from Classes.ColumnExtractor import ColumnExtractor
from Classes.DenseTransformer import DenseTransformer
from Classes.EFS import EFS
from Classes.FSC import FSC
from Classes.ItemSelector import ItemSelector
from Classes.ItemSelectorTF import ItemSelectorTF
from Helper.DebugPrint import DebugPrint

RANDOMIZER_SEED = 1

class Classifier(object):
    def GetFeatures(self, training_data_dict, training_data_classification, vectorizer_pipeline, classifier, feature_selections):
        start = time.time()
        efs_obj = EFS()

        X = vectorizer_pipeline.fit_transform(training_data_dict, training_data_classification)
        candidate_feature_indexes = efs_obj.EFS(X, training_data_classification, classifier, feature_selections)

        end = time.time()
        DebugPrint("Time Run = %fs" % (end - start))

        return candidate_feature_indexes
    
    def BuildClassifierNB(self, training_data_dict, training_data_classification, vocab, word_vocab, nb_type):
        ## Build and Train Model ######################
        if nb_type == 'tf': 
            ## TF - 1-GRAM [CHI, IG] [ALL] (0.716 ACC)
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = MultinomialNB()

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 10, encode='ordinal', strategy='uniform')),
                    ])),
                ])

            reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        ('reducer', reducer),
                        ('scaler', MaxAbsScaler()),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', classifier),
            ])
        text_clf.fit(training_data_dict, training_data_classification)
        ##############################################

        feats = text_clf.named_steps['features']
        test = feats.transform(training_data_dict)
        print('Training Vect Examples')
        for (count, feature) in zip(test[0].toarray()[0], vectorizer.get_feature_names()):
           print(str(count) + ' ' + feature)
        print(zip(test[0], vectorizer.get_feature_names()))
        print(test[1])

        feats = text_clf.named_steps['features']
        test = feats.transform(training_data_dict)
        DebugPrint(test[1])
        print(test[1].toarray())
        return text_clf
    def BuildClassifierSVM(self, training_data_dict, training_data_classification, vocab, word_vocab, svm_type):
        ## Build and Train Model ######################

        if svm_type == 'bool': 
            # Bool - 0.67
            # Kind of sucks
            # Reducer sux
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), lowercase=True)

            classifier = LinearSVC(max_iter=100000, C=0.001, penalty='l2', loss='squared_hinge')
            parameters = [
                    {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge']},
                    {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False]}
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('binarize', Binarizer()),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('binarize', Binarizer()),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('binarize', Binarizer()),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        #('discretize', KBinsDiscretizer(n_bins = 2, encode='ordinal', strategy='uniform')),
                        ('binarize', Binarizer(threshold=50)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                    ])),
                ])),
                ('clf', final_classifier),
            ])
        elif svm_type == 'discrete': 
            # Discrete - 2GRAM 0.70 ACC NO REDUC
            # Normalizer Performs Better Than MinMaxScaler
            # L2 norm performs better than L1 normalization
            # 1-Gram vs 2-Gram doen't matter - TO DO
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = LinearSVC(max_iter=100000)
            parameters = [
                    {'C': [0.1, 1, 2, 3, 4, 5]},
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                        ('scaler', Normalizer(norm='l2')),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, classifier, [FSC.CHI, FSC.IG])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                    ])),
                ])),
                ('clf', final_classifier),
            ])
        elif svm_type == 'svc': 
            # SVC 0.73 linear 1-gram
            # No reducer
            # Linear and Sigmoid are Good
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=True, tokenizer=lambda x: x.split(' '))

            classifier = SVC(C=1, kernel='linear')
            parameters = [
                    {'kernel': ['linear'], 'C': [0.1, 1, 2]},
                    #{'kernel': ['rbf'], 'C': [0.1, 1, 2], 'gamma': ['auto', 'scale']},
                    ##{'kernel': ['poly'], 'C': [0.1, 1, 2], 'gamma': ['auto', 'scale'], 'degree': [2, 3, 4]},
                    #{'kernel': ['sigmoid'], 'C': [0.1, 1, 2], 'gamma': ['auto', 'scale']},
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tfidf', TfidfTransformer(use_idf=True)),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB(), [FSC.CHI, FSC.IG, FSC.MI, FSC.CE, FSC.WOE])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                        ('scaler', MaxAbsScaler()),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', final_classifier),
            ])
        
        text_clf.fit(training_data_dict, training_data_classification)
        
        return text_clf
    def BuildClassifierSVMR(self, training_data_dict, training_data_classification, vocab, word_vocab, svmr_type):
        ## Build and Train Model ######################
        
        if svmr_type == 'linearsvr':
            # LinearSVR - 0.72
            # L2 > L1
            # TFIDF > TF
            pos_vectorizer = CountVectorizer(vocabulary=vocab, analyzer='word', ngram_range=(1, 5), tokenizer=lambda x: x.split(' '), lowercase=False)
            text_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), lowercase=True)

            classifier = LinearSVR(max_iter=100000)
            parameters = [
                    {'C': [0.1, 1, 10], 'epsilon': [0, 0.1, 1], 'loss':('epsilon_insensitive', 'squared_epsilon_insensitive')}
                ]
            final_classifier = GridSearchCV(classifier, parameters, cv=5, n_jobs=7)

            features1 = FeatureUnion([
                    ('pos', Pipeline([
                        ('selector', ItemSelector(key='pos')),
                        ('vectorizer', pos_vectorizer),
                        ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                    ])),
                    ('text', Pipeline([
                        ('selector', ItemSelector(key='tokenized_text')),
                        ('vectorizer', text_vectorizer),
                        ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                    ])),
                    ('gpf', Pipeline([
                        ('selector', ItemSelector(key='gpf')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                    ])),
                    ('fa', Pipeline([
                        ('selector', ItemSelector(key='fa')),
                        ('toarray', FunctionTransformer(self.GetMultipleGenericArray, validate = False)),
                        ('tf', TfidfTransformer(norm='l2', use_idf=True)),
                    ])),
                ])

            features2 = FeatureUnion([
                    ('wordcount', Pipeline([
                        ('selector', ItemSelector(key='wordcount')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                    ('fmeasure', Pipeline([
                        ('selector', ItemSelector(key='fmeasure')),
                        ('toarray', FunctionTransformer(self.GetGenericArray, validate = False)),
                    ])),
                ])

            #reducer_features = self.GetFeatures(training_data_dict, training_data_classification, features1, MultinomialNB(), [FSC.CHI])
            #reducer = ColumnExtractor(cols=reducer_features)

            text_clf = Pipeline([
                ('features', FeatureUnion([
                    ('pipeline', Pipeline([
                        ('features', features1),
                        #('reducer', reducer),
                    ])),
                    ('pipeline2', Pipeline([
                        ('features', features2),
                        ('scaler', MinMaxScaler()),
                    ])),
                ])),
                ('clf', final_classifier),
            ])
                    text_clf.fit(training_data_dict, training_data_classification)
        #print(final_classifier.best_params_)
        ###############################################

        return text_clf

        return text_clf
    def GetGenericArray(self, x):
        return np.array([t for t in x]).reshape(-1, 1)

    def GetMultipleGenericArray(self, x):
        return x