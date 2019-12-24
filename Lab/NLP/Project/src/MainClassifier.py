import time
import pandas as pd
import numpy as np
import statistics
import time


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from modules.classifier import *


seed = 1

def genderClassification():
    start = time.time()
    target_name = ['Female','Male']
    ## Load the Training Set and Testing Set from numpy dict #####
    train_data_dict = LoadDataSet('../data/train_dict.npy')
    test_data_dict = LoadDataSet('../data/test_dict.npy')


    ## Load POS Patterns ##########################
    pos_vocab = []
    with open('../data/POSPatterns.txt') as file:
        for line in file:
            pos_vocab.append(line.strip('\n'))
    word_vocab = []


    ## Naive Bayes 
    print("### Naive Bayes ###")
    nb_clf = naiveBayesClassifier(train_data_dict, train_data_dict['gender'], pos_vocab, word_vocab, 'tf')
    nb_predictions = nb_clf.predict(test_data_dict)
    get_results("Naive Bayes", test_data_dict['gender'], nb_predictions, target_name)
    
    ## SVM 
    print("### SVM ###")
    svm_clf = SVMClassifier(train_data_dict, train_data_dict['gender'], pos_vocab, word_vocab, 'bool')
    svm_predictions = svm_clf.predict(test_data_dict)
    get_results("SVM", test_data_dict['gender'], svm_predictions, target_name)

    svm_clf = SVMClassifier(train_data_dict, train_data_dict['gender'], pos_vocab, word_vocab, 'discrete')
    svm_predictions = svm_clf.predict(test_data_dict)
    get_results("SVM DISCRETE", test_data_dict['gender'], svm_predictions, target_name)

    svm_clf = SVMClassifier(train_data_dict, train_data_dict['gender'], pos_vocab, word_vocab, 'svc')
    svm_predictions = svm_clf.predict(test_data_dict)
    get_results("SVM SVC", test_data_dict['gender'], svm_predictions, target_name)

    ## SVM - Regression
    print("### SVM - Regression ###")
    svmr_clf = SVMRClassifier(train_data_dict, train_data_dict['gender'], pos_vocab, word_vocab, 'linearsvr')
    svmr_predictions = svmr_clf.predict(test_data_dict)
    predictions = []
    for prediction in svmr_predictions:
        if prediction >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)
    get_results("SVM-R LINEAR DEFAULT", test_data_dict['gender'], predictions, target_name)

    print("")

    end = time.time()
    print("Total Run Time = %fs" % (end - start))

    return 0

def characterClassification():
    start = time.time()

    ## Load the Training Set and Testing Set from numpy dict #####
    train_data_dict = LoadDataSet('../data/train_dict.npy')
    test_data_dict = LoadDataSet('../data/test_dict.npy')


    ## Load POS Patterns ##########################
    pos_vocab = []
    with open('../data/POSPatterns.txt') as file:
        for line in file:
            pos_vocab.append(line.strip('\n'))
    word_vocab = []


    ## Naive Bayes 
    print("### Naive Bayes ###")
    nb_clf = naiveBayesClassifier(train_data_dict, train_data_dict['character'], pos_vocab, word_vocab, 'tf')
    nb_predictions = nb_clf.predict(test_data_dict)
    get_results("Naive Bayes", test_data_dict['character'], nb_predictions)

    ## SVM 
    print("### SVM ###")
    svm_clf = SVMClassifier(train_data_dict, train_data_dict['character'], pos_vocab, word_vocab, 'bool')
    svm_predictions = svm_clf.predict(test_data_dict)
    get_results("SVM", test_data_dict['character'], svm_predictions)

    svm_clf = SVMClassifier(train_data_dict, train_data_dict['character'], pos_vocab, word_vocab, 'discrete')
    svm_predictions = svm_clf.predict(test_data_dict)
    get_results("SVM DISCRETE", test_data_dict['character'], svm_predictions)

    svm_clf = SVMClassifier(train_data_dict, train_data_dict['character'], pos_vocab, word_vocab, 'svc')
    svm_predictions = svm_clf.predict(test_data_dict)
    get_results("SVM SVC", test_data_dict['character'], svm_predictions)

    print("")

    end = time.time()
    print("Total Run Time = %fs" % (end - start))

    return 0


def get_results(model_name, y_true,y_pred, target_name=None):
    
    accuracy = accuracy_score(y_true,y_pred)
    print(classification_report(y_true, y_pred, target_names = target_name))

    confusion = confusion_matrix(y_true, y_pred)

    print("confusion matrix: ")
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print(' {} Accuracy: {:.2f}%'.format(model_name, accuracy*100))
    print('Mis-classification error for {} : {:.2f}%'.format(model_name, classification_error*100))

def LoadDataSet(path):
    data_dict = np.ndarray.tolist(np.load(path, allow_pickle=True))
    print(data_dict.keys())
    return data_dict

if __name__ == '__main__':
    genderClassification()
    characterClassification()







"""
(virt-py3tf) (base) nehas@lucy-MS-7B61:/home/projects/soft-goods/Project$ python GenderClassifier.py
dict_keys(['text', 'character', 'gender', 'text_norm', 'token_text_norm', 'POS', 'POS_tagged', 'f_measure', 'word_count', 'length', 'gf', 'fa', 'diag_act', 'LE_C', 'TS', 'mispelled', 'index'])
dict_keys(['text', 'character', 'gender', 'text_norm', 'token_text_norm', 'POS', 'POS_tagged', 'f_measure', 'word_count', 'length', 'gf', 'fa', 'diag_act', 'LE_C', 'TS', 'mispelled', 'index'])
### Naive Bayes ###
              precision    recall  f1-score   support

      Female       0.55      0.65      0.60       526
        Male       0.64      0.54      0.59       598

   micro avg       0.59      0.59      0.59      1124
   macro avg       0.60      0.59      0.59      1124
weighted avg       0.60      0.59      0.59      1124

confusion matrix:
[[340 186]
 [273 325]]
 Naive Bayes Accuracy: 59.16%
Mis-classification error for Naive Bayes : 40.84%
### SVM ###
              precision    recall  f1-score   support

      Female       0.57      0.61      0.59       526
        Male       0.63      0.59      0.61       598

   micro avg       0.60      0.60      0.60      1124
   macro avg       0.60      0.60      0.60      1124
weighted avg       0.60      0.60      0.60      1124

confusion matrix:
[[323 203]
 [247 351]]
 SVM Accuracy: 59.96%
Mis-classification error for SVM : 40.04%
              precision    recall  f1-score   support

      Female       0.56      0.62      0.59       526
        Male       0.63      0.58      0.60       598

   micro avg       0.60      0.60      0.60      1124
   macro avg       0.60      0.60      0.60      1124
weighted avg       0.60      0.60      0.60      1124

confusion matrix:
[[325 201]
 [252 346]]
 SVM DISCRETE Accuracy: 59.70%
Mis-classification error for SVM DISCRETE : 40.30%
/home/projects/virt-py3tf/lib/python3.7/site-packages/sklearn/externals/joblib/externals/loky/process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
  "timeout or by a memory leak.", UserWarning
              precision    recall  f1-score   support

      Female       0.55      0.59      0.57       526
        Male       0.61      0.58      0.60       598

   micro avg       0.58      0.58      0.58      1124
   macro avg       0.58      0.58      0.58      1124
weighted avg       0.58      0.58      0.58      1124

confusion matrix:
[[308 218]
 [252 346]]
 SVM SVC Accuracy: 58.19%
Mis-classification error for SVM SVC : 41.81%
### SVM - Regression ###
              precision    recall  f1-score   support

      Female       0.56      0.60      0.58       526
        Male       0.62      0.58      0.60       598

   micro avg       0.59      0.59      0.59      1124
   macro avg       0.59      0.59      0.59      1124
weighted avg       0.59      0.59      0.59      1124

confusion matrix:
[[318 208]
 [254 344]]
 SVM-R LINEAR DEFAULT Accuracy: 58.90%
Mis-classification error for SVM-R LINEAR DEFAULT : 41.10%

Total Run Time = 450.020319s
"""




"""
(virt-py3tf) (base) nehas@lucy-MS-7B61:/home/projects/soft-goods/Project$ python GenderClassifier.py
dict_keys(['text', 'character', 'gender', 'text_norm', 'token_text_norm', 'POS', 'POS_tagged', 'f_measure', 'word_count', 'length', 'gf', 'fa', 'diag_act', 'LE_C', 'TS', 'mispelled', 'index'])
dict_keys(['text', 'character', 'gender', 'text_norm', 'token_text_norm', 'POS', 'POS_tagged', 'f_measure', 'word_count', 'length', 'gf', 'fa', 'diag_act', 'LE_C', 'TS', 'mispelled', 'index'])
### Naive Bayes ###
/home/projects/virt-py3tf/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

     BRADLEY       1.00      0.05      0.09        41
   CHRISTIAN       0.00      0.00      0.00        46
       CLARE       0.00      0.00      0.00        31
       GARRY       0.00      0.00      0.00        48
     HEATHER       0.25      0.02      0.04        42
         IAN       0.16      0.17      0.16       101
        JACK       0.00      0.00      0.00        85
        JANE       0.50      0.11      0.17        76
         MAX       0.47      0.11      0.18        73
       MINTY       1.00      0.02      0.04        51
        PHIL       0.33      0.02      0.04        53
      RONNIE       0.20      0.02      0.04        52
        ROXY       0.00      0.00      0.00        56
        SEAN       0.00      0.00      0.00        63
     SHIRLEY       1.00      0.07      0.13        73
      STACEY       0.75      0.04      0.08        72
      STEVEN       0.00      0.00      0.00        37
       TANYA       0.12      0.96      0.22       124

   micro avg       0.15      0.15      0.15      1124
   macro avg       0.32      0.09      0.07      1124
weighted avg       0.32      0.15      0.09      1124

confusion matrix:
[[  2   0   0   0   0   4   0   0   0   0   0   0   0   0   0   0   0  35]
 [  0   0   0   0   0   9   0   1   2   0   0   0   0   0   0   0   0  34]
 [  0   0   0   0   0   5   0   0   0   0   0   0   0   0   0   0   0  26]
 [  0   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0  45]
 [  0   0   0   0   1   2   0   0   1   0   0   0   0   0   0   0   0  38]
 [  0   1   0   0   0  17   0   3   0   0   0   0   0   0   0   0   0  80]
 [  0   0   0   0   1   4   0   0   1   0   0   1   0   0   0   0   0  78]
 [  0   0   0   0   0  10   0   8   0   0   0   0   0   0   0   0   0  58]
 [  0   0   0   0   0   5   0   0   8   0   0   0   0   0   0   0   0  60]
 [  0   0   0   0   1   6   0   1   2   1   0   0   0   0   0   0   0  40]
 [  0   0   0   0   0   8   0   1   0   0   1   0   0   0   0   1   0  42]
 [  0   0   0   0   0   5   0   0   1   0   0   1   0   0   0   0   0  45]
 [  0   0   0   0   0   5   0   1   0   0   0   1   0   0   0   0   0  49]
 [  0   0   0   0   1   4   1   0   1   0   1   0   0   0   0   0   0  55]
 [  0   0   0   0   0   4   0   1   0   0   0   0   0   0   5   0   0  63]
 [  0   0   0   0   0   8   0   0   0   0   1   1   0   0   0   3   0  59]
 [  0   0   0   0   0   7   0   0   0   0   0   0   0   0   0   0   0  30]
 [  0   0   0   0   0   3   0   0   1   0   0   1   0   0   0   0   0 119]]
 Naive Bayes Accuracy: 14.77%
Mis-classification error for Naive Bayes : 0.00%
### SVM ###
/home/projects/virt-py3tf/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/projects/virt-py3tf/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
              precision    recall  f1-score   support

     BRADLEY       1.00      0.02      0.05        41
   CHRISTIAN       0.44      0.09      0.15        46
       CLARE       0.24      0.16      0.19        31
       GARRY       0.43      0.06      0.11        48
     HEATHER       0.36      0.33      0.35        42
         IAN       0.14      0.29      0.19       101
        JACK       0.23      0.09      0.13        85
        JANE       0.45      0.26      0.33        76
         MAX       0.27      0.25      0.26        73
       MINTY       0.50      0.14      0.22        51
        PHIL       0.26      0.09      0.14        53
      RONNIE       0.39      0.17      0.24        52
        ROXY       0.50      0.07      0.12        56
        SEAN       0.21      0.05      0.08        63
     SHIRLEY       0.34      0.15      0.21        73
      STACEY       0.28      0.15      0.20        72
      STEVEN       0.25      0.03      0.05        37
       TANYA       0.16      0.73      0.27       124

   micro avg       0.22      0.22      0.22      1124
   macro avg       0.36      0.17      0.18      1124
weighted avg       0.33      0.22      0.19      1124

confusion matrix:
[[ 1  1  1  0  0  7  0  1  1  0  1  1  0  1  2  2  2 20]
 [ 0  4  0  0  0 17  2  1  2  0  0  0  0  1  4  0  0 15]
 [ 0  0  5  0  0  6  1  0  2  0  0  0  0  1  1  1  0 14]
 [ 0  0  0  3  4  9  1  1  3  1  1  1  0  1  1  4  0 18]
 [ 0  0  0  0 14  5  1  0  2  1  1  0  0  1  1  1  0 15]
 [ 0  2  2  0  1 29  4  3  3  0  0  0  0  0  4  1  0 52]
 [ 0  0  1  1  1 14  8  2  5  0  0  4  1  1  0  5  0 42]
 [ 0  0  0  0  2 19  1 20  4  0  0  0  0  0  0  0  0 30]
 [ 0  1  3  1  0  9  2  1 18  1  1  1  0  1  1  4  0 29]
 [ 0  0  0  1  7  8  0  2  3  7  0  0  0  1  1  1  0 20]
 [ 0  0  0  0  3  9  4  1  3  2  5  1  0  0  3  1  0 21]
 [ 0  0  0  0  1  8  1  1  4  1  2  9  2  0  1  1  0 21]
 [ 0  0  0  0  1  9  3  3  2  0  1  2  4  0  1  0  0 30]
 [ 0  0  0  0  1  7  2  0  4  0  2  2  0  3  1  3  1 37]
 [ 0  0  2  0  3  9  3  3  1  1  3  0  0  1 11  4  0 32]
 [ 0  0  3  0  0  7  0  2  2  0  0  1  1  1  0 11  0 44]
 [ 0  1  0  1  1  8  2  1  4  0  0  0  0  0  0  0  1 18]
 [ 0  0  4  0  0 21  0  2  3  0  2  1  0  1  0  0  0 90]]
 SVM Accuracy: 21.62%
Mis-classification error for SVM : 16.67%
              precision    recall  f1-score   support

     BRADLEY       0.33      0.07      0.12        41
   CHRISTIAN       0.18      0.13      0.15        46
       CLARE       0.12      0.16      0.14        31
       GARRY       0.19      0.08      0.12        48
     HEATHER       0.26      0.33      0.29        42
         IAN       0.18      0.23      0.20       101
        JACK       0.20      0.12      0.15        85
        JANE       0.38      0.29      0.33        76
         MAX       0.30      0.34      0.32        73
       MINTY       0.33      0.22      0.26        51
        PHIL       0.23      0.21      0.22        53
      RONNIE       0.26      0.25      0.25        52
        ROXY       0.22      0.11      0.14        56
        SEAN       0.09      0.05      0.06        63
     SHIRLEY       0.23      0.18      0.20        73
      STACEY       0.17      0.18      0.17        72
      STEVEN       0.18      0.08      0.11        37
       TANYA       0.25      0.60      0.35       124

   micro avg       0.23      0.23      0.23      1124
   macro avg       0.23      0.20      0.20      1124
weighted avg       0.23      0.23      0.21      1124

confusion matrix:
[[ 3  2  2  2  0  6  0  2  0  0  0  1  0  2  3  2  1 15]
 [ 0  6  1  1  2 12  3  4  6  0  1  1  0  0  0  1  0  8]
 [ 0  1  5  0  0  1  1  0  3  0  1  0  1  0  5  6  0  7]
 [ 0  0  1  4  5  5  2  3  4  1  4  1  1  1  5  2  1  8]
 [ 0  0  2  1 14  2  1  2  2  4  1  1  0  1  2  4  0  5]
 [ 0  6  4  2  4 23  1  5  1  3  5  5  0  4  3  7  2 26]
 [ 0  3  1  3  1  7 10  2  8  2  4  8  4  4  2  9  2 15]
 [ 0  2  0  0  2 12  3 22  4  0  4  2  0  2  1  4  0 18]
 [ 0  1  4  0  1 10  2  0 25  3  2  0  1  3  1  3  0 17]
 [ 0  1  0  2  7  1  1  3  5 11  3  1  2  3  0  2  0  9]
 [ 0  1  2  0  4  7  1  2  0  3 11  3  3  2  5  2  0  7]
 [ 1  2  0  1  2  3  2  2  1  1  1 13  3  0  3  3  2 12]
 [ 0  0  1  0  1  5  4  3  2  1  2  5  6  1  4  5  0 16]
 [ 2  1  2  2  1  6  4  2 11  0  2  2  0  3  3  5  2 15]
 [ 1  1  5  1  4  7  1  1  5  3  1  2  1  2 13  8  1 16]
 [ 1  0  4  2  2  7  2  0  2  1  2  3  3  1  2 13  2 25]
 [ 0  2  2  0  1  7  3  2  0  0  1  0  2  1  2  2  3  9]
 [ 1  5  5  0  2 10  8  3  3  0  3  2  0  3  3  0  1 75]]
 SVM DISCRETE Accuracy: 23.13%
Mis-classification error for SVM DISCRETE : 18.18%
              precision    recall  f1-score   support

     BRADLEY       0.33      0.02      0.05        41
   CHRISTIAN       0.11      0.07      0.08        46
       CLARE       0.14      0.10      0.11        31
       GARRY       0.20      0.04      0.07        48
     HEATHER       0.27      0.24      0.25        42
         IAN       0.14      0.27      0.18       101
        JACK       0.19      0.12      0.15        85
        JANE       0.33      0.25      0.29        76
         MAX       0.26      0.30      0.28        73
       MINTY       0.43      0.12      0.18        51
        PHIL       0.25      0.15      0.19        53
      RONNIE       0.39      0.17      0.24        52
        ROXY       0.12      0.02      0.03        56
        SEAN       0.29      0.08      0.12        63
     SHIRLEY       0.34      0.15      0.21        73
      STACEY       0.19      0.12      0.15        72
      STEVEN       0.33      0.05      0.09        37
       TANYA       0.17      0.62      0.27       124

   micro avg       0.20      0.20      0.20      1124
   macro avg       0.25      0.16      0.16      1124
weighted avg       0.24      0.20      0.18      1124

confusion matrix:
[[ 1  1  1  1  0 10  1  2  4  0  0  1  0  0  2  0  1 16]
 [ 0  3  0  1  1 11  3  3  6  0  1  1  0  0  0  1  0 15]
 [ 0  0  3  0  0  8  1  1  3  0  0  0  0  0  1  0  0 14]
 [ 0  1  0  2  4  9  2  2  8  0  3  0  0  0  1  5  0 11]
 [ 0  0  1  0 10  5  2  3  2  2  0  0  0  0  1  2  0 14]
 [ 1  6  3  2  3 27  3  2  3  0  1  0  0  1  1  5  0 43]
 [ 0  3  1  0  1 13 10  0  4  0  5  3  2  2  0  3  0 38]
 [ 0  1  1  0  0 17  1 19  4  0  1  0  0  1  1  2  0 28]
 [ 0  1  1  0  3 12  4  2 22  1  1  0  0  1  1  0  0 24]
 [ 0  2  0  2  5  6  2  3  3  6  3  1  0  1  1  2  0 14]
 [ 0  1  1  0  2 11  2  2  4  0  8  0  0  0  3  1  0 18]
 [ 0  0  0  0  1  9  1  1  2  0  3  9  2  1  0  2  1 20]
 [ 0  0  1  1  1 10  6  3  1  1  0  3  1  0  3  1  1 23]
 [ 0  1  0  1  2  8  2  2  6  0  2  2  0  5  0  5  0 27]
 [ 1  1  1  0  2 12  1  4  3  3  1  1  2  2 11  5  0 23]
 [ 0  3  4  0  1  9  2  2  4  1  2  1  0  1  3  9  0 30]
 [ 0  0  1  0  0  8  2  3  2  0  1  1  1  0  0  1  2 15]
 [ 0  4  3  0  1 15  7  3  5  0  0  0  0  2  3  3  1 77]]
 SVM SVC Accuracy: 20.02%
Mis-classification error for SVM SVC : 20.00%
"""