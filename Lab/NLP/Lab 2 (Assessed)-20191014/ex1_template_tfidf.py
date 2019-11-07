# -*- coding: utf-8 -*-
"""
python3 Assignment2_NeerajVashistha_190573735.py
Author : Neeraj Vashistha (190573735)
"""

from __future__ import print_function  # needed for Python 2
from __future__ import division		# needed for Python 2
import csv,re,sys
import collections, itertools
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import nltk, math
from nltk.classify import SklearnClassifier
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from random import shuffle
from contractions import general_contraction
from contractions import digit_contractions

stopwords = stopwords.words('english')
porter = PorterStemmer()
# nltk.download('stopwords')



def loadData(path, Text=None):
	# load data from a file and append it to the rawData and preprocessedData
	with open(path,'r') as f:
		reader = csv.reader(f, delimiter='\t')
		# reader.next() # ignore header python 2.7
		next(reader) # ignore header python 3.x
		for line in reader:
			(Id, Text, Rating, VerPurchase, Label) = parseReview(line)
			rawData.append((Id, Text, Rating, VerPurchase, Label))
			preprocessedData.append((Id, preProcess(Text), Rating, VerPurchase, Label))


		
def splitData(percentage):
	dataSamples = len(rawData)
	halfOfData = int(len(rawData)/2)
	trainingSamples = int((percentage*dataSamples)/2)
	for (index, Text,_,_, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
		trainData.append((toFeatureVector(preProcess(Text),index),Label))
	for (index, Text,_,_, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
		testData.append((toFeatureVector(preProcess(Text),index),Label))

# QUESTION 1
# Convert line from input file into an id/text/label tuple
def parseReview(reviewLine):
	"""
	reviewLine is a list
	# Q1 returns a tuple with doc_id, review_text, raiting, verPurchase ,label
	"""
	doc_id = int(reviewLine[0])
	label = reviewLine[1]
	review_text = reviewLine[8]
	raiting = reviewLine[2]
	verPurchase = reviewLine[3]
	return (doc_id, review_text, raiting, verPurchase ,label)

# TEXT PREPROCESSING AND FEATURE VECTORIZATION
# Input: a string of one review
def preProcess(text):
	"""
	return a processed text in token form 
	"""
	text = text.lower() # lower case the text
	# Q4 replace the word with expanded contractions
	for k,v in general_contraction.items():
		if k in text.split():
			text = text.replace(k,v)
	# Q4 remove speacial char including all puncuattions and replace it with a space
	text = re.sub('[^A-Za-z0-9]+',' ',text) 
	# tokenise
	tokens = text.split()
	# stop word removal
	tokens = [w for w in tokens if w not in stopwords ]
	# Q4 Stemming
	tokens = [str(porter.stem(w)) for w in tokens]
	# if word is non-english return its english form # too much time-complexity
	# tokens = [porter.stem(w) if porter.stem(w) in set(words.words()) else w for w in tokens ]
	# for words having digits such as 12gb, 1st, etc expanding the token list
	for k in tokens:
		if len(k) >2 and re.match(r'[0-9]+',k):			
			if len(k) >2 and not k.isdigit():
				l = re.split(r'(\d+)',k)
				l = [w for w in l if w is not '' ]
				if l and len(l) <= 3:
					for i in l:
						if i in digit_contractions.keys():
							l = list(map(lambda b: b.replace(i,digit_contractions[i]), l))
					tokens.remove(k)
					tokens = tokens+l
				else:
					tokens.remove(k)
	for k,v in digit_contractions.items():
		if k in tokens:
			if tokens[tokens.index(k)-1].isdigit():	
				tokens = list(map(lambda b: b.replace(k,v), tokens))
	# remove tokens of size less than 2
	tokens = [t for t in tokens if len(t) > 2]
	return tokens

print("Pre-processing Data")
# Q2 Creating a global dict - featureDict
rawData = []
preprocessedData = []
loadData('amazon_reviews.txt')
counterDict = {} # a temp dict to store of each word count
alltokens =[]
for i in preprocessedData:
	alltokens.extend(i[1])

counterDict = collections.Counter(alltokens)
mostCommonWord = counterDict.most_common(10)
print('Most Common word',mostCommonWord)

# if mostCommonWord[0][0] == 'br':
# 	print('adding '+mostCommonWord[0][0]+' to stopwords and removing it')
# 	stopwords.append('br')
# 	counterDict.pop('br')

# QUESTION 2
featureDict = {} # A global dictionary of features
total_no_of_clean_words = sum(counterDict.values())
total_no_of_reviews = len(rawData)
print("Total Number of words: "+str(total_no_of_clean_words))
print("Total Number of reviews: "+str(total_no_of_reviews))
print("Building feature dict")
# Q2 - Q4 calculate TfIDF 
for k,v in counterDict.items():
	featureDict[k] = (float(v)/total_no_of_clean_words)*math.log(float(total_no_of_reviews) / v)


length_of_token = []
for i in preprocessedData:
	length_of_token.append(len(i[1]))

mean_token = int(np.mean(length_of_token))
median_token = int(np.median(length_of_token))
print("Avg length of tokens per review: ", mean_token)
print("Median length of tokens per review: ",median_token)



def toFeatureVector(tokens,index=None):
	# Should return a dictionary containing features as keys, and weights as values
	"""
	index to map back tokens to rawData and fetch rating and verfied purchase values.
	return a dict of the input tokens
	dict contains keys with value of: 
		token's TfIDF		# Q4 TfIDF values really imporved model preformance 
							# Q4 from .50-.65 to .75-.80 precsion and recall values
		raiting 			# Q5 after normalising the values, the performace metrics
		verified purchase   # Q5 was stabilised efficiently and was around 77-78% acc, 
		# below features add addition meaning to feature vect and improved performace metric.
		Average word length 
		Number of stopwords
		Number of special characters
		Number of numerics
	"""
	adict = {}
	tokens = [w for w in tokens if w not in stopwords]
	# Q4 Limiting the token list to average/median of all the tokens per reviews
	for i in tokens[:mean_token]: 
		adict[i] = featureDict[i]
	if index is not None:
		for i in rawData:
			if i[0] == index:
				adict['raiting'] = float(int(i[2]) - 0)/5
				adict['verPur'] = 1 if i[3] == 'Y' else 0
				adict['avgWordLen'] = sum(len(w) for w in i[1].split())/len(i[1])
				adict['stopwords'] = len([w for w in i[1].split() if w in stopwords])
				# adict['speacialChar'] = len(re.findall(r'[^A-Z0-9a-z ]+',i[1])) # performace metrics decreases
				adict['digits'] = len(re.findall(r'[0-9]+',i[1]))
	return adict

# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
	print("Training Classifier...")
	pipeline =  Pipeline([('svc', LinearSVC(penalty='l1', dual=False))])
	return SklearnClassifier(pipeline).train(trainData)

# QUESTION 3

def crossValidate(dataset, folds):
	"""
	Performs @folds cross-validation on dataset 
	by dividing dataset into equal foldsize

	"""
	shuffle(dataset)
	cv_results = []
	precision_recall_acc = []
	foldSize = int(len(dataset)/folds)
	for i in range(0,len(dataset),foldSize):
		# preparing data
		valD = dataset[i:i+foldSize]
		testD = dataset[:i]+dataset[i+foldSize:] #list(set(dataset)-set(dataset[i:i+foldSize]))
		# Training
		print("*"*60)
		print("Training on data-set size "+str(len(testD))+" of batch "+str(i/(foldSize)))
		classi = trainClassifier(testD)
		# Prediction on validation data 
		print("Predicting on heldout data-set size..."+str(len(valD))+" of batch "+str(i/(foldSize)))
		y_true = list(map(lambda t: t[1], valD))
		y_pred = predictLabels(valD,classi)		
		# Performance Metrics		
		# average based on macro as it calculate metrics for each label, and find their unweighted mean.
		precision_recall = list(precision_recall_fscore_support(y_true, y_pred, average='macro'))
		acc = accuracy_score(y_true,y_pred)
		precision_recall[-1] = acc
		print(precision_recall)
		precision_recall_acc.append(precision_recall)
	df = pd.DataFrame(precision_recall_acc,columns = ["Precision","Recall","F1 score","Accuracy Score"])
	print(df)
	cv_results = df.mean().tolist()
	return cv_results


# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(reviewSamples, classifier):
	# modified since tokens are proceesed in splitData() func, only prediction here 
	return classifier.classify_many(list(map(lambda t: t[0], reviewSamples)))



def predictLabel(reviewSample, classifier):
	return classifier.classify(toFeatureVector(preProcess(reviewSample)))

# MAIN

# loading reviews
rawData = []		  # the filtered data from the dataset file (should be 21000 samples)
preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
trainData = []		# the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
testData = []		 # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# the output classes
fakeLabel = 'fake'
realLabel = 'real'

# references to the data files
reviewPath = 'amazon_reviews.txt'

## Do the actual stuff
# We parse the dataset and put it in a raw data list
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
	  "Preparing the dataset...",sep='\n')
loadData(reviewPath) 
# We split the raw dataset into a set of training data and a set of test data (80/20)
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
	  "Preparing training and test data...",sep='\n')
splitData(0.8)
# We print the number of training samples and the number of features
print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
	  "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')

precision,recall,f1,accuracy = crossValidate(trainData, 10)
print("*"*60)
print("Average Performance Metrics on Training Data...")
print('precision',precision)
print('recall',recall)
print('f1',f1)
print('accuracy',accuracy)


'''
Pre-processing Data
Most Common word [('use', 8774), ('one', 7134), ('great', 6628), ('like', 6365), ('work', 5482), ('would', 5317), ('good', 5311), ('love', 4895), ('get', 4868), ('look', 4846)]
Total Number of words: 699377
Total Number of reviews: 21000
Building feature dict
Avg length of tokens per review:  33
Median length of tokens per review:  21
Now 0 rawData, 0 trainData, 0 testData
Preparing the dataset...
Now 21000 rawData, 0 trainData, 0 testData
Preparing training and test data...
Now 21000 rawData, 16800 trainData, 4200 testData
Training Samples: 
16800
Features: 
22676
************************************************************
Training on data-set size 15120 of batch 0.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 0.0
************************************************************
Training on data-set size 15120 of batch 1.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 1.0
************************************************************
Training on data-set size 15120 of batch 2.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 2.0
************************************************************
Training on data-set size 15120 of batch 3.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 3.0
************************************************************
Training on data-set size 15120 of batch 4.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 4.0
************************************************************
Training on data-set size 15120 of batch 5.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 5.0
************************************************************
Training on data-set size 15120 of batch 6.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 6.0
************************************************************
Training on data-set size 15120 of batch 7.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 7.0
************************************************************
Training on data-set size 15120 of batch 8.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 8.0
************************************************************
Training on data-set size 15120 of batch 9.0
Training Classifier...
Predicting on heldout data-set size...1680 of batch 9.0
   Precision    Recall  F1 score  Accuracy Score
0   0.799984  0.795036  0.793868        0.794643
1   0.795758  0.791239  0.790770        0.791667
2   0.789131  0.784936  0.783320        0.783929
3   0.765832  0.761759  0.760953        0.761905
4   0.783324  0.777439  0.777509        0.779167
5   0.807950  0.803063  0.803111        0.804167
6   0.788004  0.783037  0.779793        0.780357
7   0.793717  0.788078  0.785041        0.785714
8   0.799286  0.791538  0.792338        0.794643
9   0.798544  0.792088  0.790122        0.791071
************************************************************
Average Performance Metrics on Training Data...
precision 0.7921531428294636
recall 0.7868214239665258
f1 0.7856826269105766
accuracy 0.7867261904761904
(virt-py3tf) nv@nv-workstation:

'''