import numpy as np
import os
import time
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

def load_data(k):
	# File that contain trained model 
	data_npy_phoneme_01 = 'data/GMM_params_phoneme_01_k_0'+str(k)+'.npy'
	data_npy_phoneme_02 = 'data/GMM_params_phoneme_02_k_0'+str(k)+'.npy'
	

	# Loading data from .npy file
	model_phoneme_01 = np.ndarray.tolist(np.load(data_npy_phoneme_01, allow_pickle=True))
	model_phoneme_02 = np.ndarray.tolist(np.load(data_npy_phoneme_02, allow_pickle=True))
	return model_phoneme_01, model_phoneme_02


def get_pred(X, k, model1_weights, model2_weights):
	predClass = []
	N = X.shape[0]
	Z01 = np.zeros((N,k)) # shape Nxk
	Z02 = np.zeros((N,k)) # shape Nxk
	
	# get predictions on X from model1
	Z01 = get_predictions(model1_weights['mu'], model1_weights['s'], model1_weights['p'], X)
	# Z01 = normalize(Z01, axis=1, norm='l1')
	Z01 = Z01.astype(np.float32)
	Z01Sum = np.sum(Z01,axis=1)

	# get predictions on X from model2 
	Z02 = get_predictions(model2_weights['mu'], model2_weights['s'], model2_weights['p'], X)
	# Z02 = normalize(Z02, axis=1, norm='l1')
	Z02 = Z02.astype(np.float32)
	Z02Sum = np.sum(Z02,axis=1)

	# if sum of probabilities of any model is less than other mark it
	for z1,z2 in zip(Z01Sum,Z02Sum):
		if z1 > z2:
			predClass.append(0.0)
		else:
			predClass.append(1.0)

	y_pred = np.array(predClass)
	return y_pred

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:,0] = f1
X_full[:,1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 6

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

X_phonemes_1_2 = np.zeros((np.sum(phoneme_id==2)+np.sum(phoneme_id==1), 2))
X_phonemes_1_2 = np.concatenate((X_full[phoneme_id==1,:],X_full[phoneme_id==2,:]), axis=0)
# # X_phonemes_1_2 = 
y_true = np.concatenate((np.zeros((np.sum(phoneme_id==1))),np.ones((np.sum(phoneme_id==2)))),axis =0)

# X_phonemes_1_2 = np.zeros((np.sum(phoneme_id==1), 2))
# X_phonemes_1_2 = np.append(X_full[phoneme_id==1,:],X_full[phoneme_id==2,:],axis =1)

print(np.sum(phoneme_id==2))

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)

# time.sleep(5)
#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

########################################/

X = X_phonemes_1_2.copy()
# get number of samples
N = X.shape[0]
# get dimensionality of our dataset
D = X.shape[1]

model_phoneme_01, model_phoneme_02 = load_data(k)
y_pred = get_pred(X, k, model_phoneme_01, model_phoneme_02)

# print(y_pred)
accuracy = accuracy_score(y_true,y_pred)

target_names = ['phoneme 1', 'phoneme 2']
report = print(classification_report(y_true, y_pred, target_names=target_names))

confusion = metrics.confusion_matrix(y_true, y_pred)

print("confusion matrix: ")
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy*100))
print('Mis-classification error using GMMs with {} components: {:.2f}%'.format(k, classification_error*100))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()