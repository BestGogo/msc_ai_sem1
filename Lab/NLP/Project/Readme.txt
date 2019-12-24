The Directory Structure of this project is as below

.
+-- data
|   +-- training.csv
|   +-- train_dict.csv
|   +-- train.csv
|   +-- test_dict.csv
|   +-- test_dict.csv
|   +-- test_gender.csv
|   +-- test_character.csv
|   +-- train_gender.csv
|   +-- train_character.csv
|   +-- pos_dict.npy
|   +-- POSPatterns.txt
+-- model
|   +-- model.crf.tagger
+-- src
|   +-- modeules
|	|	+-- classifier
|	|	+-- factor_analysis.py	
|   +-- data_exploration.ipynb
|   +-- FeatureBuilder.ipynb
|   +-- FeatureBuilder.py
|   +-- MainClassifier.py
+-- Readme.txt

To Create Features
	$ cd src
	$ python FeatureBuilder.py

To build model
	$ cd src
	$ python MainClassifier.py


The FeatureBuilder.py file takes input from data/training.csv and data/test.csv and generates data/train_dict.npy and data/test_dict.npy which is used as input for MainClassifier.py

The src/modules/classifier.py implements all the required classifier, the MainClassifier.py invokes all the classifer implemented in it.