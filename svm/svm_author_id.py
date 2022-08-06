#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
#import SVC
from sklearn.svm import SVC

#create classifier
#clf = SVC(kernel='linear')
clf = SVC(C=10000, kernel='rbf')

#reduce traing dataset to speed up algorithm
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]


#fit the classifier
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

#make predictions
t0 = time()
pred = clf.predict(features_test)

# Quiz: What class does the SVM (0 or 1, corresponding to Sara 
# and Chris respectively) predict for element 10 of the test set?
# The 26th? The 50th?
# Use the RBF kernel, C=10000 and 1% of the training set
# NB: assume zero-indexed list, i.e., element #100 -> [100]

answer10 = pred[10]
answer26 = pred[26]
answer50 = pred[50]
print('The prediction of the 10th, 26th and 50th element of the test set are:')
print(answer10, ', ', answer26, ', ', answer50, ' respectively')
print('Predicting Time:', round(time()-t0, 3), 's')

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print(accuracy)

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
