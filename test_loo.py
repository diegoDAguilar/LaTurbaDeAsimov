# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:12:06 2017

@author: Andres
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut


# Read training set
training_set = np.load('training_set.npz')
clusters = training_set['cluster']
training_set = training_set['hist_matrix']

#Parameters
histograms = training_set[:,0:clusters]
labels = training_set[:,clusters]
test_size=len(histograms) 
tp=0 # True-positives


#Initialize leave-one-out
loo = LeaveOneOut()

for train_index, test_index in loo.split(histograms):
    #Initialize classifier
    clf=LinearSVC()
    
    #Train classifier
    histogram_train = histograms[train_index] 
    labels_train = labels[train_index]
    clf.fit(histogram_train, labels_train) 

    #Test classifier    
    histogram_test = histograms[test_index]
    labels_test = labels[test_index]
    result = clf.predict(histogram_test)
    
    if (result == labels_test):
        tp+=1
    else:
        print test_index
        print 'FALLO -> resultado:', result, 'label:', labels_test

# Calculate accuracy       
accuracy = tp*100/test_size
print 'Precisión:',accuracy,'%'



