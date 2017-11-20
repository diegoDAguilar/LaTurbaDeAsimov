# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:12:06 2017

@author: Andres
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut


# Read training set
#testfile = np.load(outfile)
histogram = np.array([[1, 2], [3, 4], [5,6], [6,7]])
labels = np.array([1, 2, 2, 1])

#Initialize leave-one-out
loo = LeaveOneOut()
test_size=len(histogram) #Innecesario pero lo dejo por ahora


for train_index, test_index in loo.split(histogram):
    #Initialize classifier
    clf=SVC()
    
    #Train classifier
    histogram_train = histogram[train_index] 
    labels_train = labels[train_index]
    clf.fit(histogram_train, labels_train) 
    
    #Test classifier    
    histogram_test = histogram[test_index]
    labels_test = labels[test_index]
    result = clf.predict(histogram_test)
    if (result == labels_test):
        test_size-=1
    



