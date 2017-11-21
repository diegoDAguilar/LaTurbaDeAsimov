# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:51:55 2017

@author: Andres
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

'''Main'''
# Read training set
#testfile = np.load(outfile)
histogram = np.array([[1, 2], [3, 4], [5,6], [6,7]])
labels = np.array([1, 2, 2, 1])

#Train classifier
clf = fulltrain (histogram, labels)

#Save classifier

joblib.dump(clf, 'classifier.pkl')
clf2 = joblib.load('classifier.pkl')
