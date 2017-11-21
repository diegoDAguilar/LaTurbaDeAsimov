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
des_list, des_array = extractor(node_directory, des_list, des_array, cod_op = 1)



clf2 = joblib.load('classifier.pkl')
