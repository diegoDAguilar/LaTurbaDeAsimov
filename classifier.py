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

des_list = []  
des_array = np.zeros((1,32))

'''Main'''
# Reading training set
des_list, des_array = extractor(node_directory, des_list, des_array, cod_op)
des_array = np.delete(des_array,0,0)
# Loading the Bag of Words
BoW = joblib.load('BoW.pkl')
#Codifying the training set
new_data = codec_test(BoW, clusters=30,des_list,n=0)
clf=joblib.load('classifier.pkl')
result=clf.predict(new_data)

