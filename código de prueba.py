# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:43:10 2017

@author: HP
"""

"""Detection of feature points ORB"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC

img=cv2.imread("nodo1img2.png",0)
orb=cv2.ORB_create()

kp,des=orb.detectAndCompute(img,None)

img2=np.array([])
img2=cv2.drawKeypoints(img,kp,img2,color=(0,255,0),flags=0)

plt.imshow(img2),plt.show()

number_cluster=50

ret=KMeans(n_clusters=number_cluster).fit_predict(des)
print ret

_,ax=plt.subplots(2)
ax[0].scatter(des[:,0],des[:,1])
ax[0].set_title("Initial Scatter Distribution")
ax[1].scatter(des[:,0],des[:,1],c=ret)
ax[1].set_title("Colored Partition denoting Clusters")

plt.show()
a,b,c=plt.hist(ret.ravel(),number_cluster,range=(0,number_cluster))