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
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

#Cargado de las 70 imágenes que se encuentran en el directorio raíz
raiz="Nodo"
numero_nodos=3
img_list=[]
for g in range(1,numero_nodos+1):    
    for i in range(0,70):
        file_name=raiz+str(g)+"_Img"+str(i+1)+".png"
        img=cv2.imread(file_name,0)
        img = cv2.resize(img,(800,400))
        img_list.append(img)

#Detección y cálculo de puntos ORB en las imágenes
des_array=np.zeros((1,32))
des_list=[]
orb=cv2.ORB_create()
for p in img_list:
    kp,des=orb.detectAndCompute(p,None)
    des_list.append(des)
    des_array=np.vstack((des_array,des))
    
des_array=np.delete(des_array,0,0)

#Clustering KMeans
clusters=8
kmeans=KMeans(n_clusters=clusters,random_state=0).fit(des_array)
labels=kmeans.labels_

#Cálculo de histogramas y generación del training set
training_data_array=np.zeros((1,clusters+1))
count=0
clase_nodo=1
for f in des_list:
    training_data=np.zeros((1,clusters))
    labels=kmeans.predict(f)
    for j in labels:
        training_data[0,j]+=1
    
    if count==70:
        count=0
        clase_nodo+=1
        training_data=np.append(training_data,np.array([clase_nodo]))
    else:
        training_data=np.append(training_data,np.array([clase_nodo]))
    
    count+=1
    training_data_array=np.vstack((training_data_array,training_data))

training_data_array=np.delete(training_data_array,0,0)

#Entrenamiento del clasificador SVM
clf=LinearSVC()
clf.fit(training_data_array[:,0:clusters],training_data_array[:,clusters])

#Testeo del clasificador con dos imágenes imagen

img2=cv2.imread("imagen testeo real 9.png",0)
img2=cv2.resize(img2,(800,400))
kp2,des2=orb.detectAndCompute(img2,None)
labels2=kmeans.predict(des2)
training_data2=np.zeros((1,clusters))
for j in labels2:
    training_data2[0,j]+=1

print(clf.predict(training_data2))


"""
img2=np.array([])
img2=cv2.drawKeypoints(img,kp,img2,color=(0,255,0),flags=0)

plt.imshow(img2),plt.show()

number_cluster=4

ret=KMeans(n_clusters=number_cluster).fit_predict(des)
print ret

_,ax=plt.subplots(2)
ax[0].scatter(des[:,0],des[:,1])
ax[0].set_title("Initial Scatter Distribution")
ax[1].scatter(des[:,0],des[:,1],c=ret)
ax[1].set_title("Colored Partition denoting Clusters")

plt.show()
a,b,c=plt.hist(ret.ravel(),number_cluster,range=(0,number_cluster))


"""







