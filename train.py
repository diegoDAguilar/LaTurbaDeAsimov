"""
Training & Preprocessing code

@Team: La turba de Asimov
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# Global parameters
n_clusters = 8          #Number of cluster 
images_per_node = 70    #Image dataset size

#============================================================================================================
#  Method :       extractor
#
#  @brief         Image descriptor generator
#     
#  @param         directory         - Numero de clusters en el histograma
#  @param         des_list          - List of matrices, each matrix will contain
#                                     the descriptors of each image
#  @param         des_array         - Single matrix containing every matrix
#                                     from des_list.
#  @param         sizeX, sizeY      - Image size (default 800x400)
#  @param         cod_op            - Operation code: 0-Training, 1-Single pic
#                                     (default 0)
# 
#  @return        des_matrix        - Descriptors matrix updated
#  @return        des_list          - List of matrices updated, each matrix will 
#                                     contain the descriptors of each image 
#===========================================================================================================
def extractor(directory, des_list, des_array, sizeX = 800, sizeY = 400, cod_op = 0):
    if cod_op == 0:
        #Initializations
        image_list = os.listdir(directory)
        #Extracts features from images
        for i in range(len(image_list)) :
            #Reads image and sets its size to preset
            file_name = directory + image_list[i]
            des = exOrb(file_name, sizeX, sizeY)
            des_list.append(des)
            des_array=np.vstack((des_array,des))
            
        return des_list, des_array
            
    elif cod_op == 1:
        image_list = os.listdir(directory)
        file_name = directory+image_list
        des = exOrb(file_name, sizeX, sizeY)
        return des
        
#============================================================================================================
#  Method :       exOrb
#
#  @brief         Auxiliary method. Executes ORB
#     
#  @param         pic_name          - Path to picture
#  @param         sizeX, sizeY      - Image size 
# 
#  @return        des               - Single image feature descriptors
#===========================================================================================================
def exOrb(pic_name, sizeX, sizeY):
        orb = cv2.ORB_create()
        img = cv2.imread(pic_name)
        img = cv2.resize(img, (sizeX, sizeY))
        
        #Crea la imagen en escala de grises
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        # Encuentra los puntos claves con ORB
        kp = orb.detect(img,None)
        # Devuelve los descriptores de cada punto clave
        kp, des = orb.compute(img, kp)
        return des
      
#============================================================================================================
#  Method :       codec
#
#  @brief         Genera el 'back of words' y los histogramas de cada imagen codificados con el BoW
#     
#  @param         clusters              - Numero de clusters en el histograma
#  @param         des_matrix            - Descriptors matrix 
#  @param         des_list              - List of matrices, each matrix (500x32) will 
#                                         contain the descriptors of each image
#  @param         n_images_per_node     - Imagenes por nodo en el dataset
# 
#  @return        training_data_array   - Histogram + labels training set
#                                         dimensions: img_numb x (clusters_numb + node_label)
#  @return        kmeans                - Bag of Words codification   
#===========================================================================================================
def codec(clusters, des_matrix, des_list, n_images_per_node):

    training_data_array = []
    
    #Creation of the bag of words
    kmeans = KMeans(n_clusters=clusters,random_state=0).fit(des_matrix)
    labels = kmeans.labels_
    
    #Codification of the images using the words
    training_data_array = np.zeros((1,clusters+1))
    count=0
    node_class = 1
    for f in des_list: 
        training_data = np.zeros((1,clusters))
        labels = kmeans.predict(f)
        for j in labels:
            training_data[0,j]+=1            
        if count == n_images_per_node:
            count = 0
            node_class += 1          
        training_data = np.append(training_data,np.array([node_class]))
        training_data_array = np.vstack((training_data_array,training_data))
        count+=1
    
    training_data_array = np.delete(training_data_array,0,0)
    return training_data_array, kmeans

#============================================================================================================
#  Method :       codec_test
#
#  @brief         Codifica las imagenes de testeo
#
#  @param         BoW                   - Bolsa de palabras (Clasificador KMeans entrenado)
#  @param         clusters              - Numero de clusters en el histograma
#  @param         descriptors           - Descriptores de la imagen 
#  @param         n                     - n=0:Test con varias imagenes / n=1: Test con una imagen
#===========================================================================================================

def codec_test(BoW, clusters,descriptors,n=0):
    #Codification of the images using the words
    
    if n == 0:
        test_data_array = np.zeros((1,clusters))
        for f in descriptors: 
            test_data = np.zeros((1,clusters))
            labels = BoW.predict(f)
            for j in labels:
                test_data[0,j]+=1            
            test_data_array = np.vstack((test_data_array,test_data))
        
        test_data_array = np.delete(test_data_array,0,0)
        return test_data_array
    
    elif n == 1:
        test_data = np.zeros((1,clusters))
        labels = BoW.predict(descriptors)
        for j in labels:
            test_data[0,j]+=1
        return test_data

#============================================================================================================
#  Method :       classifier
#
#  @brief         Entrenamiento del clasificador
#     
#  @param         training_set     - Histograma de entrenamiento + Etiquetas
#  @param         clusters         - Numero de clusters en el histograma
# 
#  @return        clf              - Clasificador entrenado
#===========================================================================================================
def classifier(training_set, clusters):
    clf = LinearSVC()
    clf.fit(training_set[:,0:clusters],training_set[:,clusters])
    return clf
    
#============================================================================================================
#  MAIN
#  @argv: number of nodes
#============================================================================================================   
def main(argv):
    
    #Parameters
    global n_clusters
    global images_per_node

    des_list = []  
    des_array = np.zeros((1,32))
    
    #Get directory
    working_dir = os.getcwd()
    print(working_dir)
    
    #Extracting features from dataset
    for i in range(1,int(sys.argv[1]) + 1):
        node_directory = working_dir + '\Dataset\Nodo' + str(i) + '\\'
        des_list, des_array = extractor(node_directory, des_list, des_array)
    des_array = np.delete(des_array,0,0) #Remove the zeros inicialization
    #Test extractor for one pic
    #node_directory = 'vlcsnap-2017-11-14-17h55m48s263.png'
    #des_list, des_array = extractor(node_directory, des_list, des_array, cod_op = 1)
    print('Extractor finished')
    
    #Codifying (BoW) and generating histograms
    hist_matrix, BoW = codec(n_clusters, des_array, des_list, images_per_node)
    #Save Bag od Words codification
    joblib.dump(BoW, 'BoW.pkl')
    #Save training set
    np.savez('training_set', hist_matrix = hist_matrix, cluster = n_clusters)
    print('Codec finished')  
    
    #Training classifier
    clf = classifier(hist_matrix, n_clusters)
    joblib.dump(clf, 'classifier.pkl')
    print('Classifier saved')

if __name__ == "__main__":
    main(sys.argv) 
