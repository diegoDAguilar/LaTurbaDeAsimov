import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os


des_list=[]
des_array=[]
def extractor(directory, nodo, sizeX = 800, sizeY = 400):
    global des_list
    global des_array
    orb = cv2.ORB_create()
    image_list = os.listdir(directory)
    des_array=np.zeros((1,32))
    #supermatriz 
    for i in range(len(image_list)) :
        #Reads image and sets its size to preset
        file_name = directory + image_list[i]
        img = cv2.imread(file_name)
        print(file_name)
        img = cv2.resize(img, (sizeX, sizeY))
    
        #Generates a gray image 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        des_list.append(des)
        des_array=np.vstack((des_array,des))

def codec(clusters,des_matrix,des_list,n_images_per_node):
    #clusters: Number of clusters
    #des_matrix: Descriptors matrix (generated by extractor)
    #des_list: List of matrices, each matrix (500x32) will contain the descriptors of each image (generated by extractor)
    
    #Creation of the bag of words
    kmeans=KMeans(n_clusters=clusters,random_state=0).fit(des_matrix)
    labels=kmeans.labels_
    
    #Codification of the images using the words
    training_data_array=np.zeros((1,clusters+1))
    count=0
    node_class=1
    for f in des_list:    #Aquí necesito que la función de Diego tambien bote una lista
        training_data=np.zeros((1,clusters))
        labels=kmeans.predict(f)
        for j in labels:
            training_data[0,j]+=1
        if count==n_images_per_node:
            count=0
            node_class+=1
            training_data=np.append(training_data,np.array([node_class]))
        else:
            training_data=np.append(training_data,np.array([node_class]))
        count+=1
        training_data_array=np.vstack((training_data_array,training_data))
    training_data_array=np.delete(training_data_array,0,0)
    #Function returns a matrix with the codification or "histogram" of every image.
    #Dimension of the matrix: number_of_images x number_of_clusters
    #every row of the matrix represent an image codificated
    return training_data_array

#Takes one argument: number of nodes
def main(argv):
    
    
    working_dir = os.getcwd()
    print(working_dir)
    for i in range(1,int(sys.argv[1]) + 1):
        extractor(directory = working_dir + '\Dataset\Nodo' + str(i) + '\\', nodo='Nodo' + str(i) + '_')
    codec(clusters=8,des_matrix=des_array,des_list=des_list,n_images_per_node=70)
    #tratamiento de la supermatriz
    #

if __name__ == "__main__":
    main(sys.argv) 


