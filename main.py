import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os



def extractor(directory, nodo, sizeX = 800, sizeY = 400):
    
    orb = cv2.ORB_create()
    image_list = os.listdir(directory)
    
    for i in range(len(image_list)) :
        #Reads image and sets its size to preset
        file_name = directory + image_list[i]
        img = cv2.imread(file_name)
        print(file_name)
        img = cv2.resize(img, (sizeX, sizeY))
    
        #Generates a gray image 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        kp = orb.detect(img,None)
        kp, des = orb.compute(img, kp)
    
        img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    
        
        #stores image with the orb markers in a new file
        histograms_directory = os.getcwd() + '\Histogram'
        if not os.path.exists(histograms_directory):
            os.makedirs(histograms_directory)
        save_directory = os.getcwd() + '\Histogram\\' + nodo + str(i) + '.png'
        #print(save_directory)
        cv2.imwrite(save_directory, img2)
    

#Takes one argument: number of nodes
def main(argv):
    
    
    working_dir = os.getcwd()
    print(working_dir)
    for i in range(1,int(sys.argv[1]) + 1):
        extractor(directory = working_dir + '\Dataset\Nodo' + str(i) + '\\', nodo='Nodo' + str(i) + '_')
        

if __name__ == "__main__":
    main(sys.argv)
    
