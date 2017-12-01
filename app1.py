# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:57:04 2017

@author: Andres
"""
import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import time
import train

#Global parameters
detection_time=12
filt_delay=7
#buf_len=15

def classify_image(img, BoW, clusters, clf):
    #Extract features
    des = train.extractOrb(img)
    #Generating histrogram
    histogram = train.generate_hist(BoW, clusters, des)
    #Classifying
    label = clf.predict(histogram)
    return label

def main(argv):
    
    #Get directory
    working_dir = os.getcwd()
    print(working_dir)
    
    if sys.argv[1] == 'image':
        directory = working_dir + '\Test2\Static\\'
    elif sys.argv[1] == 'video':
        directory = working_dir + '\Test2\Dynamic\\' 
        
    #Initializations
    BoW = joblib.load('Classifier/BoW.pkl')
    clf = joblib.load('Classifier/classifier.pkl')
    clusters = np.load('Classifier/training_set.npz')
    clusters = clusters['cluster']
    font = cv2.FONT_HERSHEY_SIMPLEX #Image text font
    text_print=[': Unknown','1: Escalera','2: Zona de estudio','3: Pasillo','4: Laboratorio  ', '5: Sala OpenInnovation','6: Conserjeria', '7: Aula 3202']
    
    #Static image test
    if sys.argv[1] == 'image':
        print 'Running static test'
        image_list = os.listdir(directory)  
        for i in range(len(image_list)) :
            # Reading test set
            file_name = directory + image_list[i]
            img = cv2.imread(file_name)
            label_frame = classify_image(img, BoW, clusters, clf) 
            
            #print "Etiqueta: ",label
            label = label_frame.astype(int).tolist()                   
            text = 'Nodo ' + text_print[label[0]]
            #img = cv2.resize(img, (800, 600))
            img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
            cv2.putText(img, text, (50, 50), font, 0.8, (0, 0, 255),2,cv2.LINE_AA)
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    #Dynamic image test       
    elif sys.argv[1] == 'video':
        print 'Running dynamic test'
        
        #Parameters
        label_frame = np.array([0])
        global detection_time
        global filt_delay
        #global buf_len 
        
        video_list = os.listdir(directory) 
        for i in range(len(video_list)) :
            # Uploading video set
            file_name = directory + video_list[i]
            cap = cv2.VideoCapture(file_name)
            timer = 0
            counter = 0
            last_label = np.array([0])
            #buffer_label = np.zeros((buf_len,1))
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                #Check end of the video
                if ret == True:
                    # Our operations on the frame come here
                    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    #Take images for classification
                    timer = timer + 1
                    if timer > detection_time:
                        label_frame = classify_image(frame, BoW, clusters, clf)
                        if (last_label != label_frame):
                            counter+=1
                            if counter == filt_delay:
                                last_label = label_frame
                                counter=0
#                        buffer_label [counter] = label_frame
#                        counter+=1
#                        if (counter == len(buffer_label)):
#                            counter=0
#                        for i in range(len(text_print)):
#                            counter2=0
#                            for j in range(len(buffer_label)):
#                                if (i == buffer_label[j]):
#                                    counter2+=1
#                            
#                            if (counter2 >= 5*len(buffer_label)/6):
#                                label_frame = buffer_label[i]
#                                print label_frame
#                                break
#                            else:
#                                label_frame = np.array([0])
                        timer = 0
                    
                    # Display the resulting frame
                    #label = label_frame.astype(int).tolist() 
                    label = last_label.astype(int).tolist() 
                    text = 'Nodo ' + text_print[label[0]]
                    frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                    cv2.putText(frame, text, (50, 50), font, 0.8, (0, 0, 255),2,cv2.LINE_AA)
                    cv2.imshow('frame',frame)
                    time.sleep(0.0008)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
          
            # When everything done, release the capture
            label_frame = np.array([0])
            cap.release()
            cv2.destroyAllWindows()
            
            
if __name__ == "__main__":
    main(sys.argv) 


