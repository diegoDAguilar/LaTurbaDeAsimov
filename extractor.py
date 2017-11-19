import cv2
import numpy as np
from matplotlib import pyplot as plt

#Reads image and sets its size to preset
#img = cv2.imread('vlcsnap-2017-11-14-17h55m48s263.png') #Comment one to test the other
img = cv2.imread('vlcsnap-2017-11-14-17h56m07s659.png')
img = cv2.resize(img, (800, 400))

#Generates a gray image and extract corners
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)  #TO DO --> Check values

#Increases size of the markers
dst = cv2.dilate(dst,None)

# Adds the markers to the original picture
img[dst>0.01*dst.max()]=[0,0,255] #TO DO --> Check values

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

