import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

gray, harris_only, corner_img = 0,0,0
def extractor(img, sizeX = 800, sizeY = 400):
    global gray
    global harris_only
    #Reads image and sets its size to preset
    #Comment one to test the other
    #img = cv2.imread('vlcsnap-2017-11-14-17h55m48s263.png')
    img = cv2.resize(img, (sizeX, sizeY))

    #Generates a gray image and extract corners
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    harris_only = cv2.cornerHarris(gray,2,3,0.04)  #TO DO --> Check values


    #Increases size of the markers
    harris_only = cv2.dilate(harris_only,None)

    # Adds the markers to the original picture
    img[harris_only>0.01*harris_only.max()]=[0,0,255] #TO DO --> Check values

    return img


def dealWithResults(corner_img):
    #Stores info if 's'  exits if 'ESC'
    global harris_only, gray
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  #tecla ESC
        cv2.destroyAllWindows()
    elif key == ord('s'):
        cv2.imwrite('cornerHarris.png', harris_only)
        cv2.imwrite('gray.png', gray)
        cv2.imwrite('result.png', corner_img)
        cv2.destroyAllWindows()

def main(argv):
    img = cv2.imread('vlcsnap-2017-11-14-17h56m07s659.png')
    corner_img = extractor(img)
    cv2.imshow('dst',corner_img)
    dealWithResults(corner_img)

if __name__ == "__main__":
    main(sys.argv)
