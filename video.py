import numpy as np
import cv2
import threading
import time
from matplotlib import pyplot as plt


cap = cv2.VideoCapture('V71110-171745.mp4')
cont = 0
captured_frame = 0

def getFrame(foo,stop_event):
    global captured_frame
    cont = 0
    
    while(not stop_event.is_set()):
        time.sleep(2)
        cont = cont + 1
        print ('here in thread')
        cv2.imwrite('RESULTADO' + str(cont) + '.png', captured_frame)
        
    
#Crea el thread con su condicion de parada    
t_stop = threading.Event()
t = threading.Thread(target = getFrame, args=(1,t_stop))
t.start()



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cont = cont + 1
    if cont > 100:
        captured_frame = gray
        cont = 0
        
        
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        t_stop.set()
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
