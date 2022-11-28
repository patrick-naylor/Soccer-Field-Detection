import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time
import os



def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if len(clicks) < 4:
            clicks.append([x,y])
            print(clicks)
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    if len(clicks) == 4:
        cv2.fillPoly(img, pts=[np.array(clicks)], color=(255, 255, 0))
        cv2.imshow('image', img)
 
    # checking for right mouse clicks    





if __name__ == '__main__':
    clicks = []
    raw_paths = glob.glob('Images/Raw/*')
    print(raw_paths)
    img = cv2.imread(raw_paths[0])
    h, w, c = img.shape
    
    cv2.putText(img, 'Place four corners', (w//2, h//2),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)

    while True:
        k = cv2.waitKey(0)
        print(k)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif (k == ord('s')) and (len(clicks) == 4):
            print('save here')
        elif (k == 127):
            clicks = []
            img = cv2.imread(raw_paths[0])
            cv2.imshow('image', img)