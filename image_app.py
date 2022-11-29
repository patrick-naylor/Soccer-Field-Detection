import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time
import os
save_path = 'Images/Masked/'



def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if len(clicks) < 4:
            clicks.append([x,y])
            #print(clicks)
        # displaying the coordinates
        # on the Shell
        #print(x, ' ', y)
 
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
    bool = True
    for i in range(1000000):
        print(i)
        clicks = []
        raw_paths = glob.glob('Images/Raw/*')
        path = raw_paths[0]
        file_label = path[11:-4]
        #print(path[10:-3])
        img = cv2.imread(path)
        h, w, c = img.shape
        mask_arr = np.zeros((h, w))
        
        cv2.putText(img, 'Place four corners', (w//2, h//2),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)
        while bool:
            k = cv2.waitKey(0)
            #print(k)
            if k == 27:
                bool = False
            elif (k == ord('s')) and (len(clicks)  == 4):
                #print('save here')
                os.rename(path, f'{save_path}{file_label}.jpg')
                break
            elif (k == ord('l')):
                #print('remove picture')
                os.remove(path)
                break

            elif (k == 127):
                clicks = []
                img = cv2.imread(raw_paths[0])
                cv2.imshow('image', img)
        if not bool:
            break
    cv2.destroyAllWindows
