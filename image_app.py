import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time
import os
import scipy
save_path = '/Users/patricknaylor/Desktop/Field_Detection/Images/Masked/'


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if not shape_done:
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

 
    # checking for right mouse clicks    

def flood_fill_hull(image, points):    
    #points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull   

if __name__ == '__main__':
    bool = True
    for i in range(1000000):
        shape_done = False
        print(i)
        clicks = []
        raw_paths = glob.glob('/Users/patricknaylor/Desktop/Field_Detection/Images/Raw/*')
        path = raw_paths[0]
        file_label = path[56:-4]
        print(file_label)
        #print(path[10:-3])
        img = cv2.imread(path)
        WHITE = (255, 255, 255)
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=WHITE)
        h, w, c = img.shape
        mask_arr = np.zeros((w, h))
        
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)
        while bool:
            k = cv2.waitKey(0)
            #print(k)
            if k == 27:
                bool = False
            elif (k == ord('s')) and (shape_done):
                #print('save here')
                mask, _ = flood_fill_hull(mask_arr, np.array(clicks))
                np.savetxt(f'{save_path}{file_label}.csv', mask, delimiter=',')
                os.rename(path, f'{save_path}{file_label}.jpg')
                break
            elif (k == ord('l')):
                #print('remove picture')
                os.remove(path)
                break
            elif (len(clicks) > 2) and (k == ord('a')):
                shape_done = True
                cv2.fillPoly(img, pts=[np.array(clicks)], color=(255, 255, 0))
                cv2.imshow('image', img)
            elif (k == 127):
                clicks = []
                img = cv2.imread(raw_paths[0])
                cv2.imshow('image', img)
        if not bool:
            break
    cv2.destroyAllWindows
