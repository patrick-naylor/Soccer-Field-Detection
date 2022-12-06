import numpy as np
import cv2
import glob
import os
import scipy

save_path = "/Users/patricknaylor/Desktop/Field_Detection/Images/Masked/"
# This code allows users to select the field from an image to be saved in a 2d mask for training
# On load an image is shown if the image is not a wide shot of a field the user can press 'r' to delete image and load a new image
# If the image is ok the user can select the corners of the field (any amout over 2) and his s to submit the drawing
# If the user makes a mistake they can hit delete to undo all clicks and start again
# The user can hit escape to exit program
# Arrays are saved transposed and need to be transposed again


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if not shape_done:
            # Add click location to list for shape defining
            clicks.append([x, y])

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
            cv2.imshow("image", img)

    # checking for right mouse clicks


def flood_fill_hull(image, points):
    # Create mask with users selection using a convex hull of their selected points
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull


if __name__ == "__main__":
    bool = True
    # Loop until user manually closes window
    while True:
        # Load image
        shape_done = False
        # print(i)
        clicks = []
        raw_paths = glob.glob(
            "/Users/patricknaylor/Desktop/Field_Detection/Images/Raw/*"
        )
        raw_count = len(raw_paths)
        masked_count = len(glob.glob("/Users/patricknaylor/Desktop/Field_Detection/Images/Masked/*"))//2
        print(f'Images Remaining: {raw_count} \nImages Processed: {masked_count}')
        path = raw_paths[0]
        file_label = path[56:-4]
        # print(file_label)
        ##print(path[10:-3])
        img = cv2.imread(path)
        WHITE = (255, 255, 255)
        # Add white padding to image to make selection easier for user
        img = cv2.copyMakeBorder(img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=WHITE)
        h, w, c = img.shape
        while (h != 770) or (w != 1330):
            os.remove(path)
            raw_paths = glob.glob(
                "/Users/patricknaylor/Desktop/Field_Detection/Images/Raw/*"
            )
            path = raw_paths[0]
            file_label = path[56:-4]
            img = cv2.imread(path)
            img = cv2.copyMakeBorder(
                img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=WHITE
            )
            h, w, c = img.shape
        # Create array with shape of image frame
        mask_arr = np.zeros((w, h))

        cv2.imshow("image", img)
        # On click run click_event
        cv2.setMouseCallback("image", click_event)
        while bool:
            k = cv2.waitKey(0)
            # on 'escape' close both loops and exit program
            if k == 27:
                bool = False
            elif (k == ord("s")) and (shape_done):
                # when user selects s and the shape is complete create and save mask and move image file to masked folder
                mask, _ = flood_fill_hull(mask_arr, np.array(clicks))
                #print(path, file_label)
                np.save(f"{save_path}{file_label}.npy", mask.astype("int8"))
                os.rename(path, f"{save_path}{file_label}.jpg")
                # Exit while loop
                break
            elif k == ord("r"):
                # When user selects 'r' delete and load new image
                os.remove(path)
                break
            elif (len(clicks) > 2) and (k == ord("a")):
                # When user selects 'a' the shape is filled in on the image and the user can select s to save
                shape_done = True
                cv2.fillPoly(img, pts=[np.array(clicks)], color=(255, 255, 0))
                cv2.imshow("image", img)
            elif k == 127:
                # User selects 'del' to undo clicks
                shape_done = False
                clicks = []
                img = cv2.imread(raw_paths[0])
                img = cv2.copyMakeBorder(
                    img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=WHITE
                )
                cv2.imshow("image", img)
        if not bool:
            break
    cv2.destroyAllWindows
