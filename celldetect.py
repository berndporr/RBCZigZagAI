import cv2
import numpy as np

def get_roi(img,roibox):
    """
    Crops the image around the region of interest
    """
    return img[roibox[0][1]:roibox[1][1],roibox[0][0]:roibox[1][0]]
    





def celldetect_frames(path,roibox,maxFrames = False, boxCallback = False):
    """
    Searches in roibox for a cell in the AVI file
    """

    vid= cv2.VideoCapture(path)


    object_detector= cv2.createBackgroundSubtractorMOG2(history= 100, varThreshold=20)

    count= 0
    framenumbers= []
    while True:
        ret, frame= vid.read()
    
        if not ret:
            break

        if maxFrames:
            if count > maxFrames:
                break

        roi= get_roi(frame,roibox)
        count= count+1
        mask = object_detector.apply(roi)
        contours, _ = cv2.findContours( mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area= cv2.contourArea(cnt)
        
            if area > 20:
                x, y, w, h= cv2.boundingRect(cnt)
                cv2.rectangle ( roi, (x,y), (x+w, y+h), (0,255,0),2)
                if x < 5:
                    framenumbers.append(count)
                    if boxCallback:
                        cv2.rectangle ( roi, (x,y), (x+w, y+h), (0,255,0),2)
                        boxCallback(roi)
                    break
       
    framenumbers= framenumbers[1:]
    th = 25
    idx =  np.argwhere(np.ediff1d(framenumbers) <= th) + 1
    b = np.delete(framenumbers,idx)
    return b
