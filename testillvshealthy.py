#!/usr/bin/python

import celldetect
import framegenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np

#TEST OBTAINING FRAMES IN NARROW SECTION FOR HEALTHY AND UNHEALTHY CELLS
#TESTING IT WORKS WITH HEALTHY AND ILL CELLS 

path_healthy= "/data/RBC-ZigZag/Selection/60xPhotron_20mBar_2_C001H001S0001.avi"
path_ill= "/data/RBC-ZigZag/Selection/60xPhotron_20mBar_2___1percentGA_C001H001S0001.avi"

crop = [[150,0],[400,120]]
clip_lengthhealthy = 50
clip_lengthill= 60
fret = 1
avi_healthy = framegenerator.AVIfile(path_healthy,"Healthy", crop_rect = crop, clip_length = clip_lengthhealthy, frames2ret = fret)
avi_ill  = framegenerator.AVIfile(path_ill,"Ill", crop_rect = crop, clip_length = clip_lengthill, frames2ret = fret)

detectionareanarrow = [
    [282,10],
    [500,110]
    ]
frameshealthynarrow= celldetect.celldetect_frames(path_healthy,detectionareanarrow, maxFrames= 30)
framesillnarrow= celldetect.celldetect_frames(path_ill,detectionareanarrow, maxFrames= 30)

#Withbackground
fig= plt.figure(figsize=(9,9)) 
index= 0
for i in (frameshealthynarrow[0:4]):
    index= index +1
    images= avi_healthy.get_frames_of_clip(i)   
    image= np.squeeze(images)
    fig.add_subplot(3, 3, index)
    plt.title("Healthy")
    plt.imshow(image)
index=4
for i in (framesillnarrow[0:5]):
    index= index +1
    images = avi_ill.get_frames_of_clip(i)   
    image= np.squeeze(images)
    fig.add_subplot(3, 3, index)
    plt.title("Unhealthy")
    plt.imshow(image)

plt.show()


#Without background
fig= plt.figure(figsize=(9,9)) 
index= 0
for i in (frameshealthynarrow[0:4]):
    index= index +1
    imageandlabels= avi_healthy.getimagepairs(i)   
    images=imageandlabels[1]
    labels=imageandlabels[0]

    image= np.squeeze(images)
    fig.add_subplot(3, 3, index)
    plt.title("Healthy")
    plt.imshow(image)
index=4
for i in (framesillnarrow[0:5]):
    index= index +1
    imageandlabels= avi_ill.getimagepairs(i)   
    images=imageandlabels[1]
    labels=imageandlabels[0]

    if labels== [0]:
        labelname= "Healthy"
    if labels== [1]:
        labelname= "Ill"
    image= np.squeeze(images)
    fig.add_subplot(3, 3, index)
    plt.title("Unhealthy")
    plt.imshow(image)

plt.show()