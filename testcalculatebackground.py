#!/usr/bin/python

import celldetect
import framegenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np


path_to_healthy = "/data/RBC-ZigZag/Selection/60xPhotron_20mBar_2_C001H001S0001.avi"
crop = [[150,0],[400,120]]
clip_length= 50
fret= 1
avi= framegenerator.AVIfile(path_to_healthy,"Healthy", crop_rect = crop, clip_length = clip_length, frames2ret = fret)


detectionareanarrow = [
    [282,10],
    [500,110]
    ]

frames_narrow= celldetect.celldetect_frames(path_to_healthy, detectionareanarrow,maxFrames= 20)

fig= plt.figure(figsize=(6,6)) 
index=0

for i in frames_narrow[0:3]:
    index= index +1
    images = avi.get_frames_of_clip(i)
    image= np.squeeze(images)
    fig.add_subplot(3, 3, index)
    plt.title("Frames")
    plt.imshow(image)

index=3
for i in frames_narrow[0:3]:
    index= index+1
    background = avi.calcbackground(i)
    background= np.squeeze(background)
    fig.add_subplot(3, 3, index)
    plt.imshow(background)
    plt.title("Background")

index=6
for i in frames_narrow[0:3]:
    index= index+1
    imageandlabels = avi.getimagepairs(i)
    finalimages=[]
    finalimages.append(imageandlabels[1])
    finalimage= np.squeeze(finalimages)
    fig.add_subplot(3, 3, index)
    plt.title("Final image")
    plt.imshow(finalimage)
plt.show()
