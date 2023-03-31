#!/usr/bin/python

import celldetect
import framegenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np

#TEST FOR BACKGROUND SUBTRACTION IN NARROW AND WIDE CHANNEL
detectionareanarrow = [
    [282,10],
    [500,110]
    ]

detectionareawide = [
    [220,10],
    [500,110]
    ]
path= "/data/RBC-ZigZag/Selection/60xPhotron_20mBar_2_C001H001S0001.avi"
label= "Healthy"
crop = [[150,0],[400,120]]
clip_length= 50
fret= 1
avi = framegenerator.AVIfile(path,label, crop_rect = crop, clip_length = clip_length, frames2ret = fret)
frames_narrow= celldetect.celldetect_frames(path, detectionareanarrow,maxFrames= 20)
frames_wide= celldetect.celldetect_frames(path, detectionareawide,maxFrames= 20)


#Shows same plot as testcelldetect with narrow and wide but with background subtraction (can be useful to see how the
# different images look like with and without background)
fig= plt.figure(figsize=(6,6)) 
index=0
for i in frames_wide[0:4]:
    index= index +1
    imageandlabels = avi.getimagepairs(i)
    images=[]
    images.append(imageandlabels[1])
    image= np.squeeze(images)
    fig.add_subplot(3, 3, index)
    plt.title("Wide")
    plt.imshow(image)

index=4
for i in frames_narrow[0:5]:
    index= index+1
    imageandlabels = avi.getimagepairs(i)
    images=[]
    images.append(imageandlabels[1])
    image= np.squeeze(images)
    fig.add_subplot(3, 3, index)
    plt.title("Narrow")
    plt.imshow(image)
plt.show()







