#!/usr/bin/python

import celldetect
import framegenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np

#TESTING OF CELL DETECT FUNCTION FOR NARROW AND WIDE CHANNELS

path_to_healthy = "/data/RBC-ZigZag/Selection/60xPhotron_20mBar_2_C001H001S0001.avi"
crop = [[150,0],[400,120]]
clip_length= 50
fret= 1
avi_healthy= framegenerator.AVIfile(path_to_healthy,"Healthy", crop_rect = crop, clip_length = clip_length, frames2ret = fret)

detectionareanarrow = [
    [282,10],
    [500,110]
    ]

detectionareawide = [
    [220,10],
    [500,110]
    ]

frames_narrow= celldetect.celldetect_frames(path_to_healthy,detectionareanarrow, maxFrames= 50)
print(frames_narrow)


frames_wide= celldetect.celldetect_frames(path_to_healthy,detectionareawide, maxFrames=50)
print(frames_wide)
frames_wide= np.array(frames_wide)

#PLOT OF FRAMES IN NARROW AND WIDE CHANNEL
fig= plt.figure(figsize=(6,6)) 
index=0
for i in frames_wide[0:4]:
    index= index +1
    healthy_first_clip = avi_healthy.get_frames_of_clip(i)
    image= np.squeeze(healthy_first_clip)
    fig.add_subplot(3, 3, index)
    plt.title("Wide")
    plt.imshow(image)

index=4
for i in frames_narrow[0:5]:
    index= index+1
    healthy_first_clip = avi_healthy.get_frames_of_clip(i)
    image= np.squeeze(healthy_first_clip)
    fig.add_subplot(3, 3, index)
    plt.title("Narrow")
    plt.imshow(image)

plt.show()






