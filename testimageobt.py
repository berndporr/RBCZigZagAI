#!/usr/bin/python

import imageobt
import matplotlib.pyplot as plt
import random

# y-range of the detection area
detectionarea_y = [10,110]

# width of the detection area
detectionarea_width = 250

# xcrop start in relation to detection area
crop_dx = -70

# xcrop with
crop_x_width = 150

# y-range for the cropping
crop_y = [0,120]

maxframes = 1000

finishing_line_index = 0

# [ filename, frames/clip, wide-x, narrow-x finishing lines ]
healthyavis = [
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0008.avi", 60, 150, 212],
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0009.avi", 60, 193, 254],
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0001.avi", 40, 133, 186]
]

illavis = [
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_2___FA3_7percent_C001H001S0003.avi", 60, 181, 244],
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_2___FA3_7percent_C001H001S0001.avi", 60, 181, 244]
]

def plotWideNarrow(idx):
    imob = imageobt.ImageObtain(
    detectionarea_y, detectionarea_width, crop_dx, crop_x_width, crop_y, finishing_line_index, 
    random_shuffle = True, maxFrames = 1000, backgroundsub = False)

    imagesandlabels = imob.get_pairs(healthyavis, illavis)

    labels= imagesandlabels[0]
    images= imagesandlabels[1]
    class_names = ['Healthy', 'Unhealthy']

    x= random.randint(0,(len(images)-8))
    fig= plt.figure(figsize=(8,8))
    if idx == 0:
        fig.suptitle("Wide")
    elif idx == 1:
        fig.suptitle("Narrow")
    for i in range(8):
        x= random.randint(0,(len(images)-8))
        x= x+i
        plt.subplot(4,2,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[x], cmap=plt.cm.binary)
        plt.xlabel(class_names[(labels)[x]])

for finishing_line_index in range(2):
    plotWideNarrow(finishing_line_index)

plt.show()
