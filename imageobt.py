import framegenerator
import celldetect
import cv2
import os
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import random

class ImageObtain:
    def __init__(self, detectionarea_y, detectionarea_width, crop_dx, crop_x_width, crop_y, finishing_line_index, 
                 random_shuffle = True, maxFrames= False, verb = False, backgroundsub = True):
        self.crop_dx = crop_dx
        self.crop_x_width = crop_x_width
        self.crop_y = crop_y
        self.maxFrames = maxFrames
        self.verb = verb
        self.backgroundsub = backgroundsub
        self.random_shuffle = random_shuffle
        self.finishing_line_index = finishing_line_index
        self.detectionarea_y = detectionarea_y
        self.detectionarea_width = detectionarea_width

    def obtaintensor(self, path, label, clip_length, x_finishing_line, pairs):
        detectionarea= [
            [ x_finishing_line, self.detectionarea_y[0] ],
            [ x_finishing_line + self.detectionarea_width, self.detectionarea_y[1] ]
        ]
        frameposarray= np.array(celldetect.celldetect_frames(path,detectionarea, self.maxFrames))
        fret=1
        crop = [
            [ x_finishing_line + self.crop_dx, self.crop_y[0] ],
            [ x_finishing_line + self.crop_dx + self.crop_x_width, self.crop_y[1] ]
        ]
        avi = framegenerator.AVIfile(path, label, crop_rect = crop, clip_length = clip_length, frames2ret = fret, verb = self.verb)
        for i in frameposarray[0:len(frameposarray)]:
            try:
                imageandlabels = avi.getimagepairs(i,backgroundsub = self.backgroundsub)
                pairs.append(imageandlabels)
            except LookupError as e:
                print(e)

    def get_pairs(self, healthyavis, illavis):
        pairs = []
        for h in healthyavis:
            self.obtaintensor(h[0], "Healthy", h[1], h[2+self.finishing_line_index], pairs)

        for i in illavis:
            self.obtaintensor(i[0], "Ill", i[1], i[2+self.finishing_line_index], pairs)

        if self.random_shuffle:
            random.shuffle(pairs)

        images = []
        labels = []
        for p in pairs:
            labels.append(p[0])
            images.append(p[1])

        rgb_tensor = tf.convert_to_tensor(images, dtype=tf.float32) #training one
        return labels, rgb_tensor
