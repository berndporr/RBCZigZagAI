import tqdm
import random
import pathlib
import itertools
import collections
import copy
import math

import os
import cv2
import numpy as np
import tensorflow as tf

class AVIfile:

  def __init__(self, video_path, label_name, clip_length, crop_rect = False, frame_step = 1, frames2ret = False, verb = False):
    self.verb = verb
    self.video_path = video_path
    self.clip_length = clip_length
    self.label_name = label_name
    self.crop_rect = crop_rect
    self.frame_step = frame_step
    if not frames2ret:
      self.frames2ret = clip_length
    else:
      self.frames2ret = frames2ret
    self.src = cv2.VideoCapture(str(video_path))
    if not self.src:
      print("Could not open:",video_path)
    self.video_length = int(self.src.get(cv2.CAP_PROP_FRAME_COUNT))
    self.width  = int(self.src.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    self.height = int(self.src.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    print("{} opened: {} frames, {} clips at {}x{}".format(video_path,self.video_length,self.get_number_of_clips(),
    self.width,self.height))


  def __del__(self):
    self.src.release()

  def format_frames(self,frame):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    return frame

  def get_number_of_clips(self):
    return self.video_length // self.clip_length

  def get_frames_of_clip(self, framepos):
    output_size = (self.width, self.height)

    clip_index =  math.ceil(framepos/ self.clip_length)
    
    if self.verb:
      print("---> {} going to frame pos {}, clip length = {}, clip index = {}.".format(self.video_path,framepos,self.clip_length,clip_index))
    self.src.set(cv2.CAP_PROP_POS_FRAMES, framepos)


    result = []
    ret, frame = self.src.read()

    if not ret:
      raise LookupError("Lookup error {} going to frame pos {} of total {} frames.".format(self.video_path,framepos,self.video_length))

    frame = self.format_frames(frame)

    result.append(frame)
    
    for i in range(1,self.frames2ret):
      
      if (i % self.frame_step) == 0:
        if ret:
          frame = self.format_frames(frame)
          result.append(frame)
        else:
          result.append(np.zeros_like(frame[0]))
    ## returning the array but fixing openCV's odd BGR format to RGB
    
    result = np.array(result)[...,[2,1,0]]

    if (self.crop_rect):
      result = tf.image.crop_to_bounding_box(result, 
      self.crop_rect[0][1], self.crop_rect[0][0], 
      self.crop_rect[1][1]-self.crop_rect[0][1], self.crop_rect[1][0]-self.crop_rect[0][0])

    return result

  def calcbackground(self,framepos):
    clip_index =  math.ceil(framepos/ self.clip_length)
    x= framepos-(int((self.clip_length/2)-1))

    back= self.get_frames_of_clip(x)
    background= np.squeeze(back)
    return background
  
  
  def getimagepairs(self,framepos,backgroundsub = True):
    pairs= []
    x= self.get_frames_of_clip(framepos)
    image= np.squeeze(x)
    if backgroundsub:
      background= self.calcbackground(framepos)
      frame_diff= np.abs(image-background)
      image = frame_diff
    max= np.max(image)
    frame_wobg= image/max
    frame_wobg= np.array(frame_wobg)
    
    if self.label_name== 'Ill':
      pairs= (1, frame_wobg)
    
    if self.label_name== 'Healthy':
      pairs = (0, frame_wobg)

    return pairs

  
    
    







    

 

