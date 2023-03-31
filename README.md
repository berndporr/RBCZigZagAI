# Machinelearning-Tensorflow-HealthyCellDetector
## framegenerator.py
Opens avi videos and extracts single frames, calculates the background and returns an image with its corresponding label (Healthy/Ill).
## celldetect.py 
Returns the frame positions where the cell is detected in narrow or wide section of the zig-zag channel.
## imageobt.py
Combines the framegenerator.py and celldetect.py to extract the frames where the cell is detected in the wide or narrow section. Imageobt.py returns the desired image in a tensor format with its corresponding label that can be used in Tensorflow
## train.py
Uses healthy and ill avi files to train and test a model where the user specifies the cell location. It evaluates the prediction accuracy and shows a plot for the best and worst detected images. 

Usage: python train.py -n -w -h -f maxframes

-w: wide channel

-n: narrow channel

-f maximum of frames to use for debugging

-h prints this help text
