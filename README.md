# AI based analysis of red blood cells in oscillating microchannels

[Andreas Link, Irene Luna Pardo, Bernd Porr and Thomas Franke (2023) AI based image analysis of red blood cells in oscillating microchannels. RSC Advances 41.](https://pubs.rsc.org/en/content/articlelanding/2023/RA/D3RA04644C)

 - Paper: https://doi.org/10.1039/D3RA04644C
 - Code: https://doi.org/10.5281/zenodo.7789864

## `framegenerator.py`
Opens avi videos and extracts single frames, calculates the background and returns an image with its corresponding label.
## `celldetect.py`
Returns the frame positions where the cell is detected in narrow or wide section of the zig-zag channel.
## `imageobt.py`
Combines the framegenerator.py and celldetect.py to extract the frames where the cell is detected in the wide or narrow section.
## `train.py` (main program)
Uses native and chem mod avi files to train and test a model where the user specifies the cell location in the channel. It evaluates the prediction accuracy and shows a plot for the best and worst detected images. 

Usage: `python train.py -n -w -h -f <maxframes>`

-w: wide channel

-n: narrow channel

-f maximum of frames to use for debugging

-h prints this help text
