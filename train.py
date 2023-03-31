#!/usr/bin/python

import tensorflow as tf

from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import imageobt
import getopt

# y-range of the detection area
detectionarea_y = [10,110]

# width of the detection area
detectionarea_width = 250

# xcrop start in relation to detection area
crop_dx = -80

# xcrop with
crop_x_width = 150

# y-range for the cropping
crop_y = [0,120]

# Names of the classes
class_names = ['Healthy', 'Diseased']
num_classes = len(class_names)

##############################################################
# Training
# healthy
# [ filename, frames/clip, wide-x, narrow-x finishing lines ]
healthyavis = [
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0008.avi", 60, 150, 212],
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0009.avi", 60, 193, 254],
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0010.avi", 60, 193, 254]
]
# ill
# [ filename, frames/clip, wide-x, narrow-x finishing lines ]
illavis = [
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_2___FA3_7percent_C001H001S0001.avi", 60, 181, 244],
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_2___FA3_7percent_C001H001S0002.avi", 60, 181, 244]
]

################################################################
# Test
# healthy
# [ filename, frames/clip, wide-x, narrow-x finishing lines ]
healthytestavi = [
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_C001H001S0001.avi",40, 133, 186]
]
# [ filename, frames/clip, wide-x, narrow-x finishing lines ]
illtestavi = [
  ["/data/RBC-ZigZag/ALL/60xPhotron_20mBar_2___FA3_7percent_C001H001S0003.avi",60, 181, 244]
]

# max number of epochs
epochs = 10

# batch size for training
BATCH_SIZE = 32

#########################################################################################
# End of Parameters. Private code from here.
#########################################################################################

def print_usage():
  print("Usage: python {} -n -w -h -f maxframes".format(sys.argv[0]))
  print("-w: wide channel")
  print("-n: narrow channel")
  print("-f: maxframes: maximum of frames to use for debugging")
  print("-h: prints this help text")
  print("-l: label: forces label (index number 0,1,...)")
  print("-b: no background subtraction")
  quit()

print('TensorFlow version: {}'.format(tf.__version__))

finishing_line_index = -1
detectionarea_description = ""
maxFrames = False
verbose_msg = False
forceLabel = -1
backgroundsub = True

try:
  # Gather the arguments
  all_args = sys.argv[1:]
  opts, arg = getopt.getopt(all_args, 'bvnwf:l:')
  # Iterate over the options and values
  for opt, arg_val in opts:
    if '-w' in opt:
      finishing_line_index = 0
      detectionarea_description = "wide_channel_section"
    elif '-n' in opt:
      finishing_line_index = 1
      detectionarea_description = "narrow_channel_section"
    elif '-f' in opt:
      maxFrames = int(arg_val)
    elif '-l' in opt:
      forceLabel = int(arg_val)
    elif '-b' in opt:
      backgroundsub = False
    elif '-v' in opt:
      verbose_msg = True
    elif '-h' in opt:
      raise getopt.GetoptError()
    else:
      raise getopt.GetoptError()
except getopt.GetoptError:
  print_usage()

if finishing_line_index < 0:
  print_usage()

img_height = crop_y[1] - crop_y[0]
img_width = crop_x_width
print("width = {}, height = {}".format(img_width,img_height))

if forceLabel == 0:
  print("Forcing all to be healthy")
  illavis = healthyavis
  illtestavi = healthytestavi
elif forceLabel == 1:
  print("Forcing to be all diseased")
  healthyavis = illavis
  healthytestavi = illtestavi

# give it all the constants once
imob = imageobt.ImageObtain(
  detectionarea_y, detectionarea_width, crop_dx, crop_x_width, crop_y, finishing_line_index, 
  random_shuffle = True, maxFrames= maxFrames, verb = verbose_msg, backgroundsub = backgroundsub)
train_labels, train_images = imob.get_pairs(healthyavis, illavis)
test_labels, test_images = imob.get_pairs(healthytestavi, illtestavi)

print("Shape of training images:",train_images.shape)
print("Length of training labels:",len(train_labels))

print("Shape of testing images:", test_images.shape)
print("Length of testing labels:",len(test_labels))

# 80% for trainign and 20% for validation
startIndexValidation = int(len(train_images) * 80 / 100)
print("Index of 1st validation image:",startIndexValidation)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images[:startIndexValidation], train_labels[:startIndexValidation]))
val_dataset = tf.data.Dataset.from_tensor_slices((train_images[startIndexValidation:], train_labels[startIndexValidation:]))
test_dataset= tf.data.Dataset.from_tensor_slices((test_images, test_labels))

SHUFFLE_BUFFER_SIZE = len(train_images)

train_dataset = train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.cache().batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)


model = keras.Sequential([
  keras.layers.Rescaling(1.0, input_shape=(img_height, img_width, 3)),
  keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(num_classes)
])

model.summary()

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# early stopping of the epochs if no improvement
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[callback])

print("---------------------------------")
acc = history.history['accuracy']
print("Final training accuracy:",acc[-1])
val_acc = history.history['val_accuracy']
print("Final validation accuracy:",val_acc[-1])

loss = history.history['loss']
print("Final training loss:",loss[-1])
val_loss = history.history['val_loss']
print("Final validation loss:",val_loss[-1])

print("---------------------------------")
print("Evaluating the test accuracy...")
test_loss, test_acc = model.evaluate(test_dataset)
print('\nTest accuracy: {}'.format(test_acc))

model.save("models/"+detectionarea_description)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

def plot_training_results(description):
  epochs_range = range(epochs)
  fig = plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  fig.savefig("results/valloss-{}.eps".format(description),format="eps")

def plot_image(true_label, prediction, img):
  predicted_label = np.argmax(prediction)
  perc = 100*np.max(prediction)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                       perc,
                                       class_names[true_label][0]))

def plot_value_array(prediction):
  plt.grid(False)
  plt.xticks(range(2))
  #plt.yticks([])
  thisplot = plt.bar(range(2), prediction, color="#777777")
  plt.ylim([0, 1])
  thisplot[0].set_color('green')
  thisplot[1].set_color('red')


# Evaluates if we have one of these: any, best or worst
def bestworstprediction(atest_label,prediction,type):
    predicted_label = np.argmax(prediction)
    if type== "best":
      if atest_label == predicted_label:
        for p in prediction:
          if p > 0.9:
            return True
      return False
    elif type== "worst":
      return atest_label != predicted_label
    elif type== "any":
      return True
    print("BUG: wrong argument")
    quit()


# We want to plot one row of healthy and one row of unhealthy
def finalplot(predictions, test_labels, description, type):
  # We have one row for healthy and one row for ill
  num_rows = num_classes
  # We get 5 examples of healthy or ill
  num_cols = 5
  # Total number of images
  num_images = num_rows*num_cols

  fig = plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  fig.suptitle("Final eval {} {}".format(description,type))
  # We loop through the test labels
  for l in range(num_classes):
    # We loop through the column
    testIdx = 0
    for c in range(num_cols):
      # We need to find a matching prediction which satisfies type
      while (testIdx < len(predictions)):
        a = False
        # Is that the label we are looking for?
        if l == test_labels[testIdx]:
          a = bestworstprediction(l,predictions[testIdx],type)
        if a:
          break
        testIdx += 1
      if testIdx < len(predictions):
        plt.subplot(num_rows, 2*num_cols, 2*c+1 + 2*num_cols*l)
        plot_image(test_labels[testIdx], predictions[testIdx], test_images[testIdx])
        plt.subplot(num_rows, 2*num_cols, 2*c+2 + 2*num_cols*l)
        plot_value_array(predictions[testIdx])
        testIdx += 1
  plt.tight_layout()
  fig.savefig("results/{}-{}.eps".format(description,type),format="eps")

plot_training_results(detectionarea_description)

finalplot(predictions,test_labels,detectionarea_description,"best")
finalplot(predictions,test_labels,detectionarea_description,"worst")
finalplot(predictions,test_labels,detectionarea_description,"any")

plt.show()
