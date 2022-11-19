import pathlib
import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.layers import Activation, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing import image
from keras.models import Sequential
import pyscreenshot as ImageGrab
import schedule

emotions = ['disgusted' 'happy' 'surprised' 'neutral' 'sad' 'angry' 'fearful']
output_class_units = len(emotions)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
emotionsdict = {0:'disgusted', 1:'happy', 2:'surprised', 3:'neutral', 4:'sad', 5:'angry', 6:'fearful'}

model = tf.keras.models.Sequential([
    # 1st conv
  tf.keras.layers.Conv2D(96, (11,11),strides=(4,4), activation='relu', input_shape=(227, 227, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2,2)),
    # 2nd conv
  tf.keras.layers.Conv2D(256, (11,11),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
     # 3rd conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 4th conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 5th Conv
  tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
  # To Flatten layer
  tf.keras.layers.Flatten(),
  # To FC layer 1
   tf.keras.layers.Dense(4096, activation='relu'),
    # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
  #To FC layer 2
  tf.keras.layers.Dense(4096, activation='relu'),
    # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(output_class_units, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss="categorical_crossentropy", 
              metrics=['accuracy',
                       tf.keras.metrics.Precision(), 
                       tf.keras.metrics.Recall(), 
                       tf.keras.metrics.SensitivityAtSpecificity(0.5), 
                       tf.keras.metrics.SpecificityAtSensitivity(0.5), 
                       tf.keras.metrics.AUC(curve='ROC')])

model.load_weights('final_test_model')

# getting the input

while True:
    img = ImageGrab.grab()
    img = tf.keras.utils.load_img(img, target_size=(227,227,3), interpolation="bilinear")
    img = np.array(img)
    img = img.reshape(1,227,227,3)
    img = image_generator.flow(img, batch_size=32, shuffle=True)
    result = model.predict(img) # use the model to predict our image
    result = list(result[0])
    img_index = result.index(max(result))
    emotion = emotionsdict[img_index]
    print(emotion)
    
    

# use the model to predict thye emotion

