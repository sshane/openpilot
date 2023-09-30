import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2

IMGS_PATH = 'C:/Users/Shane/bad'

images = []
print(os.listdir(IMGS_PATH))
for img_fn in os.listdir(IMGS_PATH):
  images.append(cv2.imread(f'{IMGS_PATH}/{img_fn}'))

# raise Exception
SAMPLES = len(images)
# RESOLUTION = images[0].shape[:2]  # (960, 540)
# print(f'Resolution: {RESOLUTION}')
# raise Exception
# X = np.random.random(((SAMPLES, 1920, 1080, 3)))
X = np.array(images)
Y = np.random.random((SAMPLES,))

INPUT_SHAPE = images[0].shape
print(f'X shape: {X.shape}')
print(f'Input shape: {INPUT_SHAPE}')

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(3, 3, activation='relu', input_shape=INPUT_SHAPE),
  tf.keras.layers.Dense(1, activation='relu'),
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, Y, batch_size=2, epochs=10)
