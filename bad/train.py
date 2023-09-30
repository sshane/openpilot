import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2

IMGS_PATH = 'C:/Users/Shane/bad/train'

images = []
image_fns = os.listdir(IMGS_PATH)
# print(image_fns)
for img_fn in image_fns:
  images.append(cv2.imread(f'{IMGS_PATH}/{img_fn}'))

# raise Exception
SAMPLES = len(images)
# RESOLUTION = images[0].shape[:2]  # (960, 540)
# print(f'Resolution: {RESOLUTION}')
# raise Exception
# X = np.random.random(((SAMPLES, 1920, 1080, 3)))
X = np.array(images)
Y = []
for image_fn in image_fns:
  # print(image_fn)
  y_start_idx = image_fn.split('_').index('y')
  fb = image_fn.split('_')[y_start_idx + 1]
  lr = image_fn.split('_')[y_start_idx + 2]
  # Y.append([int(fb), int(lr)])
  Y.append([1 if lr == 1 else 0,
            1 if lr == 0 else 0,
            1 if lr == -1 else 0])
  # print(image_fn, fb, lr)
# print(image_fns)
# raise Exception
Y = np.array(Y)
print(Y)
# Y = np.random.random((SAMPLES,))

INPUT_SHAPE = images[0].shape
print(f'X shape: {X.shape}')
print(f'Y shape: {Y.shape}')
print(f'Input shape: {INPUT_SHAPE}')

# model = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=INPUT_SHAPE),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(16, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(2, activation='linear'),
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(8),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax'),
])

# model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, Y, batch_size=8, epochs=10)

model.save('C:/Git/openpilot/bad/model.h5')
