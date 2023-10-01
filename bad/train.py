import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMGS_PATH = 'C:/Users/Shane/bad'

images = []
image_fns = [i for i in os.listdir(IMGS_PATH) if i.endswith('.png')]
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
  # print(image_fn)
  fb = int(image_fn.split('_')[y_start_idx + 1])
  lr = int(image_fn.split('_')[y_start_idx + 2])
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

if TO_PLOT:=False:
  for x, y in zip(X, Y):
    plt.clf()
    plt.imshow(x)
    plt.title(str(y))
    plt.pause(0.2)

  raise Exception

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
del X, Y

INPUT_SHAPE = images[0].shape
print(f'X shape: {X_train.shape}, {X_test.shape}')
print(f'Y shape: {Y_train.shape}, {Y_test.shape}')
print(f'Input shape: {INPUT_SHAPE}')

# model = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=INPUT_SHAPE),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(16, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(2, activation='linear'),
# ])

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(24),
  tf.keras.layers.LeakyReLU(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Dense(12),
  tf.keras.layers.LeakyReLU(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Dense(3, activation='softmax'),
])

# model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=6, epochs=10)

i = input('Would you like to save? ')
if i.lower().strip() in ('y', 'yes', 'sure', 'why not'):
  print('saved winner model')
  model.save('C:/Git/openpilot/bad/model.h5')
else:
  print('didn\'t save loser model')
