#!/usr/bin/env python3
import random

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import compute_class_weight
import random
from common.basedir import BASEDIR
# from sklearn.model_selection import train_test_split

# IMGS_PATH = '/mnt/c/Users/Shane/bad'
IMGS_PATH = os.path.join(BASEDIR, 'bad/data/all')

# images = []
sample_folders = [i for i in os.listdir(IMGS_PATH) #if i.endswith('.png')
             # if random.uniform(0, 1) > 0.5
             #      if '0af' in i
             ]
# print(sample_folders)

X_prev = []
X_cur = []
Y_new = []
Y_action = []
for sample_folder_name in tqdm(sample_folders):
  prev_image = cv2.imread(f'{IMGS_PATH}/{sample_folder_name}/image_0_prev.png')
  prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
  prev_image = prev_image.reshape(prev_image.shape[0], prev_image.shape[1], 1)
  X_prev.append((prev_image / 255.).astype(np.float32))

  cur_image = cv2.imread(f'{IMGS_PATH}/{sample_folder_name}/image_1_cur.png')
  cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
  cur_image = cur_image.reshape(cur_image.shape[0], cur_image.shape[1], 1)
  X_cur.append((cur_image / 255.).astype(np.float32))

  y_start_idx = sample_folder_name.split('_').index('y')
  # print(sample_folder_name)
  fb = int(sample_folder_name.split('_')[y_start_idx + 1])
  lr = int(sample_folder_name.split('_')[y_start_idx + 2])
  # print(sample_folder_name, fb, lr)
  Y_new.append([fb, lr])

  # Initialize actions as [forward, backward, left, right, do_nothing]
  actions = [0, 0, 0, 0, 0]

  # because we don't want bias if both fb and lr are actuating, randomly pick one
  use_lr = lr != 0
  use_fb = fb != 0
  if use_lr and use_fb:
    use_fb = False
    # if random.uniform(0, 1) > 0.5:
    #   use_lr = False
    # else:
    #   use_fb = False

  if use_fb:
    if fb == -1:
      actions[1] = 1
    else:
      actions[0] = 1
  elif use_lr:
    if lr == 1:
      actions[2] = 1
    else:
      actions[3] = 1
  else:  # DO NOTHING
    actions[4] = 1

  Y_action.append(actions)

# raise Exception
N_SAMPLES = len(X_prev)
X_prev = np.array(X_prev)
X_cur = np.array(X_cur)
Y_new = np.array(Y_new)
Y_action = np.array(Y_action)
# RESOLUTION = images[0].shape[:2]  # (960, 540)
# print(f'Resolution: {RESOLUTION}')
# raise Exception
# X = np.random.random(((SAMPLES, 1920, 1080, 3)))
# X = np.array(images)



# Y_new = []
# Y_fb = []
# Y_lr = []
# Y_action = []
# for image_fn in image_fns:
#   # print(image_fn)
#   y_start_idx = image_fn.split('_').index('y')
#   # print(image_fn)
#   fb = int(image_fn.split('_')[y_start_idx + 1])
#   lr = int(image_fn.split('_')[y_start_idx + 2])
#   Y_new.append([fb, lr])
#   # Y.append([int(fb), int(lr)])
#   Y_fb.append([1 if fb == 1 else 0,
#                1 if fb == 0 else 0,
#                1 if fb == -1 else 0])
#   Y_lr.append([1 if lr == 1 else 0,
#                1 if lr == 0 else 0,
#                1 if lr == -1 else 0])
#
#   # Initialize actions as [forward, backward, left, right, do_nothing]
#   actions = [0, 0, 0, 0, 0]
#
#   # because we don't want bias if both fb and lr are actuating, randomly pick one
#   use_lr = lr != 0
#   use_fb = fb != 0
#   if use_lr and use_fb:
#     if random.uniform(0, 1) > 0.5:
#       use_lr = False
#     else:
#       use_fb = False
#
#   if use_fb:
#     if fb == -1:
#       actions[1] = 1
#     else:
#       actions[0] = 1
#   elif use_lr:
#     if lr == 1:
#       actions[2] = 1
#     else:
#       actions[3] = 1
#   else:  # DO NOTHING
#     actions[4] = 1
#
#   Y_action.append(actions)
#
#   # print(image_fn, fb, lr)
#   # print(image_fns)
# # Y = np.array(Y)
# Y_combined = np.array(Y_action, dtype=np.float32)
# Y_new = np.array(Y_new, dtype=np.float32)
# print(Y_combined)

# # raise Exception
# Y_combined = []
# for fb, lr in zip(Y_fb, Y_lr):
#   # Create a 9-element list initialized with zeros
#   combined = [0] * 9
#
#   # Index calculation: 3 * fb's index + lr's index
#   index = 3 * fb.index(1) + lr.index(1)
#   combined[index] = 1
#
#   Y_combined.append(combined)
# Y_combined = np.array(Y_combined, dtype=np.float32)
# # print(Y_combined)
# # raise Exception

# del Y_fb, Y_lr


# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=10,
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     vertical_flip=False
# )
#
# o = datagen.flow(X, Y_combined, batch_size=32)
# print(len(list(o)))
# raise Exception



# Y_fb = np.array(Y_fb, dtype=np.float32)
# Y_lr = np.array(Y_lr, dtype=np.float32)


# print(np.unique(Y_fb, return_counts=True))
# print(np.unique(Y_lr, return_counts=True))
# raise Exception


# # balance
# y_labels = np.argmax(Y, axis=1)
# class_counts = np.bincount(y_labels)
# max_class_index = np.argmax(class_counts)
# target_count = sorted(class_counts)[-2]
# majority_class_indices = np.where(y_labels == max_class_index)[0]
# delete_indices = np.random.choice(majority_class_indices, size=len(majority_class_indices)-target_count, replace=False)
# X = np.delete(X, delete_indices, axis=0)
# Y = np.delete(Y, delete_indices, axis=0)
#
# print(len([i for i in Y if i[0]]))
# print(len([i for i in Y if i[1]]))
# print(len([i for i in Y if i[2]]))


if TO_PLOT := False:
  for x, y in zip(X_cur, Y_new):
    plt.clf()
    plt.imshow(x.astype(np.float32))
    plt.title(str(y))
    plt.pause(1)

  raise Exception

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
# del X, Y

INPUT_SHAPE = X_prev[0].shape
print(f'X_prev, X_cur shape: {X_prev.shape}, {X_cur.shape}')
print(f'Y shape: {Y_new.shape}')
print(f'Input shape: {INPUT_SHAPE}')


# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=False,
#     fill_mode='nearest',
#     validation_split=0.5,
# )
# #
# datagen.fit(X_train)


input_prev = tf.keras.Input(shape=INPUT_SHAPE)
input_cur = tf.keras.Input(shape=INPUT_SHAPE)

# previous frame (0.5s ago)
x = tf.keras.layers.Conv2D(16, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))(input_prev)
y = tf.keras.layers.SpatialDropout2D(0.1)(x)
x = tf.keras.layers.ELU()(x)
# x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.SpatialDropout2D(0.1)(x)
x = tf.keras.layers.ELU()(x)
# x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.models.Model(inputs=input_prev, outputs=x)

# current frame
y = tf.keras.layers.Conv2D(16, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))(input_cur)
y = tf.keras.layers.SpatialDropout2D(0.1)(y)
y = tf.keras.layers.ELU()(y)
# y = tf.keras.layers.MaxPooling2D((2, 2))(y)

y = tf.keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))(y)
y = tf.keras.layers.SpatialDropout2D(0.1)(y)
y = tf.keras.layers.ELU()(y)
# y = tf.keras.layers.MaxPooling2D((2, 2))(y)

y = tf.keras.models.Model(inputs=input_cur, outputs=y)

# concat two branches and add a few dense layers
combined = tf.keras.layers.concatenate([x.output, y.output])

z = tf.keras.layers.Flatten()(combined)
z = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))(z)
z = tf.keras.layers.ELU()(z)
# z = tf.keras.layers.BatchNormalization()(z)

z = tf.keras.layers.Flatten()(combined)
z = tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))(z)
z = tf.keras.layers.ELU()(z)
# z = tf.keras.layers.BatchNormalization()(z)

# z = tf.keras.layers.Dense(6)(z)
# # z = tf.keras.layers.BatchNormalization()(z)
# z = tf.keras.layers.ELU()(z)
# z = tf.keras.layers.Dropout(0.2)(z)


# z = tf.keras.layers.Dense(2, activation="linear")(z)
z = tf.keras.layers.Dense(5, activation="softmax")(z)

# final model
model = tf.keras.models.Model(inputs=[x.input, y.input], outputs=z)

# model = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=INPUT_SHAPE),
#   # tf.keras.layers.SpatialDropout2D(0.2),
#   # tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.LeakyReLU(),
#   tf.keras.layers.MaxPooling2D((2, 2)),
#
#   tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
#   # tf.keras.layers.SpatialDropout2D(0.2),
#   # tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.LeakyReLU(),
#   tf.keras.layers.MaxPooling2D((2, 2)),
#
#   # tf.keras.layers.Conv2D(12, (3, 3), padding='same'),
#   # # tf.keras.layers.SpatialDropout2D(0.2),
#   # tf.keras.layers.BatchNormalization(),
#   # tf.keras.layers.LeakyReLU(),
#   # tf.keras.layers.MaxPooling2D((2, 2)),
#
#   # tf.keras.layers.Conv2D(24, (3, 3), padding='same'),
#   # tf.keras.layers.SpatialDropout2D(0.2),
#   # tf.keras.layers.LeakyReLU(),
#   # tf.keras.layers.MaxPooling2D((2, 2)),
#
#   # tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
#   # tf.keras.layers.BatchNormalization(),
#   # tf.keras.layers.LeakyReLU(),
#   # tf.keras.layers.MaxPooling2D((2, 2)),
#   # tf.keras.layers.Dropout(0.1),
#
#   # tf.keras.layers.Conv2D(16, (5, 5), padding='same'),
#   # tf.keras.layers.BatchNormalization(),
#   # tf.keras.layers.LeakyReLU(),
#   # tf.keras.layers.MaxPooling2D((2, 2)),
#   # tf.keras.layers.Dropout(0.1),
#
#   tf.keras.layers.Flatten(),
#
#   tf.keras.layers.Dense(16),  # , kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#   # tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.LeakyReLU(),
#   tf.keras.layers.Dropout(0.5),
#
#   tf.keras.layers.Dense(16),#, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#   # tf.keras.layers.BatchNormalization(),
#   tf.keras.layers.LeakyReLU(),
#   tf.keras.layers.Dropout(0.5),
#
#   # tf.keras.layers.Dense(8),  # , kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#   # tf.keras.layers.LeakyReLU(),
#   # # tf.keras.layers.BatchNormalization(),
#   # # tf.keras.layers.Dropout(0.1),
#
#   # tf.keras.layers.Dense(16),  # , kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#   # # tf.keras.layers.BatchNormalization(),
#   # tf.keras.layers.LeakyReLU(),
#   # tf.keras.layers.Dropout(0.1),
#
#   # tf.keras.layers.Dense(5, activation='softmax'),
#   tf.keras.layers.Dense(2, activation='linear'),
# ])


# fb_output = tf.keras.layers.Dense(3, activation='softmax', name='fb_output')(model.output)
# lr_output = tf.keras.layers.Dense(3, activation='softmax', name='lr_output')(model.output)
# full_model = tf.keras.Model(inputs=model.input, outputs=[fb_output, lr_output])

# classes are not weighted properly
Y_classes = np.argmax(Y_action, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(Y_classes), y=Y_classes)
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)



# opt = tf.keras.optimizers.SGD(lr=0.001)
# opt = tf.keras.optimizers.Adamax(lr=0.001)
opt = tf.keras.optimizers.Adam(lr=0.001, amsgrad=True)
# Compile the model with categorical crossentropy for both outputs
# full_model.compile(optimizer='adam',
#                    loss={'fb_output': 'categorical_crossentropy',
#                          'lr_output': 'categorical_crossentropy'},
#                    metrics=['accuracy'])


# model.compile(optimizer='adam', loss='mse')
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=opt, loss='mse', metrics=['mae'])

try:
  # full_model.fit(X, {'fb_output': Y_fb, 'lr_output': Y_lr},
  # model.fit(X, Y_combined,
  # model.fit([X_prev, X_cur], Y_new,
  model.fit([X_prev, X_cur], Y_action,
  # model.fit(datagen.flow(X, Y_combined, batch_size=32, subset='training'),
            batch_size=32,
            epochs=100,
            class_weight=class_weight_dict,
            validation_split=0.25)
            # validation_data=datagen.flow(X, Y_combined, subset='validation'))
except KeyboardInterrupt:
  pass

i = input('\n\nWould you like to save? ')
if i.lower().strip() in ('y', 'yes', 'sure', 'why not'):
  print('saved winner model')
  model.save(os.path.join(BASEDIR, 'bad/models/model.h5'))
else:
  print('didn\'t save loser model')
