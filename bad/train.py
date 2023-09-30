import tensorflow as tf
from tensorflow import keras
import numpy as np

SAMPLES = 100
X = np.random.random(((100, 1920, 1080, 3)))
Y = np.random.random((SAMPLES,))

INPUT_SHAPE=(1920, 1080, 3)


model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=INPUT_SHAPE)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, Y, batch_size=32, epochs=2)

