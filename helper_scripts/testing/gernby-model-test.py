import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise, GRU

x_train = np.random.rand(1000, 5, 77)
y_train = np.random.rand(1000, 77)


model = Sequential()
model.add(GRU(156, input_shape=x_train.shape[1:], return_sequences=True))
model.add(GRU(128, return_sequences=True))
model.add(GRU(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(77))

model.compile('adam', loss='mae', metrics='mse')
model.fit(x_train, y_train, epochs=1, batch_size=32)
print(model.summary())
model.save('/Git/gernby-test.h5')
