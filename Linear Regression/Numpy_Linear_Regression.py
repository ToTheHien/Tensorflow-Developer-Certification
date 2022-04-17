  
xs = [2, 7, 9, 3, 10, 6, 1, 8]
ys = [13, 35, 41, 19, 45, 28, 10, 55]

import tensorflow as tf
import numpy as np

xs = np.array(xs)
ys = np.array(ys)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_shape=[1]))
model.summary()

model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=1000)

model.predict([10, 50])
