import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import L2

(trainX, trainY), (testX, testY) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(trainX[0])

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['acc'])

trainX = trainX / 255.0
testX = testX / 255.0
model.fit(trainX, trainY, epochs=10, verbose=1, validation_data=(testX, testY))

model2 = Sequential()
model2.add(Flatten(input_shape=(28, 28)))
model2.add(Dense(256, activation="relu", kernel_regularizer=L2(0.01)))
model2.add(Dense(128, activation='relu', kernel_regularizer=L2(0.01)))
model2.add(Dense(10, activation='softmax'))

model2.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['acc'])
model2.fit(trainX, trainY, epochs=10, verbose=1, validation_data=(testX, testY))

