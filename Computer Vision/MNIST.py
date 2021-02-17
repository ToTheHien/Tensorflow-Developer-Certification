import tensorflow as tf
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(training_images[0])

training_images.shape

training_images = training_images/255.0
training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
test_images = test_images/255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics='acc')

model.summary()
model.fit(
    training_images,
    training_labels,
    epochs=6,
    validation_data=(test_images, test_labels)
)
