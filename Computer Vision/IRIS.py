import tensorflow_datasets as tfds
data = tfds.load("iris", split="train[:80%]")

import tensorflow as tf
num_classes = 3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

def preprocess(features):
  # Return features and one-hot encoded labels
  f = features['features']
  l = features['label']

  l = tf.one_hot(l, depth=num_classes)
  return f,l

def solution_model():
  train_dataset = data.map(preprocess).batch(10)

  for item in train_dataset:
    print(item[1])
  model = Sequential()
  model.add(Flatten(input_shape=(4, 1)))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(3, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

  model.fit(train_dataset, epochs=10, verbose=1)

  return model

solution_model()

data = data_gen.flow_from_directory(
      data_folder,
      target_size=IMG_SIZE,
      batch_size=BATCH_SIZE,
      class_mode="binary",
  )

IMG_SIZE = (224, 224)
