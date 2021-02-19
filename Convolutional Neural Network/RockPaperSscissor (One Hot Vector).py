import tensorflow as tf
import tensorflow_datasets as tfds

def my_one_hot(feature, label):
  return feature, tf.one_hot(label, depth=3)

data = tfds.load('rock_paper_scissors', split='train', as_supervised=True)
val_data = tfds.load('rock_paper_scissors', split='test', as_supervised=True)

list(data)[0]

data = data.map(my_one_hot)
val_data = val_data.map(my_one_hot)

train_batches = data.shuffle(100).batch(10)
validation_batches = val_data.batch(32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(train_batches, epochs=10, validation_data=validation_batches, validation_steps=1)


