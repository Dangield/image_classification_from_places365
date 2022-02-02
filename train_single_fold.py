#!/usr/bin/python3
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import sys

# from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model

fold = sys.argv[1]
train_dir = pathlib.Path('train_final_' + str(fold) + '/')
val_dir = pathlib.Path('val_final_' + str(fold) + '/')
batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	train_dir,
	image_size=(img_height, img_width),
	batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	val_dir,
	image_size=(img_height, img_width),
	batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

data_augmentation = Sequential(
	[
		layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
		layers.RandomRotation(0.1),
		layers.RandomZoom(0.1),
	]
)

augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
augmented_val_ds = val_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32)
augmented_val_ds = augmented_val_ds.prefetch(buffer_size=32)

base_model = tf.keras.applications.MobileNet(input_shape = (img_height, img_width, 3), include_top=False)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Reshape((1, 1, 1024))(x)
x = layers.Dropout(0.001)(x)
x = layers.Conv2D(num_classes, 1)(x)
x = layers.Reshape((num_classes,))(x)
preds = layers.Activation('softmax')(x)  # final layer with softmax activation
model = Model(inputs = base_model.input, outputs = preds)

model.compile(
	optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(),
	metrics=['accuracy'])

epochs = 50
print('\nTraining:\n')
history = model.fit(
	augmented_train_ds,
	validation_data=augmented_val_ds,
	epochs=epochs,
	callbacks=[
		tf.keras.callbacks.ModelCheckpoint('my_models/mobilenet_50_' + str(fold) + '/save_{epoch}', save_weights_only = True),
		tf.keras.callbacks.TensorBoard(log_dir = 'my_models/mobilenet_50_' + str(fold) + '/logs', update_freq = 'batch'),
	],
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('my_models/mobilenet_50_' + str(fold) + '/training_plot.eps')
plt.show()
