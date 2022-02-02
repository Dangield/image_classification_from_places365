#!/usr/bin/python3
import tensorflow as tf
import pathlib
import sys

# from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np

fold = sys.argv[1]
epoch = sys.argv[2]
test_dir = pathlib.Path('test_final')
batch_size = 1
img_height = 224
img_width = 224

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
	test_dir,
	image_size=(img_height, img_width),
	batch_size=batch_size
)

class_names = test_ds.class_names
num_classes = len(class_names)

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

model.load_weights('my_models/mobilenet_50_' + fold + '/save_' + epoch)

print('\nEvaluation for testing dataset:\n')
score = model.evaluate(test_ds)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# for images, labels in test_ds.take(1):
# 	for i in range(10):
# 		predictions = model.predict(tf.expand_dims(images[i], 0))
# 		print("Original class of the image is: \t{}\n".format(class_names[labels[i]]))
# 		for j in range(5):
# 			score = predictions[0]
# 			print("\tThis image belongs to\t{} with a {:.3f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
# 			score[np.argmax(score)] = 0

top5 = 0
for images, labels in test_ds:
	predictions = model.predict(tf.expand_dims(images[0], 0))
	print("Original class is: \t{}".format(class_names[labels[0]]))
	for j in range(5):
		score = predictions[0]
		if class_names[labels[0]] == class_names[np.argmax(score)]:
			top5 += 1
		print("\tThis image belongs to\t{} with a {:.3f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
		score[np.argmax(score)] = 0
print("Top 5 accuracy: " + str(top5/len(test_ds)))
