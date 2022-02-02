#!/usr/bin/python3
import os
from shutil import copyfile
import numpy as np

n_folds = 5
for f in range(n_folds):
	print('Creating samples for fold: ' + str(f))
	os.mkdir('train_final_' + str(f))
	os.mkdir('val_final_' + str(f))
	for folder in os.listdir('train_final'):
		print('\tGenerating class: ' + folder)
		folds = np.array_split(os.listdir('train_final/' + folder), n_folds)
		os.mkdir('train_final_' + str(f) + '/' + folder)
		os.mkdir('val_final_' + str(f) + '/' + folder)
		for i in range(n_folds):
			if i == f:
				print('\t\tMoving subset ' + str(i) + ' to validation.')
			else:
				print('\t\tMoving subset ' + str(i) + ' to training.')
			for image in folds[i]:
				if i == f:
					copyfile('train_final/' + folder + '/' + image, 'val_final_' + str(f) + '/' + folder + '/' + image)
				else:
					copyfile('train_final/' + folder + '/' + image, 'train_final_' + str(f) + '/' + folder + '/' + image)
