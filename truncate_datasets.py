#!/usr/bin/python3
import os
from shutil import copyfile

desired_classes = [
	'balcony-interior',
	'bathroom',
	'bedchamber',
	'bedroom',
	'cafeteria',
	'childs_room',
	'closet',
	'corridor',
	'dining_hall',
	'dining_room',
	'dorm_room',
	'elevator-door',
	'garage-indoor',
	'gymnasium-indoor',
	'home_office',
	'home_theater',
	'kitchen',
	'library-indoor',
	'living_room',
	'nursery',
	'nursing_home',
	'pantry',
	'playroom',
	'porch',
	'recreation_room',
	'shower',
	'storage_room',
	'television_room',
	'utility_room',
]

print('Classes to copy: ' + str(len(desired_classes)))

for room in desired_classes:
	print(str(desired_classes.index(room)) + '. ' + room + ':')

	print('\tPreparing training samples...')
	try:
		os.mkdir('train_final/' + room)
	except FileExistsError:
		pass
	i = 0
	for file in os.listdir('train/' + room):
		if i % 10 == 0:
			copyfile('train/' + room + '/' + file, 'train_final/' + room + '/' + file)
		i += 1

	print('\tPreparing testing samples...')
	try:
		os.mkdir('test_final/' + room)
	except FileExistsError:
		pass
	for file in os.listdir('val/' + room):
		copyfile('val/' + room + '/' + file, 'test_final/' + room + '/' + file)
