import shutil
import os
import random
import numpy as np
import argparse
import subprocess
import Augmentor
import warnings
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders for each of the recording runs with A and B subfolders)')
parser.add_argument('--augment', action='store_true', help='whether to augment the dataset or not')

path = parser.parse_args().dataroot
augment = parser.parse_args().augment

if 'augmented' not in path and augment:
	warnings.warn("The dataset path does not contain the word 'augmented'. If you are trying to augment the dataset, consider adding 'augment' or some other flag to the name to be able to distinguish it.")
	user_input = input("Are you sure you want to proceed? (y/n): ")
	if user_input == 'n':
		sys.exit()

## Moving all subsets of dataset into one folder
print('Consolidating dataset into one folder...')
folders = [item for item in os.listdir(path) if item[0] != '.' and item != 'A' and item != 'B' and os.path.isdir(os.path.join(path, item))]

for folder in folders:
    for subfolder in ['A', 'B']:

        if subfolder not in os.listdir(path):
            os.mkdir(path+'/'+subfolder)

        files = [item for item in os.listdir(path+'/'+folder+'/'+subfolder) if item[0] != '.']

        for file in files:
            os.rename(path+'/'+folder+'/'+subfolder+'/'+file, path+'/'+subfolder+'/'+file)

    shutil.rmtree(path+'/'+folder)



## Creating train/test/val splits
print('Creating train/test/val splits...')
all_files = os.listdir(path + '/A')

random.seed(0)

subset_train = random.sample(all_files, round(len(all_files)*0.8))

remaining = list(set(all_files) - set(subset_train))

subset_val = random.sample(remaining, round(len(remaining)*0.5))

subset_test = list(set(remaining) - set(subset_val))

subfolders = ['/train', '/val', '/test']

for folder in ['/A', '/B']:
	for subfolder, subset in zip (subfolders, [subset_train, subset_val, subset_test]):
		try:
			os.listdir(path + folder + subfolder)
		except:
			os.mkdir(path + folder + subfolder)

		for file in subset:
			file_name = os.path.join(path + folder, file)
			new_file_name = os.path.join(path + folder + subfolder, file)
			os.rename(file_name, new_file_name) # Moving the files from folder (A or B) into subfolder (train, val, test)
			

## Augmenting the dataset
if augment:
	print('Augmenting the dataset...')
	p = Augmentor.Pipeline(os.path.join(path, 'B/train'))

	# Adding the ground truth data to be transformed
	p.ground_truth(os.path.join(path, 'A/train'))
	# Adding operations to the pipeline
	p.flip_left_right(probability=1)
	p.zoom_random(probability=0.3, percentage_area=0.8)
	p.process() # Transforming each image exactly once

	# Moving the generated images out of the output folder to rejoin them with the original images
	augmentor_subfolder = 'B/train/output'

	for image in [item for item in os.listdir(os.path.join(path, augmentor_subfolder)) if not os.path.isdir(os.path.join(path, augmentor_subfolder, item))]:

		image_new = image.rsplit('_', 1)[0]

		if 'groundtruth' in image_new:
			image_new = image_new.replace('_groundtruth_(1)_train_', '').replace('.bmp', '_augmented.bmp')
			os.rename(os.path.join(path, augmentor_subfolder, image), os.path.join(path, 'A/train', image_new))
		else:
			image_new = image_new.replace('train_original_', '').replace('.bmp', '_augmented.bmp')
			os.rename(os.path.join(path, augmentor_subfolder, image), os.path.join(path, 'B/train', image_new))

# Combining the A and B images into paired images
combination_command = f"python datasets/combine_A_and_B.py --fold_A {path}/A --fold_B {path}/B --fold_AB {path} --no_multiprocessing"

print('Combining the A and B folders...')
subprocess.call(combination_command, shell=True)

print("Dataset preprocessing complete.")
if augment:
	print(f"The dataset contains {len(subset_train)*2} training images, {len(subset_val)} validation images, and {len(subset_test)} test images after augmentation.")

else:
	print(f"The dataset contains {len(subset_train)} training images, {len(subset_val)} validation images, and {len(subset_test)} test images.")