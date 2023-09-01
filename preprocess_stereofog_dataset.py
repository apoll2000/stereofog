import shutil
import os
import random
import numpy as np
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders for each of the recording runs with A and B subfolders)')

path = parser.parse_args().dataroot

## Moving all subsets of dataset into one folder
folders = [item for item in os.listdir(path) if item[0] != '.' and item != 'A' and item != 'B']

for folder in folders:
    for subfolder in ['A', 'B']:

        if subfolder not in os.listdir(path):
            os.mkdir(path+'/'+subfolder)

        files = [item for item in os.listdir(path+'/'+folder+'/'+subfolder) if item[0] != '.']

        for file in files:
            os.rename(path+'/'+folder+'/'+subfolder+'/'+file, path+'/'+subfolder+'/'+file)

    shutil.rmtree(path+'/'+folder)



## Creating train/test/val splits
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
			


combination_command = "python datasets/combine_A_and_B.py --fold_A {path}/A --fold_B {path}/B --fold_AB {path}"

subprocess.call(combination_command, shell=True)

print("Dataset preprocessing complete.")
